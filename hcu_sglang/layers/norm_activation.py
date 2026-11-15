"""HCU norm and activation adaptations."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton ships with every HCU torch
    _HAS_TRITON = False

_SILU_BLOCK = 8192
_SILU_WARPS = 8
# One program per row with the whole row in registers; the gate below refuses
# rows wider than this so a call never needs a second pass.
# Load-bearing for the sha, not just perf: the one-pass kernel computes the
# variance as a single tl.sum(t * t) over the whole row, which is bit-identical
# to the two-pass vendor kernel's for-off accumulate only while one BLOCK
# covers the row. A wider row would split the sum across blocks and change its
# order, so raising this limit breaks bit-exactness, not just adds a pass.
_ADD_RMS_MAX_N = 8192
# One-strike: a raising fast path is disabled for the process and the caller
# falls back to the vendor kernel, so an opt-in knob can never wedge serving.
_FAST_FAILED = {"silu": False, "add_rms": False}


def _enabled(name: str) -> bool:
    from sglang.srt.environ import envs

    field = getattr(envs, name, None)
    if field is not None:
        return bool(field.get())
    from sglang.srt.utils import get_bool_env_var

    return get_bool_env_var(name, default="false")


if _HAS_TRITON:

    @triton.jit
    def _silu_mul_kernel(X, OUT, T, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        cols = pid * BLOCK + tl.arange(0, BLOCK)
        mask = cols < T
        row = cols // N
        col = cols % N
        src = row * (2 * N) + col
        g = tl.load(X + src, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(X + src + N, mask=mask, other=0.0).to(tl.float32)
        # Vendor semantics round silu to the activation dtype before the
        # multiply; a single fp32 rounding changes most elements by one ulp.
        silu = (g * tl.sigmoid(g)).to(tl.bfloat16)
        tl.store(OUT + cols, (silu.to(tl.float32) * u).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _gemma_add_rms_1pass_kernel(X, RES, W, stride, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(RES + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        t = x + r
        var = tl.sum(t * t, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        stored = t.to(RES.dtype.element_ty)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
        # Bit-exact with the two-pass vendor kernel: variance from the fp32
        # sum, but the normalized numerator from the stored residual dtype.
        y = stored.to(tl.float32) * rstd * w
        tl.store(RES + row * stride + cols, stored, mask=mask)
        tl.store(X + row * stride + cols, y.to(X.dtype.element_ty), mask=mask)


def _fast_silu_and_mul(x: torch.Tensor) -> Optional[torch.Tensor]:
    if _FAST_FAILED["silu"] or not (
        _HAS_TRITON
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.dim() >= 1
        and x.shape[-1] >= 2
        and x.shape[-1] % 2 == 0
        and x.numel() > 0
    ):
        return None
    n = x.shape[-1] // 2
    total = x.numel() // 2
    out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    try:
        _silu_mul_kernel[(triton.cdiv(total, _SILU_BLOCK),)](
            x, out, total, n, _SILU_BLOCK, num_warps=_SILU_WARPS, num_stages=1
        )
    except Exception:
        _FAST_FAILED["silu"] = True
        logger.exception("[hcu-norm-act] fast silu disabled; using lightop")
        return None
    return out


def _fast_gemma_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    n = int(x.shape[-1]) if x.dim() else 0
    if _FAST_FAILED["add_rms"] or not (
        _HAS_TRITON
        and x.is_cuda
        and residual.is_cuda
        and x.dtype == residual.dtype
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.shape == residual.shape
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.dim() == 1
        and weight.numel() == n
        and 0 < n <= _ADD_RMS_MAX_N
        and x.numel() > 0
    ):
        return None
    block = triton.next_power_of_2(n)
    warps = 16 if block >= 4096 else (8 if block >= 1024 else 4)
    x_2d = x.view(-1, n)
    res_2d = residual.view(-1, n)
    try:
        _gemma_add_rms_1pass_kernel[(x_2d.shape[0],)](
            x_2d,
            res_2d,
            weight,
            x_2d.stride(0),
            n,
            eps,
            BLOCK=block,
            num_warps=warps,
        )
    except Exception:
        _FAST_FAILED["add_rms"] = True
        logger.exception("[hcu-norm-act] fast add-rmsnorm disabled; using vendor")
        return None
    return x, residual


def after_silu_and_mul_init(result: Any, layer, *args, **kwargs) -> Any:
    if is_hcu() and getattr(layer._forward_method, "__name__", "") == "forward_aiter":
        layer._forward_method = layer.forward_cuda
    return result


def around_silu_and_mul_forward_cuda(
    original: Callable[..., torch.Tensor], layer, x: torch.Tensor
) -> torch.Tensor:
    if not is_hcu():
        return original(layer, x)

    if _enabled("SGLANG_HCU_FUSE_SILU"):
        fast = _fast_silu_and_mul(x)
        if fast is not None:
            return fast

    from lightop import fuse_silu_and_mul

    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    fuse_silu_and_mul(x, output)
    return output


def around_gemma_forward_cuda(
    original: Callable[..., Any],
    layer,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Any:
    if not is_hcu():
        return original(layer, x, residual, post_residual_addition)

    if (
        residual is not None
        and post_residual_addition is None
        and _enabled("SGLANG_HCU_FUSE_ADD_RMS")
    ):
        fast = _fast_gemma_add_rmsnorm(
            x, residual, layer.weight.data, layer.variance_epsilon
        )
        if fast is not None:
            return fast

    return original(layer, x, residual, post_residual_addition)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.layers.activation.SiluAndMul"
    HookRegistry.register(
        f"{target}.__init__",
        after_silu_and_mul_init,
        HookType.AFTER,
    )
    HookRegistry.register(
        f"{target}.forward_cuda",
        around_silu_and_mul_forward_cuda,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.layernorm.GemmaRMSNorm.forward_cuda",
        around_gemma_forward_cuda,
        HookType.AROUND,
    )
