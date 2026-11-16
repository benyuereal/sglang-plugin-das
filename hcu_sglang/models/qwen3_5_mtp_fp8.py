"""MTP draft lm_head in fp8 (e4m3), served by the hbird fp8 decode tiles.

The draft head is the widest GEMM of a decode step (248320 x 5120); its bf16
copy streams ~2.37 GiB per call. The fp8 copy plus the hbird tiles take it from
1320 to 792 us at M=45. The proxy replaces only the draft model's lm_head and
delegates everything but quant_method, so the target head stays bf16.
"""

import logging
import os
import time

import torch

from hcu_sglang.constants import (
    AMAX_EPS,
    BYTES_PER_GIB,
    MTP_LMHEAD_QUANT_BLOCK,
    QWEN3_5_MTP_HOOK_BASE,
)

logger = logging.getLogger(__name__)

_MTP_LMHEAD_FP8_E4M3_MAX = 448.0
_MTP_HBIRD_MAX_M_DEFAULT = 192


def _mtp_lmhead_fp8_enabled() -> bool:
    value = os.environ.get("SGLANG_MTP_LMHEAD_FP8", "0")
    return value.strip().lower() not in ("", "0", "false", "off")


def _mtp_hbird_ops():
    try:
        import hbird_ops

        return hbird_ops
    except ImportError:
        import sys

        root = os.environ.get("HBIRD_GEMM_HOME") or ""
        path = os.path.join(root, "plugins")
        if not root or not os.path.isdir(path):
            raise
        if path not in sys.path:
            sys.path.insert(0, path)
        import hbird_ops

        return hbird_ops


class _MTPDraftLMHeadFP8Method:
    def __init__(self, base, weight_fp8, weight_scale, hbird_gemm, hbird_max_m):
        self.base = base
        self.weight_fp8 = weight_fp8
        self.weight_scale = weight_scale
        self.hbird_gemm = hbird_gemm
        self.hbird_max_m = hbird_max_m
        # One buffer for every decode row count: a per-call allocation lands in
        # the CUDA-graph pool once per bs bucket (~0.4 GiB over bs 1..40) and
        # that is what pushed verify into OOM.
        self.out = torch.empty(
            (hbird_max_m, weight_fp8.shape[0]),
            dtype=torch.bfloat16,
            device=weight_fp8.device,
        )

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        assert bias is None, "draft fp8 lm_head does not support a bias"
        if x.shape[0] > self.hbird_max_m:
            # ParallelLMHead.forward raises by design (the head's weight belongs
            # to the sampler), so wide rows go through the weight directly.
            return torch.matmul(x, self.base.weight.t())
        xf = x.to(torch.float32)
        xs = (
            xf.abs().amax(dim=-1, keepdim=True).clamp_min(AMAX_EPS)
            / _MTP_LMHEAD_FP8_E4M3_MAX
        )
        xq = (xf / xs).clamp(
            -_MTP_LMHEAD_FP8_E4M3_MAX, _MTP_LMHEAD_FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn)
        out = self.out[: x.shape[0]]
        self.hbird_gemm(
            xq,
            self.weight_fp8,
            xs,
            self.weight_scale,
            out=out,
            prepare_shape=False,
        )
        return out


class _MTPDraftLMHeadFP8Proxy:
    def __init__(self, base, method: _MTPDraftLMHeadFP8Method) -> None:
        self._base = base
        self.quant_method = method

    def __getattr__(self, name):
        if name in ("_base", "set_lora", "apply_lora"):
            raise AttributeError(name)
        return getattr(self._base, name)

    @property
    def weight(self) -> torch.Tensor:
        return self._base.weight


def _mtp_quantize_lmhead_rows(weight: torch.Tensor):
    vocab, hidden = weight.shape
    weight_fp8 = torch.empty(
        (vocab, hidden), dtype=torch.float8_e4m3fn, device=weight.device
    )
    weight_scale = torch.empty((vocab, 1), dtype=torch.float32, device=weight.device)
    block = MTP_LMHEAD_QUANT_BLOCK
    for i in range(0, vocab, block):
        j = min(i + block, vocab)
        wf = weight[i:j].to(torch.float32)
        scale = (
            wf.abs().amax(dim=-1, keepdim=True).clamp_min(AMAX_EPS)
            / _MTP_LMHEAD_FP8_E4M3_MAX
        )
        weight_fp8[i:j] = (wf / scale).clamp(
            -_MTP_LMHEAD_FP8_E4M3_MAX, _MTP_LMHEAD_FP8_E4M3_MAX
        ).to(torch.float8_e4m3fn)
        weight_scale[i:j] = scale
        del wf
    return weight_fp8, weight_scale


def _quantize_draft_lm_head(weight: torch.Tensor):
    try:
        weight_fp8, weight_scale = _mtp_quantize_lmhead_rows(weight)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - optional path
        logger.warning(
            "[mtp_draft_lmhead_fp8] disabling, quantization failed: %s", exc
        )
        return None
    return weight_fp8, weight_scale


def _hbird_supported_ms(ops, *, vocab: int, hidden: int):
    ms = []
    for m in range(1, _MTP_HBIRD_MAX_M_DEFAULT + 1):
        if ops.guard(m, vocab, hidden) != 0:
            break
        ms.append(m)
    if not ms:
        raise RuntimeError("guard refuses every M")
    return ms


def _prepare_hbird_draft_head(weight_fp8: torch.Tensor):
    try:
        ops = _mtp_hbird_ops()
        vocab, hidden = weight_fp8.shape
        device = weight_fp8.device
        ms = _hbird_supported_ms(ops, vocab=vocab, hidden=hidden)
        started = time.perf_counter()
        ops.prepare_weight(weight_fp8)
        for m in ms:
            ops.prepare(m, vocab, hidden, device)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - optional path
        logger.info(
            "[mtp_draft_lmhead_fp8] hbird fp8 unavailable (%s); "
            "draft head stays bf16",
            exc,
        )
        return None, 0
    logger.info(
        "[mtp_draft_lmhead_fp8] draft head on hbird fp8: packed B %.2f GiB, "
        "%d M workspaces (%d..%d), %.2fs",
        weight_fp8.numel() / BYTES_PER_GIB,
        len(ms),
        ms[0],
        ms[-1],
        time.perf_counter() - started,
    )
    return ops.hbird_fp8_gemm, ms[-1]


def _should_use_fp8_draft_head(model) -> bool:
    """Cheap gates first: a captured stream cannot allocate, and the proxy must
    not wrap itself."""
    if getattr(model, "_mtp_fp8_installed", False):
        return False
    if not _mtp_lmhead_fp8_enabled():
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None or weight.dim() != 2 or not weight.is_cuda:
        return False
    if weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    if isinstance(head, _MTPDraftLMHeadFP8Proxy):
        return False
    logits_processor = getattr(model, "logits_processor", None)
    return not (
        logits_processor is not None
        and getattr(logits_processor, "use_fp32_lm_head", False)
    )


def _install_fp8_draft_head(
    *,
    model,
    head,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    hbird_gemm,
    hbird_max_m: int,
) -> None:
    method = _MTPDraftLMHeadFP8Method(
        head, weight_fp8, weight_scale, hbird_gemm, hbird_max_m
    )
    object.__setattr__(model, "lm_head", _MTPDraftLMHeadFP8Proxy(head, method))


def install(model) -> bool:
    if not _should_use_fp8_draft_head(model):
        return False
    head = model.lm_head
    weight = head.weight
    started = time.perf_counter()
    quantized = _quantize_draft_lm_head(weight.detach())
    if quantized is None:
        return False
    weight_fp8, weight_scale = quantized
    hbird_gemm, hbird_max_m = _prepare_hbird_draft_head(weight_fp8)
    if hbird_gemm is None:
        return False
    _install_fp8_draft_head(
        model=model,
        head=head,
        weight_fp8=weight_fp8,
        weight_scale=weight_scale,
        hbird_gemm=hbird_gemm,
        hbird_max_m=hbird_max_m,
    )
    model._mtp_fp8_installed = True
    logger.info(
        "[mtp_draft_lmhead_fp8] draft head %d x %d -> e4m3, %.2fs, "
        "%.2f -> %.2f GiB, hbird_max_m=%d",
        weight.shape[0],
        weight.shape[1],
        time.perf_counter() - started,
        weight.numel() * weight.element_size() / BYTES_PER_GIB,
        (weight_fp8.numel() + weight_scale.numel() * weight_scale.element_size())
        / BYTES_PER_GIB,
        hbird_max_m,
    )
    return True


def _drop_proxy(model) -> None:
    try:
        if isinstance(getattr(model, "lm_head", None), _MTPDraftLMHeadFP8Proxy):
            object.__delattr__(model, "lm_head")
    except Exception:  # noqa: BLE001
        pass


def _around_set_lm_head(original, self, target_lm_head):
    out = original(self, target_lm_head)
    try:
        _drop_proxy(self)
        self._mtp_fp8_installed = False
        install(self)
    except Exception:  # noqa: BLE001 - optional path
        logger.warning("[mtp_draft_lmhead_fp8] install failed", exc_info=True)
    return out


def _around_set_embed_and_head(original, self, embed, head):
    _drop_proxy(self)
    self._mtp_fp8_installed = False
    out = original(self, embed, head)
    try:
        install(self)
    except Exception:  # noqa: BLE001
        logger.warning("[mtp_draft_lmhead_fp8] reinstall failed", exc_info=True)
    return out


_AROUND_HOOKS = (
    ("set_lm_head_from_target", _around_set_lm_head),
    ("set_embed_and_head", _around_set_embed_and_head),
)


def register() -> bool:
    if not _mtp_lmhead_fp8_enabled():
        return False
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for method, hook in _AROUND_HOOKS:
        HookRegistry.register(QWEN3_5_MTP_HOOK_BASE + method, hook, HookType.AROUND)
    logger.info("[mtp_draft_lmhead_fp8] draft head fp8 hooks registered")
    return True
