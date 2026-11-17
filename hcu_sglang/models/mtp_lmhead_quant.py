"""Quantized MTP draft lm_head.

The draft head is the widest GEMM of a decode step (248320 x 5120); its bf16
copy streams ~2.37 GiB per call, and the quantized copy plus the packed tiles
take it from 1320 to 792 us at M=45. Only the draft model's lm_head is
replaced, so the target head stays bf16.

A scheme owns the dtype and the per-row quantization only; the GEMM backend
that serves the result lives in :mod:`hcu_sglang.layers.gemm.quant_lm_head`.
``fp8_e4m3`` is the only scheme wired today; a second one would replace
``SCHEME`` here, not add an environment variable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import torch

# Row block the MTP draft head is quantized in.
MTP_LMHEAD_QUANT_BLOCK = 4096

# amax floor used when scaling fp8 activations and weights.
AMAX_EPS = 1e-8

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadQuantScheme:
    """One way of storing the draft head, plus how to prepare it."""

    name: str
    dtype: torch.dtype
    amax: float
    block: int

    def quantize_rows(self, weight: torch.Tensor):
        vocab, hidden = weight.shape
        weight_q = torch.empty((vocab, hidden), dtype=self.dtype, device=weight.device)
        weight_scale = torch.empty(
            (vocab, 1), dtype=torch.float32, device=weight.device
        )
        for i in range(0, vocab, self.block):
            j = min(i + self.block, vocab)
            wf = weight[i:j].to(torch.float32)
            scale = (
                wf.abs().amax(dim=-1, keepdim=True).clamp_min(AMAX_EPS) / self.amax
            )
            weight_q[i:j] = (wf / scale).clamp(-self.amax, self.amax).to(self.dtype)
            weight_scale[i:j] = scale
            del wf
        return weight_q, weight_scale

    def quantize_activations(self, x: torch.Tensor):
        xf = x.to(torch.float32)
        scale = (
            xf.abs().amax(dim=-1, keepdim=True).clamp_min(AMAX_EPS) / self.amax
        )
        xq = (xf / scale).clamp(-self.amax, self.amax).to(self.dtype)
        return xq, scale

class DraftHeadQuantMethod:
    def __init__(self, base, weight_q, weight_scale, gemm, max_m, scheme):
        self.base = base
        self.weight_q = weight_q
        self.weight_scale = weight_scale
        self.gemm = gemm
        self.max_m = max_m
        self.scheme = scheme
        # One buffer for every decode row count: a per-call allocation lands in
        # the CUDA-graph pool once per bs bucket (~0.4 GiB over bs 1..40) and
        # that is what pushed verify into OOM.
        self.out = torch.empty(
            (max_m, weight_q.shape[0]),
            dtype=torch.bfloat16,
            device=weight_q.device,
        )

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        assert bias is None, "quantized draft lm_head does not support a bias"
        if x.shape[0] > self.max_m:
            # ParallelLMHead.forward raises by design (the head's weight belongs
            # to the sampler), so wide rows go through the weight directly.
            return torch.matmul(x, self.base.weight.t())
        xq, xs = self.scheme.quantize_activations(x)
        out = self.out[: x.shape[0]]
        self.gemm(
            xq,
            self.weight_q,
            xs,
            self.weight_scale,
            out=out,
            prepare_shape=False,
        )
        return out

class DraftHeadQuantProxy:
    """Stands in for the draft head; only quant_method is replaced."""

    def __init__(self, base, method: DraftHeadQuantMethod) -> None:
        self._base = base
        self.quant_method = method

    def __getattr__(self, name):
        if name in ("_base", "set_lora", "apply_lora"):
            raise AttributeError(name)
        return getattr(self._base, name)

    @property
    def weight(self) -> torch.Tensor:
        return self._base.weight

SCHEME = HeadQuantScheme(
    name="fp8_e4m3",
    dtype=torch.float8_e4m3fn,
    amax=448.0,
    block=MTP_LMHEAD_QUANT_BLOCK,
)

def quantize(weight: torch.Tensor, scheme: HeadQuantScheme):
    """The quantized head and its per-row scale, or None to keep bf16."""
    try:
        weight_q, weight_scale = scheme.quantize_rows(weight)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - optional path
        logger.warning(
            "[mtp_draft_lmhead_quant] disabling, quantization failed: %s", exc
        )
        return None
    return weight_q, weight_scale

