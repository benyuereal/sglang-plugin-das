"""Install the fp8 draft head on the MTP model as it loads.

The quantized head, its hbird tiles and the proxy that stands in for it live in
:mod:`hcu_sglang.models.mtp_lmhead_quant`. This module owns the model-side
question only: when the head should be replaced, and how the two head-setting
hooks put it back after the loader rewrites the head.
"""

import logging
import os
import time

import torch
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from hcu_sglang.constants import BYTES_PER_GIB
from hcu_sglang.layers.gemm import quant_lm_head
from hcu_sglang.models import mtp_lmhead_quant as mtp_head

# Prefix of the Qwen3.5 MTP setters the draft-head hooks patch.
QWEN3_5_MTP_HOOK_BASE = "sglang.srt.models.qwen3_5_mtp.Qwen3_5ForCausalLMMTP."

logger = logging.getLogger(__name__)

def _mtp_lmhead_fp8_enabled() -> bool:
    value = os.environ.get("SGLANG_MTP_LMHEAD_FP8", "0")
    return value.strip().lower() not in ("", "0", "false", "off")

def _should_use_fp8_draft_head(model) -> bool:
    """Cheap gates first: a captured stream cannot allocate, and the proxy must
    not be wrapped twice."""
    if getattr(model, "_mtp_fp8_installed", False):
        return False
    if not _mtp_lmhead_fp8_enabled():
        return False
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    head = getattr(model, "lm_head", None)
    if isinstance(head, mtp_head.DraftHeadQuantProxy):
        return False
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return False
    if not weight.is_cuda or weight.dtype not in (torch.bfloat16, torch.float16):
        return False
    logits_processor = getattr(model, "logits_processor", None)
    if logits_processor is not None and getattr(
        logits_processor, "use_fp32_lm_head", False
    ):
        return False
    return True

def _install_draft_head(*, model, head, weight_q, weight_scale, gemm, max_m, scheme):
    method = mtp_head.DraftHeadQuantMethod(
        head, weight_q, weight_scale, gemm, max_m, scheme
    )
    object.__setattr__(model, "lm_head", mtp_head.DraftHeadQuantProxy(head, method))

def install(model) -> bool:
    if not _should_use_fp8_draft_head(model):
        return False
    head = model.lm_head
    weight = head.weight
    started = time.perf_counter()
    scheme = mtp_head.SCHEME
    quantized = mtp_head.quantize(weight.detach(), scheme)
    if quantized is None:
        return False
    weight_q, weight_scale = quantized
    gemm, max_m = quant_lm_head.prepare(weight_q)
    if gemm is None:
        return False
    _install_draft_head(
        model=model,
        head=head,
        weight_q=weight_q,
        weight_scale=weight_scale,
        gemm=gemm,
        max_m=max_m,
        scheme=scheme,
    )
    model._mtp_fp8_installed = True
    logger.info(
        "[mtp_draft_lmhead_fp8] draft head %d x %d -> e4m3, %.2fs, "
        "%.2f -> %.2f GiB, max_m=%d",
        weight.shape[0],
        weight.shape[1],
        time.perf_counter() - started,
        weight.numel() * weight.element_size() / BYTES_PER_GIB,
        (weight_q.numel() + weight_scale.numel() * weight_scale.element_size())
        / BYTES_PER_GIB,
        max_m,
    )
    return True

def _drop_proxy(model) -> None:
    try:
        if isinstance(getattr(model, "lm_head", None), mtp_head.DraftHeadQuantProxy):
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
    for method, hook in _AROUND_HOOKS:
        HookRegistry.register(QWEN3_5_MTP_HOOK_BASE + method, hook, HookType.AROUND)
    logger.info("[mtp_draft_lmhead_fp8] draft head fp8 hooks registered")
    return True
