"""Qwen3.5 model hooks for HCU fused RMSNorm + RoPE KV-store."""

import logging

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from hcu_sglang.models.fused_rotary import (
    fused_rms_rotary_kv_store,
    use_fused_rms_rotary,
)
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.utils.module_patch import get_original_attr

logger = logging.getLogger(__name__)

_TARGET = "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention"
_MODULE = "sglang.srt.models.qwen3_5"
_GDN_INIT_TARGET = _MODULE + ".Qwen3_5GatedDeltaNet.__init__"


def register() -> None:
    logger.info("[hcu_sglang] register AROUND hook: %s", _TARGET)
    HookRegistry.register(_TARGET, replacement_self_attention, HookType.AROUND)
    HookRegistry.register(_GDN_INIT_TARGET, around_gdn_init, HookType.AROUND)


def replacement_self_attention(original, self, positions, hidden_states, forward_batch):
    return original(self, positions, hidden_states, forward_batch)


def around_gdn_init(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if is_hcu():
        self.alt_stream = None
    return result
