"""REPLACE hook implementations for HCU GDN backend paths."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

import torch

from sglang.srt.utils import is_cpu, is_npu
from hcu_sglang.utils.module_patch import get_original_attr


logger = logging.getLogger(__name__)

_GDN_BACKEND_MODULE = "sglang.srt.layers.attention.linear.gdn_backend"


def register() -> None:
    hooks = (
        (
            "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend.forward_decode",
            replacement_forward_decode,
        ),
        (
            "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend.forward_extend",
            replacement_forward_extend,
        ),
    )
    HookRegistry.register(
        f"{_GDN_BACKEND_MODULE}.GDNAttnBackend._convolve_prefill",
        around_convolve_prefill, HookType.AROUND,
    )
    for target, hook in hooks:
        logger.info("[hcu_sglang] register REPLACE hook: %s", target)
        HookRegistry.register(target, hook, HookType.REPLACE)


def _load_causal_conv1d_hcu():
    from causal_conv1d import causal_conv1d_fn_hcu
    from causal_conv1d.causal_conv1d_interface import (
        causal_conv1d_update as causal_conv1d_update_hcu,
    )

    return causal_conv1d_update_hcu, causal_conv1d_fn_hcu


def _load_sglang_gdn_symbols():
    try:
        from sglang.kernels.ops.attention.fla.fused_gdn_gating import (
            fused_gdn_gating,
        )
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_update,
        )
        from sglang.srt.mem_cache.memory_pool import MambaPool
    except ImportError:
        from sglang.srt.layers.attention.linear.gdn_backend import (
            MambaPool,
            fused_gdn_gating,
        )
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_update,
        )

    return MambaPool, fused_gdn_gating, causal_conv1d_update


def replacement_forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    from hcu_sglang.layers.attention.linear import gdn_source
    if not gdn_source._is_hcu:
        original = get_original_attr(_GDN_BACKEND_MODULE, "GDNAttnBackend.forward_decode")
        return original(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
    return gdn_source.forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)


def replacement_forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    from hcu_sglang.layers.attention.linear import gdn_source
    if not gdn_source._is_hcu:
        original = get_original_attr(_GDN_BACKEND_MODULE, "GDNAttnBackend.forward_extend")
        return original(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
    return gdn_source.forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)


def around_convolve_prefill(original, self, *args, **kwargs):
    from hcu_sglang.layers.attention.linear import gdn_source
    if not gdn_source._is_hcu:
        return original(self, *args, **kwargs)
    return gdn_source._convolve_prefill(self, *args, **kwargs)
