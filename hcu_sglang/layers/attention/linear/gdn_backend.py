"""REPLACE hook implementations for HCU GDN backend paths."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from hcu_sglang.layers.attention.linear import gdn_source
from hcu_sglang.utils.module_patch import get_original_attr


logger = logging.getLogger(__name__)

_GDN_BACKEND_MODULE = "sglang.srt.layers.attention.linear.gdn_backend"


def register() -> None:
    _install_conv_hook()
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


def _install_conv_hook() -> None:
    """Route the upstream GDN conv call through the plugin dispatcher.

    sglang's own GDNAttnBackend.forward_extend resolves `causal_conv1d_update`
    from its module globals; replacing the name covers that path as well as the
    plugin's gdn_source, so the HIP conv kernel is used whichever forward wins.
    """
    import sglang.srt.layers.attention.linear.gdn_backend as upstream_gdn

    upstream_gdn.causal_conv1d_update = gdn_source.conv_update
    # gdn_source copied upstream's namespace at import, before this patch, so
    # point its own binding at the dispatcher as well.
    gdn_source.causal_conv1d_update = gdn_source.conv_update


def replacement_forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    if not gdn_source._is_hcu:
        original = get_original_attr(_GDN_BACKEND_MODULE, "GDNAttnBackend.forward_decode")
        return original(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
    return gdn_source.forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)


def replacement_forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    if not gdn_source._is_hcu:
        original = get_original_attr(_GDN_BACKEND_MODULE, "GDNAttnBackend.forward_extend")
        return original(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
    return gdn_source.forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)


def around_convolve_prefill(
    original, self, layer, forward_batch, mixed_qkv, conv_states, cache_indices
):
    if not gdn_source._is_hcu:
        return original(self, layer, forward_batch, mixed_qkv, conv_states,
                        cache_indices)
    return gdn_source._convolve_prefill(
        original, self, layer, forward_batch, mixed_qkv, conv_states, cache_indices
    )
