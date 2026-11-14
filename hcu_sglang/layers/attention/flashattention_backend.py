"""FlashAttention hooks for HCU fused RMSNorm + RoPE KV-store paths."""

from __future__ import annotations

import logging
import sys
import types

import torch

from hcu_sglang.constants import FA3_MAX_DESCALE_HEAD_DIM

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from hcu_sglang.models.fused_rotary import has_fused_kv_store
from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)

_FORWARD_EXTEND_TARGET = (
    "sglang.srt.layers.attention.flashattention_backend."
    "FlashAttentionBackend.forward_extend"
)
_FORWARD_DECODE_TARGET = (
    "sglang.srt.layers.attention.flashattention_backend."
    "FlashAttentionBackend.forward_decode"
)
_INIT_TARGET = (
    "sglang.srt.layers.attention.flashattention_backend."
    "FlashAttentionBackend.__init__"
)
_PREPARE_TARGET = (
    "sglang.srt.layers.attention.flashattention_backend."
    "FlashAttentionBackend.prepare_paged_mha_query"
)


def register() -> None:
    hooks = (
        (_INIT_TARGET, around_init),
        ("sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.get_paged_mha_kv_cache", around_get_paged_mha_kv_cache),
        (_FORWARD_EXTEND_TARGET, around_forward_extend),
        (_FORWARD_DECODE_TARGET, around_forward_decode),
        (_PREPARE_TARGET, around_prepare_paged_mha_query),
    )
    for target, hook in hooks:
        logger.info("[hcu_sglang] register AROUND hook: %s", target)
        HookRegistry.register(target, hook, HookType.AROUND)
    from hcu_sglang.layers.attention import flashattention_interface

    HookRegistry.register(
        "sglang.kernels.ops.attention.flash_attention.flash_attn_with_kvcache",
        flashattention_interface.around_flash_attn_with_kvcache,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.kernels.ops.attention.flash_attention.flash_attn_varlen_func",
        flashattention_interface.around_flash_attn_varlen_func,
        HookType.AROUND,
    )


def around_init(original_fn, backend, *args, **kwargs):
    if not is_hcu():
        return original_fn(backend, *args, **kwargs)

    from hcu_sglang.layers.attention.flashattention_interface import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )

    module_name = "sgl_kernel.flash_attn"
    previous = sys.modules.get(module_name)
    compatibility_module = types.ModuleType(module_name)
    compatibility_module.flash_attn_varlen_func = flash_attn_varlen_func
    compatibility_module.flash_attn_with_kvcache = flash_attn_with_kvcache
    compatibility_module.get_scheduler_metadata = lambda *args, **kwargs: None
    sys.modules[module_name] = compatibility_module
    try:
        result = original_fn(backend, *args, **kwargs)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    backend.flash_attn_varlen_func = flash_attn_varlen_func
    backend.flash_attn_with_kvcache = flash_attn_with_kvcache
    backend._get_scheduler_metadata = None
    backend._get_fa_runtime_policy = None
    return result


def _should_skip_kv_store(args, kwargs) -> bool:
    if not is_hcu():
        return False

    # Method args:
    #   self, q, k, v, layer, forward_batch, save_kv_cache=True,
    #   q_rope=None, k_rope=None, sinks=None
    k = args[2] if len(args) > 2 else kwargs.get("k")
    if k is None:
        return False

    k_rope = args[8] if len(args) > 8 else kwargs.get("k_rope")
    if k_rope is not None:
        return False

    layer = args[4] if len(args) > 4 else kwargs.get("layer")
    forward_batch = args[5] if len(args) > 5 else kwargs.get("forward_batch")
    if layer is None or forward_batch is None:
        return False

    layer_id = getattr(layer, "layer_id", None)
    return layer_id is not None and has_fused_kv_store(forward_batch, layer_id)


def _call_without_kv_store(original_fn, *args, **kwargs):
    if not _should_skip_kv_store(args, kwargs):
        return original_fn(*args, **kwargs)

    if len(args) > 6:
        args = list(args)
        args[6] = False
        return original_fn(*args, **kwargs)

    kwargs = dict(kwargs)
    kwargs["save_kv_cache"] = False
    return original_fn(*args, **kwargs)


def around_forward_extend(original_fn, *args, **kwargs):
    if not is_hcu():
        return _call_without_kv_store(original_fn, *args, **kwargs)
    from hcu_sglang.layers.attention import flashattention_interface

    metadata = getattr(args[0], "forward_metadata", None) if args else None
    max_seq_len_k = getattr(metadata, "max_seq_len_k", None)
    previous = flashattention_interface._set_current_max_seq_len_k(max_seq_len_k)
    try:
        return _call_without_kv_store(original_fn, *args, **kwargs)
    finally:
        flashattention_interface._set_current_max_seq_len_k(previous)


def around_forward_decode(original_fn, *args, **kwargs):
    return _call_without_kv_store(original_fn, *args, **kwargs)


def around_get_paged_mha_kv_cache(original_fn, backend, layer, *, head_group_num=1):
    from hcu_sglang.mem_cache.hcu_mha_pool import HCUMHATokenToKVPool
    pool = backend.token_to_kv_pool
    hcu_pool = None
    if is_hcu():
        if isinstance(pool, HCUMHATokenToKVPool):
            hcu_pool = pool
        else:
            inner = getattr(pool, "full_kv_pool", None)
            if isinstance(inner, HCUMHATokenToKVPool):
                hcu_pool = inner
    if hcu_pool is None:
        return original_fn(backend, layer, head_group_num=head_group_num)
    key, value = pool.get_kv_buffer(layer.layer_id)
    page = backend.page_size
    assert tuple(key.shape[1:]) == (layer.tp_k_head_num, page, layer.head_dim)
    assert tuple(value.shape[1:]) == (layer.tp_v_head_num, layer.v_head_dim, page)
    if (head_group_num < 1 or layer.tp_k_head_num % head_group_num or
            layer.tp_v_head_num % head_group_num):
        raise ValueError('head_group_num must divide the K/V head counts')
    # These contiguous NHD-shaped carriers preserve HCU physical byte order.
    # The plugin FA interface restores K[pages,H,page,D]/V[pages,H,D,page]
    # immediately before the native HCU call; a permute here would corrupt it.
    return (key.view(-1, page, layer.tp_k_head_num // head_group_num, layer.head_dim),
            value.view(-1, page, layer.tp_v_head_num // head_group_num, layer.v_head_dim))


def _descale_tile(scale, shape, device):
    """Per-row descale tile, 1.0 when the model carries no scale."""
    if scale is None:
        return torch.ones(shape, dtype=torch.float32, device=device)
    try:
        return scale.expand(shape)
    except RuntimeError:  # stored as [1]
        return scale.reshape(-1)[:1].expand(shape)


def around_prepare_paged_mha_query(
    original,
    self,
    q,
    q_rope,
    k_rope,
    layer,
    *,
    logical_batch_size,
    kv_head_num,
    is_prefill,
):
    """Fill fp8 KV descales when the upstream path leaves them unset.

    0.5.18 fell back to the backend-level k_scale/v_scale; 0.5.22 only uses
    ``layer.k_scale``. The fa3 varlen entry requires q/k/v descales for an
    fp8 prefix decode, so fill the missing ones (1.0 when no scale is set).
    """
    out = original(
        self,
        q,
        q_rope,
        k_rope,
        layer,
        logical_batch_size=logical_batch_size,
        kv_head_num=kv_head_num,
        is_prefill=is_prefill,
    )
    if out[3] is not None or self.kv_cache_dtype_str == "auto":
        return out
    if (
        layer.head_dim > FA3_MAX_DESCALE_HEAD_DIM
        or logical_batch_size <= 0
        or kv_head_num <= 0
    ):
        return out
    shape = (int(logical_batch_size), int(kv_head_num))
    return (
        out[0],
        out[1],
        out[2],
        _descale_tile(layer.k_scale, shape, q.device),
        _descale_tile(layer.v_scale, shape, q.device),
    )
