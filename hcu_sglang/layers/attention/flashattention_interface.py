"""HCU FlashAttention interface compatible with official SGLang call sites."""

from __future__ import annotations

from typing import Optional, Union

import torch

from hcu_sglang.utils.detection import is_hcu, is_hcu_native_fp8_supported


def _layout_enabled() -> bool:
    from sglang.srt.utils import get_bool_env_var

    # default mirrors hcu_sglang.environ (single source): official layout
    # unless the HCU FA perf-A/B path is explicitly requested.
    return get_bool_env_var("SGLANG_KV_LAYOUT_HCU_FA", default="false")


def is_native_fp8(dtype: torch.dtype) -> bool:
    return is_hcu_native_fp8_supported() and dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )


_current_max_seq_len_k: Optional[int] = None


def _set_current_max_seq_len_k(value: Optional[int]) -> Optional[int]:
    global _current_max_seq_len_k
    previous = _current_max_seq_len_k
    _current_max_seq_len_k = value
    return previous


def _apply_out(result, out, return_softmax_lse):
    if out is None:
        return result
    if return_softmax_lse:
        attn_out, lse, *rest = result
        out.copy_(attn_out)
        return (out, lse, *rest)
    out.copy_(result)
    return out


def _restore_hcu_cache_layout(k_cache: torch.Tensor, v_cache: torch.Tensor):
    if not _layout_enabled() or k_cache.ndim != 4 or v_cache.ndim != 4:
        return k_cache, v_cache
    num_pages, page_size, num_heads, head_dim = k_cache.shape
    _, value_page_size, value_heads, value_dim = v_cache.shape
    if page_size != value_page_size or num_heads != value_heads:
        return k_cache, v_cache
    return (
        k_cache.view(num_pages, num_heads, page_size, head_dim),
        v_cache.view(num_pages, value_heads, value_dim, value_page_size),
    )


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    attention_chunk: Optional[int] = None,
    softcap=0.0,
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    sfq=None,
    sfk=None,
    sfv=None,
    rel_bias=None,
    rel_bias_prep_cache=None,
    ver=3,
    out=None,
    max_seqlen_k: Optional[int] = None,
    layout=None,
):
    from flash_attn import (
        vllm_flash_attn_varlen_func,
        vllm_flash_attn_with_kvcache,
    )

    k_cache, v_cache = _restore_hcu_cache_layout(k_cache, v_cache)

    # Varlen calls (prefill / MTP verify carry cu_seqlens_q) must go
    # through the wheel's varlen entry for both KV layouts: the paged
    # entry has no cu_seqlens parameter, and viewing a packed q to
    # [B,1,H,D] makes the wrapper classify S > 1 as mtp packing (its PA
    # kernel then faults). The varlen entry takes the pool plus
    # block_table directly and handles both page layouts (pa_layout).
    if cu_seqlens_q is not None:
        if _layout_enabled():
            page_size = k_cache.shape[2]  # [NP, H, PS, D]
        else:
            page_size = k_cache.shape[1]  # [NP, PS, H, D]
        if max_seqlen_k is None:
            max_seqlen_k = _current_max_seq_len_k or None
        page_capacity = (
            page_table.shape[1] * page_size
            if max_seqlen_k is None and page_table is not None
            else max_seqlen_k
        )
        if q_descale is None and q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            q_descale = torch.ones(1, dtype=torch.float32, device=q.device)
        result = vllm_flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=cache_seqlens,
            max_seqlen_k=page_capacity,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=page_table,
            fa_version=2,
            softcap=softcap,
            s_aux=sinks,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return _apply_out(result, out, return_softmax_lse)

    q_4d = q
    if q.ndim == 3:
        q_4d = q.contiguous().view(-1, 1, q.shape[-2], q.shape[-1])
    result = vllm_flash_attn_with_kvcache(
        q=q_4d,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=page_table,
        cache_seqlens=cache_seqlens,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_softmax_lse=return_softmax_lse,
        num_splits=num_splits,
    )
    return _apply_out(result, out, return_softmax_lse)


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    only_qv=False,
    sm_margin=0,
    return_softmax_lse=False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    sfq=None,
    sfk=None,
    sfv=None,
    rel_bias=None,
    rel_bias_prep_cache=None,
    ver=3,
    out=None,
):
    from flash_attn import flash_attn_varlen_func as hcu_flash_attn_varlen_func

    result = hcu_flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        seqused_k=seqused_k,
        block_table=page_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        return_attn_probs=return_softmax_lse,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        num_splits=num_splits,
        fa_version=ver,
        s_aux=sinks,
    )
    return _apply_out(result, out, return_softmax_lse)


def around_flash_attn_with_kvcache(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return flash_attn_with_kvcache(*args, **kwargs)


def around_flash_attn_varlen_func(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return flash_attn_varlen_func(*args, **kwargs)
