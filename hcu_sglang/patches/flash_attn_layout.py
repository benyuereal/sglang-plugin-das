"""Replace flash_attn's vLLM wrapper, which hardcodes ``layout="custom"``.

The shipped wheel builds ``vllm_flash_attn_varlen_func`` as a passthrough to
``flash_attn_varlen_func`` with a literal layout and ``num_splits=0``, so a
paged NHD pool never reaches the direct paged kernel and decode falls back to
split-KV. This module installs the same call with the layout read off the pool
strides, and with the split knob the wheel does not expose.

Retire it once the shipped wheel carries the fix (flash-attention-das).
"""

from __future__ import annotations

import inspect
import os

import torch

from hcu_sglang.constants import (
    PAGED_DECODE_PAGE_SIZE,
    PAGED_VERIFY_MAX_Q,
    PAGED_VERIFY_Q_ROWS_PER_KV_HEAD,
)

_CUSTOM = "custom"
_INSTALLED = False


def _is_hnd_page_view(t: torch.Tensor) -> bool:
    """[pages, heads, page, dim] seen through a permuted NHD view."""
    return (
        t.dim() == 4
        and t.shape[-2] > 1
        and t.stride(1) == t.shape[-1]
        and t.stride(2) == t.shape[-3] * t.shape[-1]
    )


def _direct_paged_layout(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    block_table,
    window_size,
    softcap,
    seqused_k=None,
) -> str:
    """The wheel's direct paged kernel covers decode and MTP-packed verify."""
    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        return _CUSTOM
    if max_seqlen_q is None or softcap != 0.0:
        return _CUSTOM
    if window_size is not None and tuple(window_size) != (-1, -1):
        return _CUSTOM
    if q.ndim != 3 or k.ndim != 4 or v.ndim != 4:
        return _CUSTOM
    if block_table is None or cu_seqlens_q is None:
        return _CUSTOM
    if not torch.is_tensor(seqused_k):
        return _CUSTOM
    page_size = k.shape[1]
    nheads_q, head_dim_qk = q.shape[-2], q.shape[-1]
    nheads_kv = k.shape[-2]
    if (
        page_size != PAGED_DECODE_PAGE_SIZE
        or nheads_kv <= 0
        or head_dim_qk != v.shape[-1]
    ):
        return _CUSTOM
    if nheads_q % nheads_kv:
        return _CUSTOM
    bs = cu_seqlens_q.numel() - 1
    # The paged entry reshapes q to [bs, max_seqlen_q, heads, dim] and ignores
    # cu_seqlens_q, so q must be exactly the packed decode/verify rows.
    if q.shape[0] != bs * max_seqlen_q:
        return _CUSTOM
    if max_seqlen_q == 1:
        return "bshd"
    if 1 < max_seqlen_q < PAGED_VERIFY_MAX_Q and (
        max_seqlen_q * nheads_q <= PAGED_VERIFY_Q_ROWS_PER_KV_HEAD * nheads_kv
    ):
        return "bshd"
    return _CUSTOM


def _layout_enabled() -> bool:
    from sglang.srt.utils import get_bool_env_var

    return get_bool_env_var("SGLANG_KV_LAYOUT_HCU_FA", default="false")


def _paged_bshd_enabled() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_HCU_FA_PAGED_BSHD.get()


def _choose_layout(
    q, k, v, max_seqlen_q, cu_seqlens_q, block_table, window_size, softcap, seqused_k=None
) -> str:
    if _layout_enabled():
        return _CUSTOM
    if _is_hnd_page_view(k) and _is_hnd_page_view(v):
        return "bhsd"
    if not _paged_bshd_enabled():
        return _CUSTOM
    return _direct_paged_layout(
        q, k, v, max_seqlen_q, cu_seqlens_q, block_table, window_size, softcap,
        seqused_k=seqused_k,
    )


def _num_splits() -> int:
    raw = os.environ.get("SGLANG_HCU_FA_NUM_SPLITS", "").strip()
    return int(raw) if raw else 0


_ORIGINAL = None
_ARG_NAMES = None
# The vLLM wrapper keeps these for its own compatibility surface and never
# forwards them to the varlen entry.
_NOT_FORWARDED = frozenset({"is_prefix_cache", "kv_cache_dtype"})


def _bound_kwargs(args, kwargs) -> dict:
    global _ARG_NAMES

    if _ARG_NAMES is None:
        _ARG_NAMES = tuple(inspect.signature(_ORIGINAL).parameters)
    bound = dict(zip(_ARG_NAMES, args))
    bound.update(kwargs)
    return bound


def vllm_flash_attn_varlen_func(*args, **kwargs):
    """flash_attn's vLLM wrapper with the layout and split width made explicit."""
    import flash_attn

    bound = _bound_kwargs(args, kwargs)
    layout = _choose_layout(
        bound["q"],
        bound["k"],
        bound["v"],
        bound.get("max_seqlen_q"),
        bound.get("cu_seqlens_q"),
        bound.get("block_table"),
        bound.get("window_size"),
        bound.get("softcap", 0.0),
        seqused_k=bound.get("seqused_k"),
    )
    if layout == _CUSTOM:
        return _ORIGINAL(*args, **kwargs)

    forwarded = {k: v for k, v in bound.items() if k not in _NOT_FORWARDED}
    if forwarded.get("q_descale") is None and bound["q"].dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        forwarded["q_descale"] = torch.ones(1, dtype=torch.float32, device=bound["q"].device)
    forwarded["layout"] = layout
    forwarded["num_splits"] = _num_splits()
    return flash_attn.flash_attn_varlen_func(**forwarded)


def install() -> bool:
    """Point flash_attn's wrapper at the layout-aware replacement."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        import flash_attn
    except ImportError:
        return False
    global _ORIGINAL

    if not hasattr(flash_attn, "vllm_flash_attn_varlen_func"):
        return False
    _ORIGINAL = flash_attn.vllm_flash_attn_varlen_func
    flash_attn.vllm_flash_attn_varlen_func = vllm_flash_attn_varlen_func
    _INSTALLED = True
    return True
