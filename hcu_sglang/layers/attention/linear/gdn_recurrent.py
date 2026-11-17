"""GDN recurrent kernels, served by ``lightop.gdn`` when it is installed.

The kernel and its tensor marshalling live in the lightop operator library;
this module only maps the binary switch onto that API and reports
availability. When lightop is absent the overlay keeps the upstream
Triton dispatcher path.
"""

import logging
import os

import torch

try:
    from lightop import gdn as _gdn
except Exception:  # noqa: BLE001 - the backend is optional
    _gdn = None

logger = logging.getLogger(__name__)


def enabled() -> bool:
    """Whether the operator library that serves these kernels is importable."""
    return _gdn is not None


_SWITCH_ENV = "SGLANG_GDN_HIP"
_ON = frozenset({"1", "all", "on", "true", "yes"})
_OFF = frozenset({"", "0", "off", "none", "false", "no"})
_LEGACY_PREFIX = "SGLANG_GDN_HIP_"
_switch_cache: dict[str, bool] = {}
_legacy_warned = False


def _warn_legacy() -> None:
    global _legacy_warned
    if _legacy_warned:
        return
    _legacy_warned = True
    legacy = sorted(
        name for name in os.environ
        if name.startswith(_LEGACY_PREFIX) and name != _SWITCH_ENV
    )
    if legacy:
        logger.warning("[gdn] %s are ignored; set %s=1 instead",
                       ", ".join(legacy), _SWITCH_ENV)


def _switched_on() -> bool:
    _warn_legacy()
    value = os.environ.get(_SWITCH_ENV, "").strip().lower()
    result = _switch_cache.get(value)
    if result is None:
        result = _switch_cache[value] = value not in _OFF
        if value not in _ON and value not in _OFF:
            logger.warning("[gdn] unknown %s=%r; treating it as on",
                           _SWITCH_ENV, value)
    return result


def overlay_enabled() -> bool:
    """The operator asked for the GDN overlay and lightop can serve it."""
    return _switched_on() and enabled()


def verify(q, k, v, a, b, A_log, dt_bias, cu_seqlens, ssm_states, cache_indices,
           snap_flat, snap_idx, cache_steps, draft_token_num, o,
           prev_step=None, store=None, store_row=None):
    """Write the full-run snapshot to row (prev_step + 1) % cache_steps."""
    n = int(snap_idx.shape[0])
    if store is None:
        store = torch.full((n,), int(draft_token_num) - 1,
                           dtype=torch.int32, device=snap_idx.device)
    if store_row is None:
        if prev_step is None:
            store_row = store
        else:
            safe = snap_idx.to(torch.int64).clamp(0, int(prev_step.numel()) - 1)
            prev = prev_step.index_select(0, safe).to(torch.int32)
            store_row = torch.remainder(prev + 1, int(cache_steps)).to(torch.int32)
    return _gdn.gdn_verify(
        q, k, v, a, b, A_log, dt_bias, cu_seqlens, ssm_states, cache_indices,
        snap_flat, snap_idx, cache_steps, draft_token_num, o,
        prev_step=prev_step, store=store, store_row=store_row,
        prenorm=True, prenorm_gate=True,
    )


def verify_full(q, k, v, a, b, A_log, dt_bias, cu_seqlens, ssm_states,
                cache_indices, snap_flat, snap_idx, cache_steps,
                draft_token_num, o):
    """Full-snapshot entry, for the A/B only: same kernel and inputs, every step."""
    return _gdn.gdn_verify_full(
        q, k, v, a, b, A_log, dt_bias, cu_seqlens, ssm_states, cache_indices,
        snap_flat, snap_idx, cache_steps, draft_token_num, o,
    )


def compensate(q, k, v, a, b, A_log, dt_bias, ssm_states, h0_rows,
               snap_flat, snap_idx, cache_steps, store_step, n_seq,
               tok_per_seq, o):
    """Replay state only (no o); slots with h0_rows < 0 exit immediately."""
    return _gdn.gdn_compensate(
        q, k, v, a, b, A_log, dt_bias, ssm_states, h0_rows, snap_flat,
        snap_idx, cache_steps, store_step, n_seq, tok_per_seq, o,
    )


def has_multi() -> bool:
    return enabled()


def compensate_multi(layer_ptrs, h0_indices, snap_idx, cache_steps, store_step,
                     n_seq, tok_per_seq, hv, h, kdim, vdim):
    """One-launch state replay for every layer in ``layer_ptrs``."""
    return _gdn.gdn_compensate_multi(
        layer_ptrs, h0_indices, snap_idx, cache_steps, store_step, n_seq,
        tok_per_seq, hv, h, kdim, vdim,
    )


def f32(x: torch.Tensor) -> torch.Tensor:
    """Cached fp32 view of A_log/dt_bias for the kernel's float pointers."""
    return _gdn.f32(x)


_CONV_LAST_REASON = None


def conv_last_reason():
    """Why the last conv_update() call stayed on Triton (None if it ran)."""
    return _CONV_LAST_REASON


def conv_update(x, conv_state, weight, bias=None, activation=None,
                cache_seqlens=None, conv_state_indices=None,
                num_accept_tokens=None, intermediate_conv_window=None,
                intermediate_state_indices=None, retrieve_next_token=None,
                retrieve_next_sibling=None, retrieve_parent_token=None,
                pad_slot_id=-1):
    """Launch lightop's conv update; None means the caller keeps Triton."""
    global _CONV_LAST_REASON
    reason = _gdn.gdn_conv1d_unsupported_reason(
        x, conv_state, weight, bias,
        cache_seqlens=cache_seqlens, conv_state_indices=conv_state_indices,
        num_accept_tokens=num_accept_tokens,
        intermediate_conv_window=intermediate_conv_window,
        intermediate_state_indices=intermediate_state_indices,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token, pad_slot_id=pad_slot_id,
    )
    if reason is not None:
        _CONV_LAST_REASON = reason
        return None
    return _gdn.gdn_conv1d_update(
        x, conv_state, weight, bias=bias, activation=activation,
        cache_seqlens=cache_seqlens, conv_state_indices=conv_state_indices,
        num_accept_tokens=num_accept_tokens,
        intermediate_conv_window=intermediate_conv_window,
        intermediate_state_indices=intermediate_state_indices,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token, pad_slot_id=pad_slot_id,
    )
