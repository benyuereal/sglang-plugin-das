"""GDN B+ commit chain for the 0.5.22 stack (no state handoff).

verify writes only the full-accept row (draft_token_num - 1); the commit
replays the same stash through the same kernel for the truncated slots and
writes the accepted step. 0.5.22 has no handoff, so the row base is the
batch position and the snapshot rows are the verify's
``intermediate_state_indices`` (the same rows the upstream scatter reads).

The 0.5.18 original (gdn_backend.py) is handoff-based; this port follows the
no-handoff variant: ssm_states = the temporal pool, h0_rows = the mamba
cache indices, store_row = draft_token_num - 1 (constant), store_step = the
accepted step for the truncated slots and -1 for the full-accept ones.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get("SGLANG_GDN_HIP_BPLUS") == "1"


_HANDOFF_OK = None
_HANDOFF_LOGGED = False


def handoff_enabled() -> bool:
    """Zero-copy state handoff: verify reads h0 from the snapshot rows.

    Requires B+ and ``SGLANG_GDN_HIP_HANDOFF_SKIP_TRACK=1``, which is only
    valid with ``--disable-radix-cache``: the mamba interval checkpoints
    (``mamba_track_indices``) are radix-cache bookkeeping, so with radix off
    the commit can drop both full-state scatters. With radix on keep the
    upstream copy path (never touch the pointer).
    """
    global _HANDOFF_OK
    if _HANDOFF_OK is not None:
        return _HANDOFF_OK
    _HANDOFF_OK = (
        enabled()
        and os.environ.get("SGLANG_GDN_HIP_HANDOFF") == "1"
        and os.environ.get("SGLANG_GDN_HIP_HANDOFF_SKIP_TRACK") == "1"
    )
    return _HANDOFF_OK


def _slot_tensor(backend, slots, fill: int, size: int) -> torch.Tensor:
    """Per-slot buffer sized by the mamba pool, not by the running batch."""
    buf = getattr(backend, "_gdn_hip_prev", None)
    if buf is None or int(buf.numel()) < size:
        buf = torch.full((size,), fill, dtype=torch.int32, device=slots.device)
        object.__setattr__(backend, "_gdn_hip_prev", buf)
    return buf


def ensure_prev(backend, slots: torch.Tensor, mamba_cache_params, D: int):
    """Return ``prev_step`` for this step's slots, seeding new slots from temporal.

    A slot whose state has never gone through the handoff has ``prev_step < 0``.
    Its post-prefill state lives in ``temporal[slot]``; copy it once into
    ``intermediate_ssm[slot, 0]`` so every later verify reads the same buffer.
    """
    del D  # row stride is owned by seed_prev / h0_rows
    prev_buf = _slot_tensor(
        backend, slots, -1, int(mamba_cache_params.temporal.shape[0]) + 3
    )
    # Pure gather: graph-capture safe (no data-dependent branch, no temp).
    s64 = slots.to(torch.int64).clamp_min(0)
    return prev_buf.index_select(0, s64).to(torch.int32)


def seed_prev(backend, slots: torch.Tensor, mamba_cache_params, cache_steps: int) -> None:
    """Publish a freshly written temporal state to the handoff scratch rows.

    Runs at the end of prefill/extend, after the kernel wrote ``temporal``.
    Unconditional over the batch's slots, so it stays capture-safe; one state
    copy per slot per prefill (amortized over the request's decode steps).
    """
    if not handoff_enabled():
        return
    temporal = mamba_cache_params.temporal
    inter = mamba_cache_params.intermediate_ssm
    if inter is None or temporal is None:
        return
    prev_buf = _slot_tensor(backend, slots, -1, int(temporal.shape[0]) + 3)
    s64 = slots.to(torch.int64).clamp_min(0)
    flat = inter.reshape(-1, *inter.shape[2:])
    flat.index_copy_(
        0, s64 * int(cache_steps), temporal.index_select(0, s64)
    )
    prev_buf.index_fill_(0, s64, 0)


def invalidate_prev(backend, slots: torch.Tensor) -> None:
    """Drop the handoff pointer for slots whose state was rewritten by prefill."""
    if not handoff_enabled():
        return
    buf = getattr(backend, "_gdn_hip_prev", None)
    if buf is None:
        return
    # Static-shape form: prefill may run inside a captured graph.
    buf.index_fill_(0, slots.to(torch.int64).clamp_min(0), -1)


def remember_step(backend, slots: torch.Tensor, prev: torch.Tensor, cache_steps: int):
    """Record the step's prev/store_row so the commit can rotate the pointer.

    ``cache_steps`` is the allocated row stride (``intermediate_ssm.shape[1]``),
    not the runtime draft-token count: the rows are addressed by the stride.
    """
    store_row = torch.remainder(prev + 1, cache_steps).to(torch.int32)
    object.__setattr__(backend, "_gdn_hip_slots_now", slots.to(torch.int32))
    object.__setattr__(backend, "_gdn_hip_prev_now", prev.to(torch.int32))
    object.__setattr__(backend, "_gdn_hip_store_row_now", store_row)
    object.__setattr__(backend, "_gdn_hip_cache_steps", int(cache_steps))
    return store_row


def h0_rows(slots: torch.Tensor, prev: torch.Tensor, D: int) -> torch.Tensor:
    rows = slots.to(torch.int64).clamp_min(0) * D + prev.to(torch.int64).clamp_min(0)
    return rows.to(torch.int32)


def commit_prev(backend, state_indices, last_correct, T: int, cache_steps: int) -> bool:
    """Advance ``prev_step`` to the committed row; no state copy.

    Full-accept slots point at the verify's rotated row; truncated slots point
    at the row the compensate replay wrote (``last_correct``). Row stride is the
    allocated ``cache_steps`` recorded by :func:`remember_step`.
    """
    if not handoff_enabled():
        return False
    try:
        prev_buf = getattr(backend, "_gdn_hip_prev")
        idx = state_indices.to(torch.int64)
        last = last_correct.to(torch.int64)
        # Compute the rotated row here, not from the forward: the forward runs
        # inside captured CUDA graphs whose Python-level tensors keep the
        # capture batch's length, while the commit sees the live batch.
        prev = prev_buf.index_select(0, idx.clamp_min(0)).to(torch.int64)
        store_row = torch.remainder(prev + 1, cache_steps)
        keep = torch.where(last < T - 1, last, store_row).to(torch.int32)
        valid = idx >= 0
        prev_buf.index_copy_(0, idx[valid], keep[valid])
        global _HANDOFF_LOGGED
        if not _HANDOFF_LOGGED:
            _HANDOFF_LOGGED = True
            print("[gdn-bplus] handoff commit active (scatter skipped)", flush=True)
        return True
    except Exception:  # noqa: BLE001 - fall back to the scatter
        logger.warning("[gdn-bplus] commit_prev failed", exc_info=True)
        return False


def _fused_qkv_src(q, k, v, n: int, T: int):
    """Return the [n, T, W] view when q/k/v share one fused buffer, else None."""
    try:
        qd = int(q.shape[2]) * int(q.shape[3])
        kd = int(k.shape[2]) * int(k.shape[3])
        vd = int(v.shape[2]) * int(v.shape[3])
        W = int(q.stride(1))
        esz = q.element_size()
        if W != qd + kd + vd:
            return None
        if int(k.stride(1)) != W or int(v.stride(1)) != W:
            return None
        if int(q.stride(3)) != 1 or int(k.stride(3)) != 1 or int(v.stride(3)) != 1:
            return None
        if (
            int(q.stride(2)) != int(q.shape[3])
            or int(k.stride(2)) != int(k.shape[3])
            or int(v.stride(2)) != int(v.shape[3])
        ):
            return None
        if q.dtype != k.dtype or q.dtype != v.dtype:
            return None
        base = q.untyped_storage().data_ptr()
        if k.untyped_storage().data_ptr() != base or v.untyped_storage().data_ptr() != base:
            return None
        if int(k.data_ptr()) - int(q.data_ptr()) != qd * esz:
            return None
        if int(v.data_ptr()) - int(k.data_ptr()) != kd * esz:
            return None
        if int(q.shape[1]) != n * T:
            return None
        return torch.as_strided(q, (n, T, W), (T * W, W, 1))
    except Exception:  # noqa: BLE001 - fall back to three copies
        return None


def _max_slots(backend) -> int:
    value = int(getattr(backend, "_gdn_hip_max_slots", 0) or 0)
    if value > 0:
        return value
    try:
        return int(backend.req_to_token_pool.size)
    except Exception:  # noqa: BLE001
        return 0


def _ensure(backend, layer, q, k, v, a, b, D: int):
    ents = getattr(backend, "_gdn_hip_stash_ents", None)
    if ents is None:
        ents = {}
        object.__setattr__(backend, "_gdn_hip_stash_ents", ents)
    max_slots = _max_slots(backend)
    if max_slots <= 0:
        return None, 0
    rows = max_slots * D
    lid = layer.layer_id
    ent = ents.get(lid)
    if ent is None or int(ent["rows"]) < rows:
        dev = q.device
        qd = int(q.shape[2]) * int(q.shape[3])
        kd = int(k.shape[2]) * int(k.shape[3])
        vd = int(v.shape[2]) * int(v.shape[3])
        W = qd + kd + vd
        qkv = torch.zeros((rows, W), dtype=q.dtype, device=dev)
        ent = {
            "rows": rows,
            "qh": q.shape[2], "qk": q.shape[3],
            "kh": k.shape[2], "kk": k.shape[3],
            "vh": v.shape[2], "vv": v.shape[3],
            "qkv": qkv,
            "q": qkv[:, :qd],
            "k": qkv[:, qd:qd + kd],
            "v": qkv[:, qd + kd:W],
            "a": torch.zeros((rows, a.shape[-1]), dtype=a.dtype, device=dev),
            "b": torch.zeros((rows, b.shape[-1]), dtype=b.dtype, device=dev),
            "out": torch.zeros(
                (1, 1, v.shape[2], v.shape[3]), dtype=v.dtype, device=dev
            ),
            "A_log": None,
            "dt_bias": None,
        }
        ents[lid] = ent
    ent["A_log"] = layer.A_log
    ent["dt_bias"] = layer.dt_bias
    return ent, max_slots


def stash(backend, layer, q, k, v, a, b, draft_token_num) -> bool:
    if not enabled():
        return False
    # Host-side int: the commit needs T without a device sync (.item() would
    # serialize the CPU behind the verify every step).
    object.__setattr__(backend, "_gdn_hip_T", int(draft_token_num))
    try:
        T = int(draft_token_num)
        n = int(q.shape[1]) // T
        if T <= 0 or n <= 0:
            return False
        D = T
        ent, max_slots = _ensure(backend, layer, q, k, v, a, b, D)
        if ent is None or n > max_slots:
            return False
        m = n * D
        src = _fused_qkv_src(q, k, v, n, T)
        if src is not None:
            ent["qkv"][:m].view(n, D, -1)[:, :T].copy_(src)
        else:
            qd = int(q.shape[2]) * int(q.shape[3])
            kd = int(k.shape[2]) * int(k.shape[3])
            W = qd + kd + int(v.shape[2]) * int(v.shape[3])
            dest = ent["qkv"][:m].view(n, D, W)
            dest[:, :T, :qd].copy_(q.reshape(n, T, -1))
            dest[:, :T, qd:qd + kd].copy_(k.reshape(n, T, -1))
            dest[:, :T, qd + kd:W].copy_(v.reshape(n, T, -1))
        ent["a"][:m].view(n, D, -1)[:, :T].copy_(a.reshape(n, T, -1))
        ent["b"][:m].view(n, D, -1)[:, :T].copy_(b.reshape(n, T, -1))
        tdev = getattr(backend, "_gdn_hip_tdev", None)
        if tdev is None:
            tdev = torch.zeros(1, dtype=torch.int32, device=q.device)
            object.__setattr__(backend, "_gdn_hip_tdev", tdev)
        tdev.fill_(T)
        return True
    except Exception:  # noqa: BLE001 - never take serving down
        logger.warning("[gdn-bplus] stash failed", exc_info=True)
        return False


def compensate_from_stash(backend, state_indices, last_correct, mamba_caches=None) -> bool:
    """Replay the stash for the truncated slots, before the upstream scatter."""
    if not enabled():
        return False
    ents = getattr(backend, "_gdn_hip_stash_ents", None)
    if not ents:
        return False
    snap_idx = getattr(backend, "_gdn_hip_snap_idx", None)
    if snap_idx is None:
        return False
    try:
        T = int(getattr(backend, "_gdn_hip_T", 0))
        if T <= 1:
            return False
        n = int(state_indices.shape[0])
        if int(snap_idx.numel()) != n:
            return False
        from hcu_sglang.layers.attention.linear import gdn_hip_bplus

        idx32 = state_indices.to(torch.int32)
        last32 = last_correct.to(torch.int32)
        trunc = (idx32 > 0) & (last32 >= 0) & (last32 < T - 1)
        store_step = torch.where(
            trunc, last32, torch.full_like(last32, -1)
        ).to(torch.int32)
        h0_idx_default = idx32
        snap32 = snap_idx.to(torch.int32)
        D = T
        m = n * D
        handoff = handoff_enabled()
        if handoff:
            prev_buf = getattr(backend, "_gdn_hip_prev", None)
            if prev_buf is None:
                return False
            # Eager read: the forward's Python tensors are capture-length.
            slots_now = state_indices.to(torch.int32)
            prev_now = prev_buf.index_select(
                0, slots_now.to(torch.int64).clamp_min(0)
            ).to(torch.int32)
        for lid, ent in list(ents.items()):
            try:
                layer_cache = ent.get("layer_cache")
                if layer_cache is None:
                    layer_cache = backend.req_to_token_pool.mamba2_layer_cache(lid)
                    ent["layer_cache"] = layer_cache
                inter = layer_cache.intermediate_ssm
                if inter is None:
                    continue
                flat = ent.get("flat")
                if flat is None:
                    flat = inter.reshape(-1, *inter.shape[2:])
                    ent["flat"] = flat
                if handoff:
                    ssm = flat
                    h0_idx = h0_rows(slots_now, prev_now, int(inter.shape[1]))
                else:
                    ssm = layer_cache.temporal
                    h0_idx = h0_idx_default
                gdn_hip_bplus.compensate(
                    ent["q"][:m].view(n, D, ent["qh"], ent["qk"]),
                    ent["k"][:m].view(n, D, ent["kh"], ent["kk"]),
                    ent["v"][:m].view(n, D, ent["vh"], ent["vv"]),
                    ent["a"][:m].view(n, D, -1),
                    ent["b"][:m].view(n, D, -1),
                    ent["A_log"],
                    ent["dt_bias"],
                    ssm,
                    h0_idx,
                    flat,
                    snap32,
                    int(inter.shape[1]),
                    store_step,
                    n,
                    D,
                    ent["out"],
                )
            except Exception:  # noqa: BLE001 - optional path
                logger.warning("[gdn-bplus] compensate failed for layer %s", lid)
                continue
        return True
    except Exception:  # noqa: BLE001
        logger.warning("[gdn-bplus] compensate_from_stash failed", exc_info=True)
        return False
