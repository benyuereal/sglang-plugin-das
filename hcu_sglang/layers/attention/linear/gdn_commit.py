"""Commit only the accepted step's GDN state.

Verify writes the full-run snapshot row; the commit replays the stash for the
slots the accept truncated, so the 6 GiB state copy is not needed. The path
works with the radix cache on: a shared prefix changes which slots the accepted
step lands on, never the slot-indexed state layout.

"B+" is the original name of this path and survives in the kernel entry points;
the module is named after what it does.
"""

import logging
from typing import NamedTuple

import torch

from hcu_sglang.layers.attention.linear import gdn_recurrent
from hcu_sglang.layers.attention.linear import gdn_switches as switches

logger = logging.getLogger(__name__)

class HeadDims(NamedTuple):
    """Per-layer head layout, as the recurrent kernel indexes it."""
    qh: int
    qk: int
    kh: int
    kk: int
    vh: int
    vv: int

    @classmethod
    def of(cls, q, k, v):
        return cls(q.shape[2], q.shape[3], k.shape[2], k.shape[3],
                   v.shape[2], v.shape[3])

    @classmethod
    def from_entry(cls, ent):
        dims = ent["dims"]
        return dims if isinstance(dims, HeadDims) else cls(*dims)

class CommitState:
    """Per-backend scratch, kept off the nn.Module so plain attributes work."""

    __slots__ = ("stash", "snap_idx", "capture_T", "live_T", "max_slots",
                 "ml_table", "ml_key", "commit_failed")

    def __init__(self):
        self.stash = self.snap_idx = self.ml_table = self.ml_key = None
        self.capture_T = self.live_T = self.max_slots = 0
        self.commit_failed = False

def state(backend) -> CommitState:
    # The adaptive controller swaps a second backend in between batches, so
    # key the pointer on the shared request pool, not on the backend.
    pool = getattr(backend, "req_to_token_pool", None)
    owner = backend if pool is None else pool
    st = getattr(owner, "_gdn_commit", None)
    if st is None:
        st = owner._gdn_commit = CommitState()
    return st

def enabled() -> bool:
    return switches.enabled()

def _head_dims(*tensors) -> list[int]:
    return [int(t.shape[2]) * int(t.shape[3]) for t in tensors]

def _fused_qkv_view(q, k, v, n: int, T: int):
    """[n, T, W] view over one fused q|k|v buffer, or None when they are apart."""
    qd, kd, vd = _head_dims(q, k, v)
    W, ts = int(q.stride(1)), (q, k, v)
    if (q.dtype != k.dtype or q.dtype != v.dtype or W != qd + kd + vd
            or any(int(t.stride(1)) != W for t in ts)):
        return None
    # Row-major within the head dim: the head axis sits immediately above it.
    if any(int(t.stride(3)) != 1 or int(t.stride(2)) != int(t.shape[3]) for t in ts):
        return None
    base, esz = q.untyped_storage().data_ptr(), q.element_size()
    for t, off in zip(ts, (0, qd, qd + kd)):
        if (t.untyped_storage().data_ptr() != base
                or int(t.data_ptr()) - int(q.data_ptr()) != off * esz):
            return None
    if int(q.shape[1]) != n * T:
        return None
    return torch.as_strided(q, (n, T, W), (T * W, W, 1))

def _ensure_layer_entry(backend, layer, q, k, v, a, b, D: int):
    st = state(backend)
    if st.stash is None:
        st.stash = {}
    if st.max_slots <= 0:
        st.max_slots = int(backend.req_to_token_pool.size)
    if st.max_slots <= 0:
        return None, 0
    rows, lid = st.max_slots * D, layer.layer_id
    ent = st.stash.get(lid)
    if ent is None or int(ent["rows"]) < rows:
        st.ml_table = None  # layer pointers move with the reallocation
        dev, dims = q.device, HeadDims.of(q, k, v)
        qd, kd, vd = _head_dims(q, k, v)
        qkv = torch.zeros((rows, qd + kd + vd), dtype=q.dtype, device=dev)
        ent = {"rows": rows, "dims": dims, "qkv": qkv,
               "q": qkv[:, :qd], "k": qkv[:, qd:qd + kd], "v": qkv[:, qd + kd:],
               "a": torch.zeros((rows, a.shape[-1]), dtype=a.dtype, device=dev),
               "b": torch.zeros((rows, b.shape[-1]), dtype=b.dtype, device=dev),
               # One output row per layer launch of the compensate kernel.
               "out": torch.zeros((1, 1, *dims[4:]), dtype=v.dtype, device=dev),
               "A_log": None, "dt_bias": None}
        st.stash[lid] = ent
    ent["A_log"], ent["dt_bias"] = layer.A_log, layer.dt_bias
    return ent, st.max_slots

def stash(backend, layer, q, k, v, a, b, draft_token_num) -> bool:
    if not enabled():
        return False
    st = state(backend)
    # Host-side int: the commit needs T without a device sync.
    st.capture_T = int(draft_token_num)
    try:
        T = int(draft_token_num)
        n = int(q.shape[1]) // T
        if T <= 0 or n <= 0:
            return False
        ent, max_slots = _ensure_layer_entry(backend, layer, q, k, v, a, b, T)
        if ent is None or n > max_slots:
            return False
        m, a_width, b_width = n * T, ent["a"].shape[-1], ent["b"].shape[-1]
        src = _fused_qkv_view(q, k, v, n, T)
        if src is not None:
            ent["qkv"][:m].view(n, T, int(ent["qkv"].shape[-1]))[:, :T].copy_(src)
        else:
            qd, kd, vd = _head_dims(q, k, v)
            W = qd + kd + vd
            dest = ent["qkv"][:m].view(n, T, W)
            dest[:, :T, :qd].copy_(q.reshape(n, T, -1))
            dest[:, :T, qd:qd + kd].copy_(k.reshape(n, T, -1))
            dest[:, :T, qd + kd:].copy_(v.reshape(n, T, -1))
        ent["a"][:m].view(n, T, a_width)[:, :T].copy_(a.reshape(n, T, a_width))
        ent["b"][:m].view(n, T, b_width)[:, :T].copy_(b.reshape(n, T, b_width))
        return True
    except Exception:  # noqa: BLE001 - never take serving down
        logger.warning("[gdn-bplus] stash failed", exc_info=True)
        return False

def verify_step(*, backend, layer, q, k, v, a, b, draft_token_num, n, snap_cache,
                cache_indices, intermediate_indices, mamba_cache, ss_states,
                query_start_loc):
    """Write this step's full snapshot row and hand the truncated rows to ``stash``.

    Returns the kernel output, or None when the chain declined.
    """
    if not stash(backend, layer, q, k, v, a, b, draft_token_num):
        return None
    flat = snap_cache.reshape(-1, *snap_cache.shape[2:])
    st = state(backend)
    # Default verify rows are batch positions: the snapshot lands in the
    # intermediate buffer, and the commit replays truncated slots into the
    # slot-indexed temporal state.
    snap_idx = intermediate_indices[:n].to(torch.int32)
    st.snap_idx = snap_idx
    out = q.new_empty((1, q.shape[1], v.shape[2], v.shape[3]))
    gdn_recurrent.verify(
        q, k, v, a, b, layer.A_log, layer.dt_bias, query_start_loc[: n + 1],
        ss_states, cache_indices[:n].to(torch.int32), flat, snap_idx,
        int(snap_cache.shape[1]), int(draft_token_num), out)
    return out

def compensate_from_stash(backend, state_indices, last_correct) -> bool:
    """Replay the stash for the truncated slots, before the upstream scatter."""
    if not enabled():
        return False
    st = state(backend)
    if not st.stash or st.snap_idx is None:
        return False
    try:
        # Graph replay skips verify's Python: prefer the live value.
        T = int(st.live_T or st.capture_T)
        if T <= 1:
            return False
        n, idx32 = int(state_indices.shape[0]), state_indices.to(torch.int32)
        from sglang.srt.layers.attention.linear.utils import (
            pp_spec_stable_rows_enabled,
        )
        if pp_spec_stable_rows_enabled():
            if int(st.snap_idx.numel()) != n:
                return False
            snap32 = st.snap_idx.to(torch.int32)
        else:
            # Default verify rows are batch positions; rebuild for the live batch.
            snap32 = torch.arange(n, dtype=torch.int32,
                                  device=state_indices.device)
        last32 = last_correct.to(torch.int32)
        trunc = (idx32 > 0) & (last32 >= 0) & (last32 < T - 1)
        store_step = torch.where(trunc, last32, torch.full_like(last32, -1))
        D, m = T, n * T
        # A full accept already wrote the snapshot row, so replay only the
        # truncated slots. ``neg`` is built unconditionally: ``torch.where``
        # rejects None, and the per-layer except would drop every layer.
        neg = torch.full_like(idx32, -1)
        h0_plain = torch.where(trunc, idx32, neg)
        rows = []
        for lid, ent in st.stash.items():
            try:
                layer_cache = ent.get("layer_cache")
                if layer_cache is None:
                    layer_cache = backend.req_to_token_pool.mamba2_layer_cache(lid)
                    ent["layer_cache"] = layer_cache
                inter = layer_cache.intermediate_ssm
                if inter is None:
                    st.commit_failed = True
                    continue
                flat = ent.get("flat")
                if flat is None:
                    flat = ent["flat"] = inter.reshape(-1, *inter.shape[2:])
                stride = int(inter.shape[1])
                # Plain rows are slot-indexed temporal state; the snapshot
                # rows are the verify's batch positions.
                rows.append((ent, layer_cache.temporal, flat, h0_plain,
                             stride, lid))
            except Exception:  # noqa: BLE001 - optional path
                logger.warning("[gdn-bplus] compensate failed for layer %s",
                               lid, exc_info=True)
                st.commit_failed = True
        if not rows:
            return False
        if gdn_recurrent.has_multi():
            try:
                if _compensate_multi(st, rows, n, D, m, store_step, snap32):
                    return True
            except Exception:  # noqa: BLE001 - per-layer path is the fallback
                logger.warning("[gdn-bplus] multi compensate failed", exc_info=True)
        for ent, ssm, flat, h0_idx, cache_steps, lid in rows:
            try:
                dims = HeadDims.from_entry(ent)
                gdn_recurrent.compensate(
                    ent["q"][:m].view(n, D, *dims[:2]),
                    ent["k"][:m].view(n, D, *dims[2:4]),
                    ent["v"][:m].view(n, D, *dims[4:]),
                    ent["a"][:m].view(n, D, ent["a"].shape[-1]),
                    ent["b"][:m].view(n, D, ent["b"].shape[-1]),
                    ent["A_log"], ent["dt_bias"], ssm, h0_idx, flat, snap32,
                    cache_steps, store_step, n, D, ent["out"])
            except Exception:  # noqa: BLE001 - optional path
                logger.warning("[gdn-bplus] compensate failed for layer %s",
                               lid, exc_info=True)
                st.commit_failed = True
        return True
    except Exception:  # noqa: BLE001
        logger.warning("[gdn-bplus] compensate_from_stash failed", exc_info=True)
        return False

def _compensate_multi(st, rows, n, D, m, store_step, snap32) -> bool:
    """One launch covers every layer when their layout matches the table."""
    if len(rows) < 2:
        return False
    ent0 = rows[0][0]
    dims, cache_steps = HeadDims.from_entry(ent0), rows[0][4]
    qh, qk, kh, kk, vh, vv = dims
    views = (ent0["q"][:m].view(n, D, qh, qk),
             ent0["k"][:m].view(n, D, kh, kk),
             ent0["v"][:m].view(n, D, vh, vv),
             ent0["a"][:m].view(n, D, ent0["a"].shape[-1]),
             ent0["b"][:m].view(n, D, ent0["b"].shape[-1]))
    strides = tuple(int(t.stride(1)) for t in views)
    # The table holds raw data_ptrs, so this key is only valid because mamba
    # pools are allocated once, before graph capture; a replaced layer entry
    # clears it at `st.ml_table = None`.
    key = (len(rows), dims, cache_steps, strides)
    table = st.ml_table
    if table is None or st.ml_key != key:
        cells = []
        for ent, ssm, flat, h0_idx, cs, _ in rows:
            if (HeadDims.from_entry(ent) != dims or cs != cache_steps
                    or int(ent["q"].stride(0)) != strides[0]
                    or int(ent["a"].stride(0)) != strides[3]
                    or int(ent["b"].stride(0)) != strides[4]):
                return False
            cells.append([ent["q"].data_ptr(), ent["k"].data_ptr(),
                          ent["v"].data_ptr(), ent["a"].data_ptr(),
                          ent["b"].data_ptr(),
                          gdn_recurrent.f32(ent["A_log"]).data_ptr(),
                          gdn_recurrent.f32(ent["dt_bias"]).data_ptr(),
                          ssm.data_ptr(), ent["out"].data_ptr(), flat.data_ptr(),
                          *strides])
        table = torch.tensor(cells, dtype=torch.int64, device=ent0["q"].device)
        st.ml_table, st.ml_key = table, key
    gdn_recurrent.compensate_multi(table, rows[0][3], snap32, cache_steps,
                                   store_step, n, D, vh, qh, kk, vv)
    return True
