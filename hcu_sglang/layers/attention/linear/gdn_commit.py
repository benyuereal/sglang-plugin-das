"""Commit only the accepted step's GDN state.

Verify keeps a single snapshot row; the commit replays the stash for the slots
the accept truncated, so the 6 GiB state copy is not needed. The handoff
variant points verify at the committed row and is valid only with
``--disable-radix-cache``.

"B+" is the original name of this path and survives in the kernel entry points
and SGLANG_GDN_HIP_BPLUS; the module is named after what it does.
"""

import logging
import os
from typing import NamedTuple

import torch

from hcu_sglang.layers.attention.linear import gdn_recurrent

# Row floor for the seed scratch: a one-slot chunk must not reallocate next chunk.
SEED_SCRATCH_MIN_ROWS = 128

# Spare rows on slot-indexed buffers: an out-of-range slot parks without a
# bounds check on the hot path.
SLOT_BUFFER_SLACK = 3

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

    __slots__ = (
        "prev", "stash", "snap_idx", "capture_T", "live_T", "seed_scratch",
        "ml_table", "ml_key", "max_slots", "logged", "commit_failed",
        "step_gen", "step_lid", "step_cache", "s64_buf", "prev32_buf",
        "prev64_buf", "store_buf", "row_bufs", "h0_bufs", "tmp32_buf",
        "tmp64_buf",
    )

    def __init__(self):
        self.prev = self.stash = self.snap_idx = None
        self.seed_scratch = self.ml_table = self.ml_key = self.step_cache = None
        self.capture_T = self.live_T = self.max_slots = self.step_gen = 0
        self.step_lid = -1
        self.logged = self.commit_failed = False
        for name in ("s64_buf", "prev32_buf", "prev64_buf", "store_buf",
                     "row_bufs", "h0_bufs", "tmp32_buf", "tmp64_buf"):
            setattr(self, name, {})

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
    return os.environ.get("SGLANG_GDN_HIP_BPLUS") == "1"

_HANDOFF_OK = None

def handoff_enabled() -> bool:
    """Zero-copy state handoff: verify reads h0 from the committed row."""
    global _HANDOFF_OK
    if _HANDOFF_OK is None:
        _HANDOFF_OK = (enabled()
                       and os.environ.get("SGLANG_GDN_HIP_HANDOFF") == "1"
                       and os.environ.get("SGLANG_GDN_HIP_HANDOFF_SKIP_TRACK") == "1")
    return _HANDOFF_OK

def _slot_ids(slots) -> torch.Tensor:
    return slots.to(torch.int64).clamp_min(0)

def _head_dims(*tensors) -> list[int]:
    return [int(t.shape[2]) * int(t.shape[3]) for t in tensors]

def _prev_buf(backend, slots, size: int) -> torch.Tensor:
    st = state(backend)
    if st.prev is None or int(st.prev.numel()) < size:
        st.prev = torch.full((size,), -1, dtype=torch.int32, device=slots.device)
    return st.prev

def _buf(st, name: str, key, rows: int, dtype, device) -> torch.Tensor:
    """Persistent scratch keyed per size: captures bake buffer addresses."""
    bufs = getattr(st, name)
    buf = bufs.get(key)
    if buf is None or buf.dtype != dtype or buf.device != device:
        bufs[key] = buf = torch.empty(rows, dtype=dtype, device=device)
    return buf

def seed_prev(backend, slots, mamba_cache_params, cache_steps: int) -> None:
    """Publish a freshly written temporal state to the handoff scratch rows."""
    if not handoff_enabled():
        return
    temporal = mamba_cache_params.temporal
    inter = getattr(mamba_cache_params, "intermediate_ssm", None)
    if inter is None or temporal is None:
        return
    st = state(backend)
    prev_buf = _prev_buf(backend, slots, int(temporal.shape[0]) + SLOT_BUFFER_SLACK)
    s64, n = _slot_ids(slots), int(slots.numel())
    flat = inter.reshape(-1, *inter.shape[2:])
    # One persistent scratch: a per-chunk temporary fragments the allocator.
    sc = st.seed_scratch
    if (sc is None or int(sc.shape[0]) < n
            or tuple(sc.shape[1:]) != tuple(temporal.shape[1:])
            or sc.dtype != temporal.dtype):
        st.seed_scratch = sc = torch.empty(
            (max(n, SEED_SCRATCH_MIN_ROWS), *temporal.shape[1:]),
            dtype=temporal.dtype, device=temporal.device)
    torch.index_select(temporal, 0, s64, out=sc[:n])
    flat.index_copy_(0, s64 * int(cache_steps), sc[:n])
    prev_buf.index_fill_(0, s64, 0)

def begin_verify_step(backend, layer_id: int) -> None:
    """Rotate the per-step cache at the start of a verify forward."""
    st, lid = state(backend), int(layer_id)
    if lid <= st.step_lid:
        st.step_gen += 1
        st.step_cache = None
    st.step_lid = lid

def shared_step_values(backend, slots, mamba_cache_params, cache_steps: int,
                       T: int) -> dict:
    """Per-batch tensors every layer of this verify forward uses unchanged."""
    st = state(backend)
    n = int(slots.numel())
    buf = _prev_buf(backend, slots,
                    int(mamba_cache_params.temporal.shape[0]) + SLOT_BUFFER_SLACK)
    rows = int(buf.numel())
    sh = st.step_cache
    if (sh is None or sh["gen"] != st.step_gen
            or sh["ptr"] != int(slots.data_ptr()) or sh["n"] != n):
        dev = slots.device
        s64 = _buf(st, "s64_buf", rows, rows, torch.int64, dev)[:n]
        s64.copy_(slots)
        torch.clamp_min(s64, 0, out=s64)
        prev32 = _buf(st, "prev32_buf", rows, rows, torch.int32, dev)[:n]
        torch.index_select(buf, 0, s64, out=prev32)
        store = _buf(st, "store_buf", rows, rows, torch.int32, dev)[:n]
        store.fill_(int(T) - 1)
        sh = {"gen": st.step_gen, "ptr": int(slots.data_ptr()), "n": n,
              "s64": s64, "prev": prev32, "store": store, "rows": {}}
        st.step_cache = sh
    ent = sh["rows"].get(int(cache_steps))
    if ent is None:
        ent = {"store_row": _step_store_row(st, sh, cache_steps)}
        ent["h0"] = _step_h0(st, sh["s64"], sh["prev"], cache_steps)
        sh["rows"][int(cache_steps)] = ent
    return sh

def _step_store_row(st, sh, cache_steps: int) -> torch.Tensor:
    n, rows = sh["n"], int(sh["prev"].numel())
    dev = sh["prev"].device
    tmp = _buf(st, "tmp32_buf", rows, rows, torch.int32, dev)[:n]
    row = _buf(st, "row_bufs", (int(cache_steps), rows), rows,
               torch.int32, dev)[:n]
    torch.add(sh["prev"], 1, out=tmp)
    torch.remainder(tmp, int(cache_steps), out=row)
    return row

def _step_h0(st, s64, prev32, cache_steps: int) -> torch.Tensor:
    n, dev = int(s64.numel()), s64.device
    prev64 = _buf(st, "prev64_buf", n, n, torch.int64, dev)[:n]
    prev64.copy_(prev32)
    torch.clamp_min(prev64, 0, out=prev64)
    tmp = _buf(st, "tmp64_buf", n, n, torch.int64, dev)[:n]
    torch.mul(s64, int(cache_steps), out=tmp)
    torch.add(tmp, prev64, out=tmp)
    h0 = _buf(st, "h0_bufs", (int(cache_steps), n), n, torch.int32, dev)[:n]
    h0.copy_(tmp)
    return h0

def step_h0(backend, slots, cache_steps: int):
    """The cached h0 rows for this batch, or None when this forward did not"""
    st = state(backend)
    sh = st.step_cache
    if (sh is None or sh["gen"] != st.step_gen
            or sh["ptr"] != int(slots.data_ptr()) or sh["n"] != int(slots.numel())):
        return None
    ent = sh["rows"].get(int(cache_steps))
    return None if ent is None else ent["h0"]

def h0_rows(slots: torch.Tensor, prev: torch.Tensor, row_stride: int) -> torch.Tensor:
    rows = _slot_ids(slots) * row_stride + prev.to(torch.int64).clamp_min(0)
    return rows.to(torch.int32)

def commit_prev(backend, state_indices, last_correct, T: int, cache_steps: int) -> bool:
    """Advance the committed row; no state copy."""
    if not handoff_enabled():
        return False
    st = state(backend)
    if st.prev is None or st.commit_failed:
        return False
    try:
        idx, last = state_indices.to(torch.int64), last_correct.to(torch.int64)
        # Rotate here: the forward keeps the capture batch's length, this sees live.
        prev = st.prev.index_select(0, idx.clamp_min(0)).to(torch.int64)
        keep = torch.where(last < T - 1, last,
                           torch.remainder(prev + 1, cache_steps)).to(torch.int32)
        # idx[valid] sizes its output on the device and stalls the host; route
        # invalid slots to the buffer's discard row instead.
        put = torch.where(idx >= 0, idx, torch.full_like(idx, st.prev.numel() - 1))
        st.prev.index_copy_(0, put, keep)
    except Exception:  # noqa: BLE001 - the upstream scatter is always correct
        st.commit_failed = True
        logger.warning("[gdn-bplus] commit_prev failed; falling back to scatter",
                       exc_info=True)
        return False
    if not st.logged:
        st.logged = True
        logger.info("[gdn-bplus] handoff commit active (scatter skipped)")
    return True

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
    """Write this step's snapshot row and hand the truncated rows to ``stash``.

    Returns the kernel output, or None when the chain declined.
    """
    if not stash(backend, layer, q, k, v, a, b, draft_token_num):
        return None
    flat = snap_cache.reshape(-1, *snap_cache.shape[2:])
    store_arg = store_row_arg = None
    st = state(backend)
    if handoff_enabled():
        # Slot-indexed scratch rows give the committed row a stable address,
        # so commit only moves an int32 pointer; stride is the allocated
        # cache_steps, not the runtime draft-token count.
        cache_steps = int(snap_cache.shape[1])
        snap_idx = cache_indices[:n].to(torch.int32)
        begin_verify_step(backend, layer.layer_id)
        sh = shared_step_values(backend, snap_idx, mamba_cache, cache_steps,
                                int(draft_token_num))
        row = sh["rows"][cache_steps]
        h0_src, h0_idx, prev_arg = flat, row["h0"], st.prev
        store_arg, store_row_arg = sh["store"], row["store_row"]
    else:
        snap_idx = intermediate_indices[:n].to(torch.int32)
        h0_src, h0_idx, prev_arg = ss_states, cache_indices[:n].to(torch.int32), None
    st.snap_idx = snap_idx
    out = q.new_empty((1, q.shape[1], v.shape[2], v.shape[3]))
    gdn_recurrent.verify(
        q, k, v, a, b, layer.A_log, layer.dt_bias, query_start_loc[: n + 1],
        h0_src, h0_idx, flat, snap_idx, int(snap_cache.shape[1]),
        int(draft_token_num), out,
        prev_step=prev_arg, store=store_arg, store_row=store_row_arg)
    return out

def compensate_from_stash(backend, state_indices, last_correct) -> bool:
    """Replay the stash for the truncated slots, before the upstream scatter."""
    if not enabled():
        return False
    st = state(backend)
    if not st.stash or st.snap_idx is None:
        return False
    handoff = handoff_enabled()
    try:
        # Graph replay skips verify's Python: prefer the live value.
        T = int(st.live_T or st.capture_T)
        if T <= 1:
            return False
        n, idx32 = int(state_indices.shape[0]), state_indices.to(torch.int32)
        if handoff:
            snap32 = idx32
        else:
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
        prev_now = None
        if handoff and st.prev is None:
            return False
        skip_full = os.environ.get("SGLANG_GDN_HIP_BPLUS_SKIP_FULL", "1") != "0"
        # A full accept already committed verify's row, so replay only the
        # truncated slots. ``neg`` is built unconditionally: ``torch.where``
        # rejects None, and the per-layer except would drop every layer.
        # With the switch off a full accept still replays.
        neg = torch.full_like(idx32, -1) if skip_full else idx32
        h0_by_stride, h0_plain, rows = {}, None, []
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
                if handoff:
                    ssm = flat
                    h0_idx = h0_by_stride.get(stride)
                    if h0_idx is None:
                        h0_idx = step_h0(backend, idx32, stride)
                        if h0_idx is None:
                            if prev_now is None:
                                prev_now = st.prev.index_select(
                                    0, idx32.to(torch.int64).clamp_min(0)
                                ).to(torch.int32)
                            h0_idx = h0_rows(idx32, prev_now, stride)
                        if skip_full:
                            h0_idx = torch.where(trunc, h0_idx, neg)
                        h0_by_stride[stride] = h0_idx
                else:
                    ssm = layer_cache.temporal
                    if h0_plain is None:
                        h0_plain = torch.where(trunc, idx32, neg)
                    h0_idx = h0_plain
                rows.append((ent, ssm, flat, h0_idx, stride, lid))
            except Exception:  # noqa: BLE001 - optional path
                logger.warning("[gdn-bplus] compensate failed for layer %s",
                               lid, exc_info=True)
                st.commit_failed = True
        if not rows:
            return False
        if (os.environ.get("SGLANG_GDN_HIP_BPLUS_MULTI", "1") != "0"
                and gdn_recurrent.has_multi()):
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
