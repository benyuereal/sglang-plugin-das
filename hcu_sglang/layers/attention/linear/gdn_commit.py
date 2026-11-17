"""Commit only the accepted step's GDN state.

Verify keeps a single snapshot row; the commit points the next verify at it and
replays the stash for the slots the accept truncated, so the 6 GiB state copy
is not needed. With the radix cache on, the commit also scatters the radix
track rows: they live outside the committed row and would otherwise go stale.

Handoff state is request-private, so it lives on the request's speculative
scratch row (``intermediate_ssm[row]``), not on the persistent mamba slot:
under a radix cache the slot ids span the whole mamba pool (max checkpoints)
while the scratch has one row per live request. The request pool row is stable
for a request's lifetime and both pools are sized ``max_num_reqs + 1``, so it
is the scratch-row id here. Persistent slots stay the destination of the conv
state and of the radix track scatter.

"B+" is the original name of this path and survives in the kernel entry points;
the module is named after what it does.
"""

import logging
from typing import NamedTuple

import torch
from sglang.srt.layers.attention.linear.utils import pp_spec_stable_rows_enabled

from hcu_sglang.layers.attention.linear import gdn_recurrent
from hcu_sglang.layers.attention.linear import gdn_switches as switches

# Row floor for the seed scratch: a one-slot chunk must not reallocate next chunk.
SEED_SCRATCH_MIN_ROWS = 128

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
        "tmp64_buf", "gather_buf", "step_ok", "disabled", "geometry",
        "rejected", "step_rows", "step_rows_gen",
    )

    def __init__(self):
        self.prev = self.stash = self.snap_idx = None
        self.seed_scratch = self.ml_table = self.ml_key = self.step_cache = None
        self.capture_T = self.live_T = self.max_slots = self.step_gen = 0
        self.step_lid = -1
        self.logged = self.commit_failed = self.step_ok = False
        self.disabled = False
        self.geometry = None
        self.rejected = False
        self.step_rows = None
        self.step_rows_gen = None
        for name in ("s64_buf", "prev32_buf", "prev64_buf", "store_buf",
                     "row_bufs", "h0_bufs", "tmp32_buf", "tmp64_buf",
                     "gather_buf"):
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
    return switches.enabled()

def _head_dims(*tensors) -> list[int]:
    return [int(t.shape[2]) * int(t.shape[3]) for t in tensors]

def _prev_buf(backend, rows, size: int) -> torch.Tensor:
    st = state(backend)
    if st.prev is None or int(st.prev.numel()) < size:
        st.prev = torch.full((size,), -1, dtype=torch.int32, device=rows.device)
    return st.prev

def _buf(st, name: str, key, rows: int, dtype, device) -> torch.Tensor:
    """Persistent scratch keyed per size: captures bake buffer addresses."""
    bufs = getattr(st, name)
    buf = bufs.get(key)
    if buf is None or buf.dtype != dtype or buf.device != device:
        bufs[key] = buf = torch.empty(rows, dtype=dtype, device=device)
    return buf

def _rows_capacity(mamba_cache_params) -> int:
    """Number of request scratch rows in the speculative intermediate buffer."""
    inter = getattr(mamba_cache_params, "intermediate_ssm", None)
    return 0 if inter is None else int(inter.shape[0])

def _geometry_ok(backend, capacity: int) -> bool:
    """One handoff row per request-pool row: index the scratch by pool row."""
    st = state(backend)
    if st.geometry is not None:
        return st.geometry
    pool = getattr(backend, "req_to_token_pool", None)
    ok = False
    if pool is not None and capacity > 0:
        size = int(getattr(pool, "size", 0))
        alloc = int(getattr(pool, "_alloc_size", size + 1))
        ok = size > 0 and alloc <= capacity and size + 1 <= capacity
    if not ok:
        st.disabled = True
        logger.error(
            "[gdn-bplus] handoff disabled: request rows and speculative "
            "scratch disagree (pool_size=%s alloc=%s scratch_rows=%s)",
            getattr(pool, "size", None),
            getattr(pool, "_alloc_size", None), capacity)
    st.geometry = ok
    return ok

def _reject(st, where: str, detail: str) -> None:
    if not st.rejected:
        st.rejected = True
        logger.warning("[gdn-bplus] %s refused, falling back to the plain "
                       "scatter: %s", where, detail)

def _disable(st, where: str, detail: str) -> None:
    """Give the whole handoff up: a partial step cannot be half-handoff."""
    st.disabled = True
    _reject(st, where, detail)

def request_rows(rows, capacity: int, n: int):
    """Map request-pool rows onto handoff scratch rows (-1 = no handoff row).

    Pure device work: cuda-graph replay re-evaluates it from the static row
    buffer, so it must not branch on tensor values.
    """
    if rows is None or capacity <= 0 or n <= 0 or int(rows.numel()) < n:
        return None
    rows = rows[:n].to(torch.int32)
    ok = (rows >= 0) & (rows < int(capacity))
    return torch.where(ok, rows, torch.full_like(rows, -1))

def batch_rows(backend, forward_batch):
    """This batch's handoff scratch rows.

    ``handoff_rows_list`` is a static per-bs buffer refreshed in place by the
    attention metadata path (eager forward and cuda-graph replay alike), so a
    captured verify kernel reads the live request rows.
    """
    bufs = getattr(backend, "handoff_rows_list", None)
    bs = int(forward_batch.batch_size)
    if bufs is None or bs <= 0 or bs > len(bufs):
        return forward_batch.req_pool_indices
    return bufs[bs - 1][:bs]

def seed_prev(backend, slots, mamba_cache_params, cache_steps: int,
              req_rows=None) -> bool:
    """Publish a freshly written temporal state to the request scratch rows."""
    if not enabled():
        return False
    temporal = mamba_cache_params.temporal
    inter = getattr(mamba_cache_params, "intermediate_ssm", None)
    if inter is None or temporal is None or req_rows is None:
        return False
    capacity = int(inter.shape[0])
    st = state(backend)
    if st.disabled or not _geometry_ok(backend, capacity):
        return False
    n = int(slots.numel())
    if n <= 0 or int(req_rows.numel()) < n:
        return False
    rows = req_rows[:n].to(torch.int64)
    slots64 = slots.to(torch.int64)
    # Prefill path: a host check here costs one sync per chunk, and an
    # out-of-range row must never reach the scatter kernel.
    if (int(rows.min()) < 0 or int(rows.max()) >= capacity
            or int(slots64.min()) < 0 or int(slots64.max()) >= int(temporal.shape[0])):
        _reject(st, "seed",
                f"rows[{int(rows.min())},{int(rows.max())}] cap={capacity} "
                f"slots[{int(slots64.min())},{int(slots64.max())}] "
                f"temporal={int(temporal.shape[0])}")
        return False
    prev_buf = _prev_buf(backend, rows, capacity)
    flat = inter.reshape(-1, *inter.shape[2:])
    # One persistent scratch: a per-chunk temporary fragments the allocator.
    sc = st.seed_scratch
    if (sc is None or int(sc.shape[0]) < n
            or tuple(sc.shape[1:]) != tuple(temporal.shape[1:])
            or sc.dtype != temporal.dtype):
        st.seed_scratch = sc = torch.empty(
            (max(n, SEED_SCRATCH_MIN_ROWS), *temporal.shape[1:]),
            dtype=temporal.dtype, device=temporal.device)
    torch.index_select(temporal, 0, slots64, out=sc[:n])
    flat.index_copy_(0, rows * int(cache_steps), sc[:n])
    prev_buf.index_fill_(0, rows, 0)
    return True

def resync_plain(backend, rows, cache_indices, mamba_cache, cache_steps: int,
                 n: int) -> None:
    """Copy the committed handoff rows back into the persistent slots.

    Runs once when the handoff declines after it had taken over state, so the
    plain full scatter starts from the same state the handoff held.
    """
    st = state(backend)
    inter = getattr(mamba_cache, "intermediate_ssm", None)
    temporal = getattr(mamba_cache, "temporal", None)
    if inter is None or temporal is None or st.prev is None or rows is None:
        return
    rows32 = request_rows(rows, int(inter.shape[0]), n)
    if rows32 is None:
        return
    idx = rows32.to(torch.int64)
    safe = idx.clamp(0, int(inter.shape[0]) - 1)
    prev = st.prev.index_select(0, safe).to(torch.int64)
    slots = cache_indices[:n].to(torch.int64)
    live = (idx >= 0) & (prev >= 0) & (slots >= 0) & (
        slots < int(temporal.shape[0]))
    src = torch.where(live, safe * int(cache_steps) + prev.clamp_min(0),
                      torch.zeros_like(safe))
    flat = inter.reshape(-1, *inter.shape[2:])
    dst = torch.where(live, slots, torch.zeros_like(slots))
    temporal.index_copy_(0, dst, flat.index_select(0, src))
    # A synced row is no longer handoff state: the plain path owns it from now
    # on, so a repeated decline does not copy it again.
    st.prev.index_fill_(0, safe[live], -1)

def begin_verify_step(backend, layer_id: int) -> None:
    """Rotate the per-step cache at the start of a verify forward."""
    st, lid = state(backend), int(layer_id)
    if lid <= st.step_lid:
        st.step_gen += 1
        st.step_cache = None
        st.step_ok = False
    st.step_lid = lid

def shared_step_values(backend, rows, mamba_cache_params, cache_steps: int,
                       T: int) -> dict:
    """Per-batch tensors every layer of this verify forward uses unchanged."""
    st = state(backend)
    n = int(rows.numel())
    capacity = _rows_capacity(mamba_cache_params)
    buf = _prev_buf(backend, rows,
                    capacity if capacity > 0 else int(rows.numel()))
    rows_n = int(buf.numel())
    sh = st.step_cache
    if (sh is None or sh["gen"] != st.step_gen
            or sh["ptr"] != int(rows.data_ptr()) or sh["n"] != n):
        dev = rows.device
        s64 = _buf(st, "s64_buf", rows_n, rows_n, torch.int64, dev)[:n]
        s64.copy_(rows)
        gather = _buf(st, "gather_buf", rows_n, rows_n, torch.int64, dev)[:n]
        torch.clamp_min(s64, 0, out=gather)
        prev32 = _buf(st, "prev32_buf", rows_n, rows_n, torch.int32, dev)[:n]
        torch.index_select(buf, 0, gather, out=prev32)
        store = _buf(st, "store_buf", rows_n, rows_n, torch.int32, dev)[:n]
        store.fill_(int(T) - 1)
        valid = (s64 >= 0) & (prev32 >= 0)
        sh = {"gen": st.step_gen, "ptr": int(rows.data_ptr()), "n": n,
              "s64": s64, "gather": gather, "prev": prev32, "valid": valid,
              "store": store, "rows": {}}
        st.step_cache = sh
    ent = sh["rows"].get(int(cache_steps))
    if ent is None:
        ent = {"store_row": _step_store_row(st, sh, cache_steps)}
        ent["h0"] = _step_h0(st, sh, cache_steps)
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
    # Unseeded rows carry no snapshot; the kernel skips row < 0.
    torch.where(sh["valid"], row, torch.full_like(row, -1), out=row)
    return row

def _step_h0(st, sh, cache_steps: int) -> torch.Tensor:
    n, dev = sh["n"], sh["s64"].device
    prev64 = _buf(st, "prev64_buf", n, n, torch.int64, dev)[:n]
    prev64.copy_(sh["prev"])
    torch.clamp_min(prev64, 0, out=prev64)
    tmp = _buf(st, "tmp64_buf", n, n, torch.int64, dev)[:n]
    torch.mul(sh["gather"], int(cache_steps), out=tmp)
    torch.add(tmp, prev64, out=tmp)
    h0 = _buf(st, "h0_bufs", (int(cache_steps), n), n, torch.int32, dev)[:n]
    h0.copy_(tmp)
    torch.where(sh["valid"], h0, torch.full_like(h0, -1), out=h0)
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

def h0_rows(rows: torch.Tensor, prev: torch.Tensor, row_stride: int) -> torch.Tensor:
    return (rows.to(torch.int64) * row_stride
            + prev.to(torch.int64).clamp_min(0)).to(torch.int32)

def commit_prev(backend, rows, last_correct, T: int, cache_steps: int) -> bool:
    """Advance the committed row; no state copy."""
    if not enabled():
        return False
    st = state(backend)
    if st.prev is None or st.commit_failed or st.disabled or not st.step_ok:
        return False
    try:
        cap = int(st.prev.numel())
        idx = rows.to(torch.int64)
        safe = idx.clamp(0, cap - 1)
        # Rotate here: the forward keeps the capture batch's length, this sees live.
        prev = st.prev.index_select(0, safe).to(torch.int64)
        keep = torch.where(last_correct.to(torch.int64) < T - 1,
                           last_correct.to(torch.int64),
                           torch.remainder(prev + 1, cache_steps))
        keep = keep.to(torch.int32)
        # Only rows that hold a handoff state advance; the rest stay unseeded.
        live = (idx >= 0) & (idx < cap) & (prev >= 0)
        torch.where(live, keep, torch.full_like(keep, -1), out=keep)
        st.prev.index_copy_(0, safe, keep)
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
                cache_indices, mamba_cache, query_start_loc, req_rows=None):
    """Write this step's snapshot row and hand the truncated rows to ``stash``.

    Returns the kernel output, or None when the chain declined.
    """
    begin_verify_step(backend, layer.layer_id)
    st = state(backend)
    # Pipeline-parallel stable rows are request rows, but their commit needs
    # the plain scatter; keep the whole path on one row space.
    if st.disabled or pp_spec_stable_rows_enabled():
        return None
    flat = snap_cache.reshape(-1, *snap_cache.shape[2:])
    # Request-indexed scratch rows carry the committed state, so the commit only
    # moves an int32 pointer; stride is the allocated cache_steps, not the
    # runtime draft-token count.
    cache_steps = int(snap_cache.shape[1])
    capacity = int(snap_cache.shape[0])
    if not _geometry_ok(backend, capacity):
        return None
    # One batch-row map per verify forward, not one per GDN layer: a fresh
    # tensor per layer misses ``shared_step_values``' pointer cache and bakes
    # the whole per-step setup into the captured graph 48 times.
    rows_ptr = int(req_rows.data_ptr()) if req_rows is not None else 0
    key = (st.step_gen, n, capacity, rows_ptr)
    if st.step_rows is None or st.step_rows_gen != key:
        snap_idx = request_rows(req_rows, capacity, n)
        st.step_rows, st.step_rows_gen = snap_idx, key
    else:
        snap_idx = st.step_rows
    if snap_idx is None:
        _disable(st, "verify", "no request rows")
        return None
    if not stash(backend, layer, q, k, v, a, b, draft_token_num):
        _disable(st, "verify", "stash")
        return None
    sh = shared_step_values(backend, snap_idx, mamba_cache, cache_steps,
                            int(draft_token_num))
    row = sh["rows"][cache_steps]
    st.snap_idx = snap_idx
    st.step_ok = True
    out = q.new_empty((1, q.shape[1], v.shape[2], v.shape[3]))
    gdn_recurrent.verify(
        q, k, v, a, b, layer.A_log, layer.dt_bias, query_start_loc[: n + 1],
        flat, row["h0"], flat, snap_idx, cache_steps,
        int(draft_token_num), out,
        prev_step=st.prev, store=sh["store"], store_row=row["store_row"])
    return out

def compensate_from_stash(backend, rows, last_correct, track_steps=None) -> bool:
    """Replay the stash for the truncated rows, before the upstream scatter.

    With ``track_steps``, also materialize the radix interval-crossing snapshot
    at the step that crossed it: the handoff row keeps one committed state, so
    the per-step snapshot the track scatter reads does not exist otherwise.
    """
    if not enabled():
        return False
    st = state(backend)
    if not st.step_ok or not st.stash or st.prev is None or rows is None:
        return False
    try:
        # Graph replay skips verify's Python: prefer the live value.
        T = int(st.live_T or st.capture_T)
        if T <= 1:
            return False
        n = int(rows.shape[0])
        row32 = rows[:n].to(torch.int32)
        last32 = last_correct.to(torch.int32)
        idx64 = row32.to(torch.int64).clamp_min(0)
        prev32 = st.prev.index_select(0, idx64).to(torch.int32)
        live = (row32 >= 0) & (prev32 >= 0)
        trunc = live & (last32 >= 0) & (last32 < T - 1)
        store_step = torch.where(trunc, last32, torch.full_like(last32, -1))
        # Rows without an interval crossing keep h0 < 0 on this pass, so the
        # kernel exits immediately; no host read is needed to skip it.
        track, track_step = None, None
        if track_steps is not None:
            track32 = track_steps[:n].to(torch.int32)
            track = live & (track32 >= 0) & (track32 < T)
            track_step = torch.where(track, track32,
                                     torch.full_like(track32, -1))
        D, m = T, n * T
        # A full accept already committed verify's row, so replay only the
        # truncated rows. ``neg`` is built unconditionally: ``torch.where``
        # rejects None.
        neg = torch.full_like(row32, -1)
        h0_by_stride, h0_trunc_by_stride, prev_now = {}, {}, None
        layer_rows, track_rows = [], []
        track_h0_by_stride = {}
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
                h0_idx = h0_by_stride.get(stride)
                if h0_idx is None:
                    h0_idx = step_h0(backend, row32, stride)
                    if h0_idx is None:
                        if prev_now is None:
                            prev_now = st.prev.index_select(
                                0, idx64
                            ).to(torch.int32)
                        h0_idx = h0_rows(row32, prev_now, stride)
                        h0_idx = torch.where(live, h0_idx, neg)
                    h0_by_stride[stride] = h0_idx
                    # Only truncated rows need a replay: a full accept's
                    # snapshot already sits on the committed column, and a
                    # live-but-unreplayed row would run the whole recurrence
                    # for a write it never makes (store_step < 0).
                    h0_trunc_by_stride[stride] = torch.where(trunc, h0_idx, neg)
                layer_rows.append(
                    (ent, flat, h0_trunc_by_stride[stride], stride, lid))
                if track_step is not None:
                    track_h0 = track_h0_by_stride.get(stride)
                    if track_h0 is None:
                        track_h0 = torch.where(track, h0_idx, neg)
                        track_h0_by_stride[stride] = track_h0
                    track_rows.append((ent, flat, track_h0, stride, lid))
            except Exception:  # noqa: BLE001 - optional path
                logger.warning("[gdn-bplus] compensate failed for layer %s",
                               lid, exc_info=True)
                st.commit_failed = True
        if not layer_rows:
            return False
        if not _replay_rows(st, layer_rows, row32, n, D, m, store_step):
            return False
        if track_rows:
            if not _replay_rows(st, track_rows, row32, n, D, m, track_step):
                return False
        return True
    except Exception:  # noqa: BLE001
        logger.warning("[gdn-bplus] compensate_from_stash failed", exc_info=True)
        return False

def _replay_rows(st, layer_rows, snap32, n, D, m, store_step) -> bool:
    """One state-replay pass over every layer for the given per-row steps."""
    if gdn_recurrent.has_multi():
        try:
            if _compensate_multi(st, layer_rows, n, D, m, store_step, snap32):
                return True
        except Exception:  # noqa: BLE001 - per-layer path is the fallback
            logger.warning("[gdn-bplus] multi compensate failed", exc_info=True)
    for ent, flat, h0_idx, cache_steps, lid in layer_rows:
        try:
            dims = HeadDims.from_entry(ent)
            gdn_recurrent.compensate(
                ent["q"][:m].view(n, D, *dims[:2]),
                ent["k"][:m].view(n, D, *dims[2:4]),
                ent["v"][:m].view(n, D, *dims[4:]),
                ent["a"][:m].view(n, D, ent["a"].shape[-1]),
                ent["b"][:m].view(n, D, ent["b"].shape[-1]),
                ent["A_log"], ent["dt_bias"], flat, h0_idx, flat, snap32,
                cache_steps, store_step, n, D, ent["out"])
        except Exception:  # noqa: BLE001 - optional path
            logger.warning("[gdn-bplus] compensate failed for layer %s",
                           lid, exc_info=True)
            st.commit_failed = True
    return True

def _compensate_multi(st, rows, n, D, m, store_step, snap32) -> bool:
    """One launch covers every layer when their layout matches the table."""
    if len(rows) < 2:
        return False
    ent0, h0_idx = rows[0][0], rows[0][2]
    dims, cache_steps = HeadDims.from_entry(ent0), rows[0][3]
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
        for ent, flat, h0_idx_i, cs, _ in rows:
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
                          flat.data_ptr(), ent["out"].data_ptr(), flat.data_ptr(),
                          *strides])
        table = torch.tensor(cells, dtype=torch.int64, device=ent0["q"].device)
        st.ml_table, st.ml_key = table, key
    gdn_recurrent.compensate_multi(table, h0_idx, snap32, cache_steps,
                                   store_step, n, D, vh, qh, kk, vv)
    return True
