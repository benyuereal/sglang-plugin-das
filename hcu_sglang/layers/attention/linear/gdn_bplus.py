"""GDN B+ commit chain: verify stores only the last snapshot row, and the
commit replays the stash for the truncated slots. The handoff variant points
verify at the committed row, valid only with --disable-radix-cache.
"""

import logging
import os

import torch

from hcu_sglang.constants import SEED_SCRATCH_MIN_ROWS, SLOT_BUFFER_SLACK

logger = logging.getLogger(__name__)


class BplusState:
    """Per-backend scratch, kept off the nn.Module so plain attributes work."""

    __slots__ = (
        "prev",
        "stash",
        "snap_idx",
        "capture_T",
        "live_T",
        "slots_now",
        "prev_now",
        "store_row",
        "cache_steps",
        "seed_scratch",
        "tdev",
        "max_slots",
        "logged",
        "commit_failed",
    )

    def __init__(self):
        self.prev = None
        self.stash = None
        self.snap_idx = None
        self.capture_T = 0
        self.live_T = 0
        self.slots_now = None
        self.prev_now = None
        self.store_row = None
        self.cache_steps = 0
        self.seed_scratch = None
        self.tdev = None
        self.max_slots = 0
        self.logged = False
        self.commit_failed = False


def state(backend) -> BplusState:
    st = getattr(backend, "_gdn_bplus", None)
    if st is None:
        st = BplusState()
        object.__setattr__(backend, "_gdn_bplus", st)
    return st


def enabled() -> bool:
    return os.environ.get("SGLANG_GDN_HIP_BPLUS") == "1"


_HANDOFF_OK = None


def handoff_enabled() -> bool:
    """Zero-copy state handoff: verify reads h0 from the committed row."""
    global _HANDOFF_OK
    if _HANDOFF_OK is None:
        _HANDOFF_OK = (
            enabled()
            and os.environ.get("SGLANG_GDN_HIP_HANDOFF") == "1"
            and os.environ.get("SGLANG_GDN_HIP_HANDOFF_SKIP_TRACK") == "1"
        )
    return _HANDOFF_OK


def _prev_buf(backend, slots, size: int) -> torch.Tensor:
    st = state(backend)
    if st.prev is None or int(st.prev.numel()) < size:
        st.prev = torch.full((size,), -1, dtype=torch.int32, device=slots.device)
    return st.prev


def ensure_prev(backend, slots: torch.Tensor, mamba_cache_params, D: int):
    """Return the committed step for this batch's slots."""
    del D  # row stride belongs to seed_prev / h0_rows
    st = state(backend)
    size = int(mamba_cache_params.temporal.shape[0]) + SLOT_BUFFER_SLACK
    prev_buf = _prev_buf(backend, slots, size)
    s64 = slots.to(torch.int64).clamp_min(0)
    return prev_buf.index_select(0, s64).to(torch.int32)


def seed_prev(
    backend, slots: torch.Tensor, mamba_cache_params, cache_steps: int
) -> None:
    """Publish a freshly written temporal state to the handoff scratch rows."""
    if not handoff_enabled():
        return
    temporal = mamba_cache_params.temporal
    inter = getattr(mamba_cache_params, "intermediate_ssm", None)
    if inter is None or temporal is None:
        return
    st = state(backend)
    prev_buf = _prev_buf(backend, slots, int(temporal.shape[0]) + SLOT_BUFFER_SLACK)
    s64 = slots.to(torch.int64).clamp_min(0)
    flat = inter.reshape(-1, *inter.shape[2:])
    # One persistent scratch: a per-chunk temporary fragments the allocator.
    n = int(s64.numel())
    if (
        st.seed_scratch is None
        or int(st.seed_scratch.shape[0]) < n
        or tuple(st.seed_scratch.shape[1:]) != tuple(temporal.shape[1:])
        or st.seed_scratch.dtype != temporal.dtype
    ):
        st.seed_scratch = torch.empty(
            (max(n, SEED_SCRATCH_MIN_ROWS), *temporal.shape[1:]),
            dtype=temporal.dtype,
            device=temporal.device,
        )
    torch.index_select(temporal, 0, s64, out=st.seed_scratch[:n])
    flat.index_copy_(0, s64 * int(cache_steps), st.seed_scratch[:n])
    prev_buf.index_fill_(0, s64, 0)


def invalidate_prev(backend, slots: torch.Tensor) -> None:
    """Drop the handoff pointer for slots whose state was rewritten by prefill."""
    if not handoff_enabled():
        return
    buf = state(backend).prev
    if buf is None:
        return
    buf.index_fill_(0, slots.to(torch.int64).clamp_min(0), -1)


def remember_step(backend, slots: torch.Tensor, prev: torch.Tensor, cache_steps: int):
    """Record the step's slots and rows so the commit can rotate the pointer.

    ``cache_steps`` is the allocated row stride (``intermediate_ssm.shape[1]``),
    not the runtime draft-token count: rows are addressed by the stride.
    """
    st = state(backend)
    store_row = torch.remainder(prev + 1, cache_steps).to(torch.int32)
    st.slots_now = slots.to(torch.int32)
    st.prev_now = prev.to(torch.int32)
    st.store_row = store_row
    st.cache_steps = int(cache_steps)
    return store_row


def h0_rows(slots: torch.Tensor, prev: torch.Tensor, D: int) -> torch.Tensor:
    rows = slots.to(torch.int64).clamp_min(0) * D + prev.to(torch.int64).clamp_min(0)
    return rows.to(torch.int32)


def commit_prev(backend, state_indices, last_correct, T: int, cache_steps: int) -> bool:
    """Advance the committed row; no state copy."""
    if not handoff_enabled():
        return False
    st = state(backend)
    if st.prev is None or st.commit_failed:
        return False
    try:
        _commit_prev(backend, st, state_indices, last_correct, T, cache_steps)
    except Exception:  # noqa: BLE001 - the upstream scatter is always correct
        st.commit_failed = True
        logger.warning("[gdn-bplus] commit_prev failed; falling back to scatter",
                       exc_info=True)
        return False
    if not st.logged:
        st.logged = True
        print("[gdn-bplus] handoff commit active (scatter skipped)", flush=True)
    return True


def _commit_prev(backend, st, state_indices, last_correct, T: int, cache_steps: int):
    idx = state_indices.to(torch.int64)
    last = last_correct.to(torch.int64)
    # Rows are rotated here, not in the forward: the forward's Python tensors
    # keep the capture batch's length, the commit sees the live batch.
    prev = st.prev.index_select(0, idx.clamp_min(0)).to(torch.int64)
    store_row = torch.remainder(prev + 1, cache_steps)
    keep = torch.where(last < T - 1, last, store_row).to(torch.int32)
    # Boolean-mask indexing (idx[valid]) sizes its output on the device, which
    # stalls the host behind the whole decode step; route invalid slots to the
    # buffer's discard row instead.
    discard = torch.full_like(idx, st.prev.numel() - 1)
    put_idx = torch.where(idx >= 0, idx, discard)
    st.prev.index_copy_(0, put_idx, keep)


def _same_storage_in_order(tensors, offsets) -> bool:
    base = tensors[0].untyped_storage().data_ptr()
    esz = tensors[0].element_size()
    for t, off in zip(tensors, offsets):
        if t.untyped_storage().data_ptr() != base:
            return False
        if int(t.data_ptr()) - int(tensors[0].data_ptr()) != off * esz:
            return False
    return True


def _fused_qkv_view(q, k, v, n: int, T: int):
    """[n, T, W] view over one fused q|k|v buffer, or None when they are apart."""
    qd, kd, vd = (int(t.shape[2]) * int(t.shape[3]) for t in (q, k, v))
    W = int(q.stride(1))
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return None
    if W != qd + kd + vd:
        return None
    if any(int(t.stride(1)) != W for t in (k, v)):
        return None
    # Row-major within the head dim: the head axis sits immediately above it.
    if any(
        int(t.stride(3)) != 1 or int(t.stride(2)) != int(t.shape[3])
        for t in (q, k, v)
    ):
        return None
    if not _same_storage_in_order((q, k, v), (0, qd, qd + kd)):
        return None
    if int(q.shape[1]) != n * T:
        return None
    return torch.as_strided(q, (n, T, W), (T * W, W, 1))


def _max_slots(backend) -> int:
    st = state(backend)
    if st.max_slots <= 0:
        st.max_slots = int(backend.req_to_token_pool.size)
    return st.max_slots


def _stash_entries(backend):
    st = state(backend)
    if st.stash is None:
        st.stash = {}
    return st.stash


def _ensure(backend, layer, q, k, v, a, b, D: int):
    ents = _stash_entries(backend)
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
        qkv = torch.zeros((rows, qd + kd + vd), dtype=q.dtype, device=dev)
        ent = {
            "rows": rows,
            "qh": q.shape[2], "qk": q.shape[3],
            "kh": k.shape[2], "kk": k.shape[3],
            "vh": v.shape[2], "vv": v.shape[3],
            "qkv": qkv,
            "q": qkv[:, :qd],
            "k": qkv[:, qd:qd + kd],
            "v": qkv[:, qd + kd:],
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
    st = state(backend)
    # Host-side int: the commit needs T without a device sync.
    st.capture_T = int(draft_token_num)
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
        src = _fused_qkv_view(q, k, v, n, T)
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
        if st.tdev is None:
            st.tdev = torch.zeros(1, dtype=torch.int32, device=q.device)
        st.tdev.fill_(T)
        return True
    except Exception:  # noqa: BLE001 - never take serving down
        logger.warning("[gdn-bplus] stash failed", exc_info=True)
        return False


def compensate_from_stash(
    backend, state_indices, last_correct, mamba_caches=None
) -> bool:
    """Replay the stash for the truncated slots, before the upstream scatter."""
    if not enabled():
        return False
    st = state(backend)
    if not st.stash or st.snap_idx is None:
        return False
    handoff = handoff_enabled()
    try:
        # Graph replay skips the verify forward's Python, so anything it
        # recorded describes the last captured bucket; prefer the live value.
        T = int(st.live_T or st.capture_T)
        if T <= 1:
            return False
        n = int(state_indices.shape[0])
        if handoff:
            snap32 = state_indices.to(torch.int32)
        else:
            from sglang.srt.layers.attention.linear.utils import (
                pp_spec_stable_rows_enabled,
            )

            if pp_spec_stable_rows_enabled():
                if int(st.snap_idx.numel()) != n:
                    return False
                snap32 = st.snap_idx.to(torch.int32)
            else:
                # Default verify rows are batch positions; rebuild them for the
                # live batch instead of reusing the capture bucket's view.
                snap32 = torch.arange(
                    n, dtype=torch.int32, device=state_indices.device
                )
        from hcu_sglang.layers.attention.linear import gdn_hip_bplus

        idx32 = state_indices.to(torch.int32)
        last32 = last_correct.to(torch.int32)
        trunc = (idx32 > 0) & (last32 >= 0) & (last32 < T - 1)
        store_step = torch.where(
            trunc, last32, torch.full_like(last32, -1)
        ).to(torch.int32)
        h0_idx_default = idx32
        D = T
        m = n * D
        if handoff:
            if st.prev is None:
                return False
            slots_now = state_indices.to(torch.int32)
            prev_now = st.prev.index_select(
                0, slots_now.to(torch.int64).clamp_min(0)
            ).to(torch.int32)
        for lid, ent in list(st.stash.items()):
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
