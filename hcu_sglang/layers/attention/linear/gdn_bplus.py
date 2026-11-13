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
    tdev = getattr(backend, "_gdn_hip_tdev", None)
    snap_idx = getattr(backend, "_gdn_hip_snap_idx", None)
    if tdev is None or snap_idx is None:
        return False
    try:
        T = int(tdev.item())
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
        h0_rows = idx32
        snap32 = snap_idx.to(torch.int32)
        D = T
        m = n * D
        for lid, ent in list(ents.items()):
            try:
                layer_cache = backend.req_to_token_pool.mamba2_layer_cache(lid)
                inter = layer_cache.intermediate_ssm
                if inter is None:
                    continue
                flat = inter.reshape(-1, *inter.shape[2:])
                ssm = layer_cache.temporal
                gdn_hip_bplus.compensate(
                    ent["q"][:m].view(n, D, ent["qh"], ent["qk"]),
                    ent["k"][:m].view(n, D, ent["kh"], ent["kk"]),
                    ent["v"][:m].view(n, D, ent["vh"], ent["vv"]),
                    ent["a"][:m].view(n, D, -1),
                    ent["b"][:m].view(n, D, -1),
                    ent["A_log"],
                    ent["dt_bias"],
                    ssm,
                    h0_rows,
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
