"""HIP GDN recurrent kernel with B+ (write only the committed step).

B+ omits writes, not math: the chain, its reduction order and the accept rule
are unchanged, and the row it does write matches the full-snapshot kernel bit
for bit. This kernel is not bit-identical to aiter: three K=128 fp32 reductions
are regrouped, giving <=64 ULP and a non-degrading match rate.
"""
import ctypes
import os

import torch

# 默认恢复为 **gdn_exp.so**（B+ 原本使用的核）。逐位版 gdn_bw_ship.so 只给
# kernel-only 那条线用（SGLANG_GDN_HIP_KERNEL=1 时由调用方显式指定），
# 不再作为这里的默认值 —— 2026-11-11 起 B+ 与 kernel-only 的核彻底解耦。
_DEFAULT_SO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lib", "gdn_bw_ship.so"
)
_BV = 32  # 与 gdn_recurrent.hip 的 BV 一致

# 13 个指针 + cache_steps + store_step + store_row + (N,T,HV,K,V,hratio)
# + 5 stride (q,k,v,**a**,b) + 3 float + 3 int + stream
_ARGS = (
    [ctypes.c_void_p] * 13
    + [ctypes.c_int]
    + [ctypes.c_void_p]
    + [ctypes.c_void_p]
    + [ctypes.c_int] * 11
    + [ctypes.c_float] * 3
    + [ctypes.c_int] * 3
    + [ctypes.c_void_p]
)

# 全量快照入口 `gdn_recurrent_update_nofma`：同一个内核、同一份输入，只是
# BPLUS=false —— 每步都落一份快照。B+ 的「纯写入省略」就该和它逐位相同，
# 所以对拍必须用它当参考臂，而不是 aiter（aiter 与本核本就差几个 ULP）。
_ARGS_FULL = (
    [ctypes.c_void_p] * 13
    + [ctypes.c_int] * 7
    + [ctypes.c_int] * 5
    + [ctypes.c_float] * 3
    + [ctypes.c_int] * 3
    + [ctypes.c_void_p]
)

_LIB = None
_ERR = None


def lib():
    global _LIB, _ERR
    if _LIB is not None:
        return _LIB
    if _ERR is not None:
        raise RuntimeError(_ERR)
    path = os.environ.get("GDN_HIP_SO", _DEFAULT_SO)
    try:
        h = ctypes.CDLL(path)
        for name in ("gdn_recurrent_update_bplus",
                     "gdn_recurrent_update_bplus_state"):
            h[name].argtypes = _ARGS
            h[name].restype = None
        h.gdn_recurrent_update_nofma.argtypes = _ARGS_FULL
        h.gdn_recurrent_update_nofma.restype = None
        _LIB = h
    except Exception as e:  # noqa: BLE001 - 报清楚为什么没起来
        _ERR = f"[gdn-hip-bplus] 加载 {path} 失败: {e}"
        raise RuntimeError(_ERR) from e
    return _LIB


def enabled() -> bool:
    return os.environ.get("SGLANG_GDN_HIP_BPLUS") == "1"


def enabled_kernel() -> bool:
    return os.environ.get("SGLANG_GDN_HIP_KERNEL") == "1"


def _tok_stride(x: torch.Tensor) -> int:
    """Stride between tokens: q/k/v are (1, seq, H, D), a/b are (seq, HV)."""
    return int(x.stride(1)) if x.dim() >= 3 else int(x.stride(0))


# The kernel takes A_log/dt_bias as const float*, but under bf16 weights dt_bias
# is bf16 [HV]: reading it as float* runs off the buffer and every head's decay
# is wrong. Cast once and cache on the tensor; verify runs in a captured graph.
def _f32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float32:
        return x
    hit = getattr(x, "_gdn_f32", None)
    if hit is None or hit[1] != x._version:
        hit = (x.detach().float().contiguous(), x._version)
        try:
            x._gdn_f32 = hit
        except Exception:  # noqa: BLE001 - 不允许挂属性就每次现算
            pass
    return hit[0]


def _call(fn, q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source, h0_indices,
          o, snap_buf, snap_idx, cache_steps, store_step, store_row, n_seq,
          tok_per_seq, hv, h, kdim, vdim, use_h0=None):
    # use_initial_state and varlen are independent: the compensate replay has no
    # cu_seqlens (fixed D steps) but must still read h0.
    if use_h0 is None:
        use_h0 = True
    lib()[fn](
        q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(),
        _f32(A_log).data_ptr(), _f32(dt_bias).data_ptr(),
        0 if cu_seqlens is None else cu_seqlens.data_ptr(),
        h0_source.data_ptr(), h0_indices.data_ptr(), o.data_ptr(),
        0 if snap_buf is None else snap_buf.data_ptr(),
        0 if snap_idx is None else snap_idx.data_ptr(),
        int(cache_steps),
        0 if store_step is None else store_step.data_ptr(),
        0 if store_row is None else store_row.data_ptr(),
        int(n_seq), int(tok_per_seq), int(hv), int(kdim), int(vdim),
        int(hv // h),
        # a and b carry separate strides: b is often a slice of a wider buffer
        # (96 vs a's 48) and a shared one makes a read another token's row.
        _tok_stride(q), _tok_stride(k), _tok_stride(v),
        _tok_stride(a), _tok_stride(b),
        1.0 / (kdim ** 0.5), 1.0, 20.0,
        # (use_initial_state, varlen, disable_state_update)
        1 if use_h0 else 0,
        # varlen must be off without cu_seqlens: the kernel would read through
        # the null pointer.
        0 if cu_seqlens is None else 1,
        1,
        torch.cuda.current_stream().cuda_stream,
    )


def verify(q, k, v, a, b, A_log, dt_bias, cu_seqlens, ssm_states, cache_indices,
           snap_flat, snap_idx, cache_steps, draft_token_num, o, prev_step=None):
    """Write the full-run snapshot to row (prev_step+1) % cache_steps.

    Never row draft_token_num-1: that is the row the commit points at after a
    full accept, and the next verify reads it as h0. Each slot has one live row,
    so the rotated one is dead and the write touches nothing being read.
    """
    n = int(snap_idx.shape[0])
    store = torch.full((n,), draft_token_num - 1, dtype=torch.int32,
                       device=snap_idx.device)
    # Slot ids come from snap_idx: cache_indices holds flat h0 rows here
    # (slot*D+prev) and indexing prev_step with them reads out of bounds, which
    # a captured graph turns into a VMFault.
    if prev_step is None:
        # No state handoff: write the full-accept row (draft_token_num - 1).
        store_row = store
    else:
        safe = snap_idx.to(torch.int64).clamp(0, int(prev_step.numel()) - 1)
        prev = prev_step.index_select(0, safe).to(torch.int32)
        store_row = torch.remainder(prev + 1, int(cache_steps)).to(torch.int32)
    _call("gdn_recurrent_update_bplus", q, k, v, a, b, A_log, dt_bias,
          cu_seqlens, ssm_states, cache_indices, o, snap_flat, snap_idx,
          cache_steps, store, store_row, n, draft_token_num, v.shape[-2],
          q.shape[-2], k.shape[-1], v.shape[-1])


def verify_full(q, k, v, a, b, A_log, dt_bias, cu_seqlens, ssm_states,
                cache_indices, snap_flat, snap_idx, cache_steps,
                draft_token_num, o):
    """Full-snapshot entry, for the A/B only: same kernel and inputs, every step
    written, so the row B+ writes must be torch.equal to it."""
    n = int(snap_idx.shape[0])
    lib()["gdn_recurrent_update_nofma"](
        q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(),
        _f32(A_log).data_ptr(), _f32(dt_bias).data_ptr(),
        0 if cu_seqlens is None else cu_seqlens.data_ptr(),
        ssm_states.data_ptr(), cache_indices.data_ptr(), o.data_ptr(),
        snap_flat.data_ptr(), snap_idx.data_ptr(),
        int(cache_steps), int(n), int(draft_token_num), int(v.shape[-2]),
        int(k.shape[-1]), int(v.shape[-1]), int(v.shape[-2] // q.shape[-2]),
        _tok_stride(q), _tok_stride(k), _tok_stride(v),
        _tok_stride(a), _tok_stride(b),
        1.0 / (k.shape[-1] ** 0.5), 1.0, 20.0,
        1, 0 if cu_seqlens is None else 1, 1,
        torch.cuda.current_stream().cuda_stream,
    )


def compensate(q, k, v, a, b, A_log, dt_bias, ssm_states, h0_rows,
               snap_flat, snap_idx, cache_steps, store_step, n_seq,
               tok_per_seq, o):
    """Replay state only (no o); slots with h0_rows < 0 exit immediately."""
    _call("gdn_recurrent_update_bplus_state", q, k, v, a, b, A_log, dt_bias,
          None, ssm_states, h0_rows, o, snap_flat, snap_idx, cache_steps,
          store_step, None, n_seq, tok_per_seq, v.shape[-2], q.shape[-2],
          k.shape[-1], v.shape[-1], use_h0=True)
