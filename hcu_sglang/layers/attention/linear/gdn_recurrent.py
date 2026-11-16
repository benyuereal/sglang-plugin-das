"""HIP GDN recurrent kernel with B+ (write only the committed step).

B+ omits writes, not math: the chain, its reduction order and the accept rule
are unchanged, and the row it does write matches the full-snapshot kernel bit
for bit. The three K=128 fp32 reductions (l2norm, delta, o) and the 1/sqrt
norm reciprocal reproduce the production Triton kernel
(sglang.kernels.ops.attention.fla), not aiter's tl.rsqrt variant.
"""
import ctypes
import os

import torch

# The B+ entries (single-layer and the one-launch multi) live in this build;
# GDN_HIP_SO overrides it when a deployment ships the object elsewhere.
_DEFAULT_SO = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "csrc", "gdn", "build", "gdn_recurrent.so",
    )
)
_BV = 32  # 与 gdn_recurrent.hip 的 BV 一致
# softplus(beta * x + threshold) literals the recurrent kernel expects; they
# must match gdn_recurrent.hip.
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0

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
# 所以对拍必须用它当参考臂；两者的算术都已对齐产线 Triton 核。
_ARGS_FULL = (
    [ctypes.c_void_p] * 13
    + [ctypes.c_int] * 7
    + [ctypes.c_int] * 5
    + [ctypes.c_float] * 3
    + [ctypes.c_int] * 3
    + [ctypes.c_void_p]
)

# Multi-layer compensate: the per-layer pointers/strides arrive as an int64
# meta table, so one launch covers every GDN layer (blockIdx.z picks a row).
_ARGS_MULTI = (
    [ctypes.c_void_p] * 3
    + [ctypes.c_int]
    + [ctypes.c_void_p]
    + [ctypes.c_int] * 7
    + [ctypes.c_float] * 3
    + [ctypes.c_int] * 3
    + [ctypes.c_void_p]
)

# norm pass：q/k/eg/beta 一次算完（每层一次 launch）。
_ARGS_NORM = (
    [ctypes.c_void_p] * 10
    + [ctypes.c_int] * 9
    + [ctypes.c_float] * 3
    + [ctypes.c_void_p]
)
# PRENORM verify：在 `..._bplus` 的参数之后追加 qn/kn/eg/beta + stride + gate。
_ARGS_PRENORM = (
    [ctypes.c_void_p] * 13
    + [ctypes.c_int]
    + [ctypes.c_void_p] * 2
    + [ctypes.c_int] * 11
    + [ctypes.c_float] * 3
    + [ctypes.c_int] * 3
    + [ctypes.c_void_p] * 4
    + [ctypes.c_int] * 3
    + [ctypes.c_void_p]
)

# causal_conv1d update (target-verify, one launch per GDN layer): 11 pointers +
# (batch,dim,seqlen,state_len,num_cache_lines) + x/w strides + conv_state
# strides + window strides + retrieve strides + parent strides + o strides +
# pad_slot_id + 6 flags + idx64 + stream.
_ARGS_CONV = (
    [ctypes.c_void_p] * 11
    + [ctypes.c_int] * 5
    + [ctypes.c_int] * 5
    + [ctypes.c_int] * 4
    + [ctypes.c_int] * 5
    + [ctypes.c_int] * 4
    + [ctypes.c_int] * 2
    + [ctypes.c_int] * 3
    + [ctypes.c_int]
    + [ctypes.c_int] * 6
    + [ctypes.c_int]
    + [ctypes.c_void_p]
)

_LIB = None
_ERR = None
_PRENORM_BUF = {}
_CONV_LAST_REASON = None


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
            fn = getattr(h, name)
            fn.argtypes = _ARGS
            fn.restype = None
        h.gdn_recurrent_update_nofma.argtypes = _ARGS_FULL
        h.gdn_recurrent_update_nofma.restype = None
        multi = getattr(h, "gdn_recurrent_update_bplus_state_multi", None)
        if multi is not None:
            multi.argtypes = _ARGS_MULTI
            multi.restype = None
        norm = getattr(h, "gdn_verify_norm_pass", None)
        if norm is not None:
            norm.argtypes = _ARGS_NORM
            norm.restype = None
        pre = getattr(h, "gdn_recurrent_update_bplus_prenorm", None)
        if pre is not None:
            pre.argtypes = _ARGS_PRENORM
            pre.restype = None
        conv = getattr(h, "gdn_causal_conv1d_update", None)
        if conv is not None:
            conv.argtypes = _ARGS_CONV
            conv.restype = None
        _LIB = h
    except Exception as e:  # noqa: BLE001 - 报清楚为什么没起来
        _ERR = f"[gdn-hip-bplus] 加载 {path} 失败: {e}"
        raise RuntimeError(_ERR) from e
    return _LIB


def enabled() -> bool:
    return os.environ.get("SGLANG_GDN_HIP_BPLUS") == "1"


def enabled_kernel() -> bool:
    return os.environ.get("SGLANG_GDN_HIP_KERNEL") == "1"


def prenorm_enabled() -> bool:
    """Hoist q/k l2norm + gating out of the 12x-redundant verify CTAs.

    Needs the norm-pass entry; a .so built before that entry silently keeps the
    old path instead of failing a serving start.
    """
    if os.environ.get("SGLANG_GDN_HIP_PRENORM") != "1":
        return False
    return getattr(lib(), "gdn_verify_norm_pass", None) is not None


def prenorm_gate_enabled() -> bool:
    """Also hoist softplus/sigmoid gating to per-(token,hv) eg/beta buffers."""
    return os.environ.get("SGLANG_GDN_HIP_PRENORM_GATE", "1") == "1"


def prenorm_buffers(rows: int, h: int, kdim: int, hv: int, device):
    """Persistent scratch, one entry per size.

    A captured graph bakes these addresses, so a later eager batch must never
    replace an existing entry; ``rows`` is the pool's slot*D budget and stays
    constant for the life of the process.  The buffer is consumed by the norm
    pass of the very next verify launch on the same stream, so one shared
    scratch serves every GDN layer.
    """
    key = (int(rows), int(h), int(kdim), int(hv), str(device))
    ent = _PRENORM_BUF.get(key)
    if ent is None:
        qn = torch.empty((rows, h, kdim), dtype=torch.float32, device=device)
        kn = torch.empty_like(qn)
        eg = torch.empty((rows, hv), dtype=torch.float32, device=device)
        beta = torch.empty_like(eg)
        ent = (qn, kn, eg, beta)
        _PRENORM_BUF[key] = ent
    return ent


def _norm_pass(q, k, a, b, A_log, dt_bias, qn, kn, eg, beta, hv):
    h = int(q.shape[-2])
    kdim = int(q.shape[-1])
    getattr(lib(), "gdn_verify_norm_pass")(
        q.data_ptr(), k.data_ptr(), a.data_ptr(), b.data_ptr(),
        _f32(A_log).data_ptr(), _f32(dt_bias).data_ptr(),
        qn.data_ptr(), kn.data_ptr(), eg.data_ptr(), beta.data_ptr(),
        int(q.shape[1]) * h, h, int(hv), kdim, int(hv) // h,
        _tok_stride(q), _tok_stride(k), _tok_stride(a), _tok_stride(b),
        _SOFTPLUS_BETA, _SOFTPLUS_THRESHOLD, 1.0 / (kdim ** 0.5),
        torch.cuda.current_stream().cuda_stream,
    )


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


_NULL_PTR = 0  # absent optional tensors are passed as a null pointer


def _ptr(tensor):
    """Data pointer, or the null pointer the kernels check when it is absent."""
    return _NULL_PTR if tensor is None else tensor.data_ptr()


def _call(fn, q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source, h0_indices,
          o, snap_buf, snap_idx, cache_steps, store_step, store_row, n_seq,
          tok_per_seq, hv, h, kdim, vdim, use_h0=None):
    # use_initial_state and varlen are independent: the compensate replay has no
    # cu_seqlens (fixed D steps) but must still read h0.
    if use_h0 is None:
        use_h0 = True
    getattr(lib(), fn)(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(),
        _f32(A_log).data_ptr(), _f32(dt_bias).data_ptr(),
        _ptr(cu_seqlens),
        h0_source.data_ptr(), h0_indices.data_ptr(), o.data_ptr(),
        _ptr(snap_buf),
        _ptr(snap_idx),
        int(cache_steps),
        _ptr(store_step),
        _ptr(store_row),
        int(n_seq), int(tok_per_seq), int(hv), int(kdim), int(vdim),
        int(hv // h),
        # a and b carry separate strides: b is often a slice of a wider buffer
        # (96 vs a's 48) and a shared one makes a read another token's row.
        _tok_stride(q), _tok_stride(k), _tok_stride(v),
        _tok_stride(a), _tok_stride(b),
        1.0 / (kdim ** 0.5), _SOFTPLUS_BETA, _SOFTPLUS_THRESHOLD,
        # (use_initial_state, varlen, disable_state_update)
        1 if use_h0 else 0,
        # varlen must be off without cu_seqlens: the kernel would read through
        # the null pointer.
        0 if cu_seqlens is None else 1,
        1,
        torch.cuda.current_stream().cuda_stream,
    )


def verify(q, k, v, a, b, A_log, dt_bias, cu_seqlens, ssm_states, cache_indices,
           snap_flat, snap_idx, cache_steps, draft_token_num, o, prev_step=None,
           store=None, store_row=None):
    """Write the full-run snapshot to row (prev_step+1) % cache_steps.

    Never row draft_token_num-1: that is the row the commit points at after a
    full accept, and the next verify reads it as h0. Each slot has one live row,
    so the rotated one is dead and the write touches nothing being read.
    """
    # store/store_row are per-batch, not per-layer: the handoff caller computes
    # them once per step and passes them in; the fallback keeps the old path.
    n = int(snap_idx.shape[0])
    if store is None:
        store = torch.full((n,), draft_token_num - 1, dtype=torch.int32,
                           device=snap_idx.device)
    # Slot ids come from snap_idx: cache_indices holds flat h0 rows here
    # (slot*D+prev) and indexing prev_step with them reads out of bounds, which
    # a captured graph turns into a VMFault.
    if store_row is None:
        if prev_step is None:
            # No state handoff: write the full-accept row (draft_token_num - 1).
            store_row = store
        else:
            safe = snap_idx.to(torch.int64).clamp(0, int(prev_step.numel()) - 1)
            prev = prev_step.index_select(0, safe).to(torch.int32)
            store_row = torch.remainder(prev + 1, int(cache_steps)).to(torch.int32)
    prenorm = prenorm_enabled() and snap_flat is not None
    if prenorm:
        hv = int(v.shape[-2])
        h = int(q.shape[-2])
        kdim = int(q.shape[-1])
        rows = max(int(snap_flat.shape[0]), int(q.shape[1]))
        qn, kn, eg, beta = prenorm_buffers(rows, h, kdim, hv, q.device)
        _norm_pass(q, k, a, b, A_log, dt_bias, qn, kn, eg, beta, hv)
        getattr(lib(), "gdn_recurrent_update_bplus_prenorm")(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(),
            b.data_ptr(), _f32(A_log).data_ptr(), _f32(dt_bias).data_ptr(),
            _ptr(cu_seqlens),
            ssm_states.data_ptr(), cache_indices.data_ptr(), o.data_ptr(),
            _ptr(snap_flat),
            _ptr(snap_idx),
            int(cache_steps),
            _ptr(store),
            _ptr(store_row),
            n, int(draft_token_num), hv, kdim, int(v.shape[-1]),
            int(hv // h),
            _tok_stride(q), _tok_stride(k), _tok_stride(v),
            _tok_stride(a), _tok_stride(b),
            1.0 / (kdim ** 0.5), _SOFTPLUS_BETA, _SOFTPLUS_THRESHOLD,
            1, 0 if cu_seqlens is None else 1, 1,
            qn.data_ptr(), kn.data_ptr(), eg.data_ptr(), beta.data_ptr(),
            h * kdim, hv, 1 if prenorm_gate_enabled() else 0,
            torch.cuda.current_stream().cuda_stream,
        )
        return
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
    getattr(lib(), "gdn_recurrent_update_nofma")(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), a.data_ptr(), b.data_ptr(),
        _f32(A_log).data_ptr(), _f32(dt_bias).data_ptr(),
        _ptr(cu_seqlens),
        ssm_states.data_ptr(), cache_indices.data_ptr(), o.data_ptr(),
        snap_flat.data_ptr(), snap_idx.data_ptr(),
        int(cache_steps), int(n), int(draft_token_num), int(v.shape[-2]),
        int(k.shape[-1]), int(v.shape[-1]), int(v.shape[-2] // q.shape[-2]),
        _tok_stride(q), _tok_stride(k), _tok_stride(v),
        _tok_stride(a), _tok_stride(b),
        1.0 / (k.shape[-1] ** 0.5), _SOFTPLUS_BETA, _SOFTPLUS_THRESHOLD,
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


def f32(x: torch.Tensor) -> torch.Tensor:
    """Cached fp32 view of A_log/dt_bias for the pointer table."""
    return _f32(x)


def has_multi() -> bool:
    return getattr(lib(), "gdn_recurrent_update_bplus_state_multi", None) is not None


def compensate_multi(layer_ptrs, h0_indices, snap_idx, cache_steps, store_step,
                     n_seq, tok_per_seq, hv, h, kdim, vdim):
    """One-launch state replay for every layer in ``layer_ptrs``."""
    getattr(lib(), "gdn_recurrent_update_bplus_state_multi")(
        layer_ptrs.data_ptr(),
        h0_indices.data_ptr(),
        snap_idx.data_ptr(),
        int(cache_steps),
        _ptr(store_step),
        int(layer_ptrs.shape[0]),
        int(n_seq), int(tok_per_seq), int(hv), int(kdim), int(vdim),
        int(hv // h),
        1.0 / (kdim ** 0.5), _SOFTPLUS_BETA, _SOFTPLUS_THRESHOLD,
        1, 0, 1,
        torch.cuda.current_stream().cuda_stream,
    )


# ---------------------------------------------------------------------------
# causal_conv1d update (target verify): same tensor interface as the Triton
# `causal_conv1d_update`. Returns None when this call falls outside what the
# HIP kernel implements, and the caller keeps the Triton kernel.
# ---------------------------------------------------------------------------

_CONV_MAX_T = 16


def conv_last_reason():
    """Why the last conv_update() call stayed on Triton (None if it ran)."""
    return _CONV_LAST_REASON


def conv_enabled() -> bool:
    """SGLANG_GDN_HIP_CONV=1 and the loaded .so exports the conv entry."""
    if os.environ.get("SGLANG_GDN_HIP_CONV") != "1":
        return False
    return getattr(lib(), "gdn_causal_conv1d_update", None) is not None


def conv_unsupported_reason(x, conv_state, weight, bias, activation,
                            cache_seqlens, conv_state_indices,
                            num_accept_tokens, intermediate_conv_window,
                            intermediate_state_indices, retrieve_next_token,
                            retrieve_next_sibling, retrieve_parent_token,
                            pad_slot_id):
    """Short reason this call must stay on Triton; None when HIP can run it."""
    if x.dim() not in (2, 3):
        return f"x.dim={x.dim()}"
    if weight.dim() != 2 or int(weight.size(1)) != 4:
        return f"weight={tuple(weight.shape)}"
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return f"dtype x={x.dtype} w={weight.dtype}"
    if conv_state.dtype != torch.bfloat16:
        return f"conv_state.dtype={conv_state.dtype}"
    if bias is not None and (bias.dtype != torch.bfloat16 or not bias.is_contiguous()):
        return f"bias={bias.dtype}/contig={bias.is_contiguous()}"
    if cache_seqlens is not None or num_accept_tokens is not None:
        return "circular / multi-accept conv_state is not implemented"
    if conv_state.dim() != 3:
        return f"conv_state.dim={conv_state.dim()}"
    # 2D x is (batch, dim); the wrapper unsqueezes it to (batch, dim, 1).
    seqlen = int(x.size(-1)) if x.dim() == 3 else 1
    if seqlen < 1 or seqlen > _CONV_MAX_T:
        return f"seqlen={seqlen}"
    dim = int(x.size(1))
    if dim != int(weight.size(0)) or dim != int(conv_state.size(1)):
        return (f"dim mismatch x={dim} w={weight.size(0)} "
                f"state={conv_state.size(1)}")
    if int(conv_state.size(2)) < 3:
        return f"state_len={conv_state.size(2)}"
    if conv_state_indices is not None and (
        conv_state_indices.dtype != torch.int32 or conv_state_indices.dim() != 1
    ):
        return f"conv_state_indices={conv_state_indices.dtype}"
    if intermediate_conv_window is not None:
        win = intermediate_conv_window
        if win.dtype != torch.bfloat16 or win.dim() != 4:
            return f"window={win.dtype}/{win.dim()}d"
        # A dummy/warmup call can carry more window steps than tokens; only
        # the first `seqlen` rows are touched.
        if int(win.size(1)) < seqlen or (int(win.size(2)), int(win.size(3))) != (
            dim,
            3,
        ):
            return f"window shape={tuple(win.shape)}"
        if intermediate_state_indices is None or (
            intermediate_state_indices.dtype != torch.int32
        ):
            return "window needs int32 intermediate_state_indices"
    if retrieve_next_token is None:
        if retrieve_next_sibling is not None or retrieve_parent_token is not None:
            return "sibling/parent passed without retrieve_next_token"
    else:
        if retrieve_next_sibling is None or retrieve_parent_token is None:
            return "tree arrays incomplete"
        if retrieve_next_token.dtype not in (torch.int32, torch.int64):
            return f"tree index dtype={retrieve_next_token.dtype}"
        for other in (retrieve_next_sibling, retrieve_parent_token):
            if other.dtype != retrieve_next_token.dtype or tuple(
                other.shape
            ) != tuple(retrieve_next_token.shape):
                return "tree arrays must share dtype and shape"
        if int(retrieve_next_token.size(1)) < seqlen:
            return f"tree width={retrieve_next_token.size(1)} < seqlen={seqlen}"
    if pad_slot_id is not None and not isinstance(pad_slot_id, int):
        return f"pad_slot_id={pad_slot_id!r}"
    return None


def conv_update(x, conv_state, weight, bias=None, activation=None,
                cache_seqlens=None, conv_state_indices=None,
                num_accept_tokens=None, intermediate_conv_window=None,
                intermediate_state_indices=None, retrieve_next_token=None,
                retrieve_next_sibling=None, retrieve_parent_token=None,
                pad_slot_id=-1):
    """Launch the HIP conv update; returns None when the caller must use Triton."""
    global _CONV_LAST_REASON
    if not conv_enabled():
        _CONV_LAST_REASON = "SGLANG_GDN_HIP_CONV=1 and the .so exports no conv entry"
        return None
    if conv_unsupported_reason(
        x, conv_state, weight, bias, activation, cache_seqlens,
        conv_state_indices, num_accept_tokens, intermediate_conv_window,
        intermediate_state_indices, retrieve_next_token,
        retrieve_next_sibling, retrieve_parent_token, pad_slot_id,
    ) is not None:
        _CONV_LAST_REASON = conv_unsupported_reason(
            x, conv_state, weight, bias, activation, cache_seqlens,
            conv_state_indices, num_accept_tokens, intermediate_conv_window,
            intermediate_state_indices, retrieve_next_token,
            retrieve_next_sibling, retrieve_parent_token, pad_slot_id,
        )
        return None

    unsq = x.dim() == 2
    x3 = x.unsqueeze(-1) if unsq else x
    batch, dim, seqlen = (int(v) for v in x3.shape)
    out = torch.empty_like(x3)
    tree = retrieve_next_token is not None
    save_inter = intermediate_conv_window is not None
    pad = 0 if pad_slot_id is None else int(pad_slot_id)

    def ptr(t):
        return _ptr(t)

    def stride2(t):
        return (0, 0) if t is None else (int(t.stride(0)), int(t.stride(1)))

    win = intermediate_conv_window
    win_strides = (0, 0, 0, 0) if win is None else tuple(int(s) for s in win.stride())
    nxt_strides = stride2(retrieve_next_token)
    sib_strides = stride2(retrieve_next_sibling)
    par_strides = stride2(retrieve_parent_token)
    idx64 = 1 if not tree or retrieve_next_token.dtype == torch.int64 else 0
    s_ii = 0 if intermediate_state_indices is None else int(
        intermediate_state_indices.stride(0)
    )

    getattr(lib(), "gdn_causal_conv1d_update")(
        x3.data_ptr(), weight.data_ptr(), ptr(bias), conv_state.data_ptr(),
        ptr(conv_state_indices), ptr(win), ptr(intermediate_state_indices),
        ptr(retrieve_next_token), ptr(retrieve_next_sibling),
        ptr(retrieve_parent_token), out.data_ptr(),
        batch, dim, seqlen, int(conv_state.size(2)), int(conv_state.size(0)),
        *[int(s) for s in x3.stride()],
        *[int(s) for s in weight.stride()],
        *[int(s) for s in conv_state.stride()],
        0 if conv_state_indices is None else int(conv_state_indices.stride(0)),
        *win_strides, s_ii,
        *nxt_strides, *sib_strides, *par_strides,
        *[int(s) for s in out.stride()],
        pad,
        1 if bias is not None else 0,
        1 if activation in (True, "silu", "swish") else 0,
        1 if conv_state_indices is not None else 0,
        1 if save_inter else 0,
        1 if tree else 0,
        1 if pad_slot_id is not None else 0,
        idx64,
        torch.cuda.current_stream().cuda_stream,
    )
    return out.squeeze(-1) if unsq else out
