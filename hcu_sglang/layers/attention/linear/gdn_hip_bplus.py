"""手写 HIP GDN 递推核 + B+（只写 committed step 的快照）。

两件事分开看：

* **B+ 是纯写入省略**：状态链 h0→h1→…→hT、每一步的归约顺序、accept 判定全不变，
  只是不再把 T 份中间态都落盘。verify 只写最后一行（全接受时它就是提交态），
  被截断的槽在 commit 时用同一份输入、同一个 h0、同一条 T 步链重跑一次，
  只写被接受的那一步。内核级已证：写出的行与全量快照跑的对应行 ``torch.equal``。

* **HIP 核不是逐位等价**：它把三处 K=128 的 fp32 归约（l2norm / delta / o）
  重排成按 G=8 分组的 shuffle 树。实测 o 99.986% 逐位相同、最大 6 ULP（c9）
  / 64 ULP（c36），且逐位率**不随序列长度退化**（T=1/2/8/32/64 全 99.98%）
  ⇒ 不是累积漂移。臂内完全确定（无原子、无竞争），同输入跑两次结果相同。

所以它换来的是「与 aiter 臂不逐位相同」，不是「不可复现」。
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
            f = getattr(h, name)
            f.argtypes = _ARGS
            f.restype = None
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
    """每 token 的跨度（内核里 `x + t*stride` 的 stride）。

    产线的 q/k/v 是 (1, seq, H, D)、a/b 是 (seq, HV)，所以不能写死
    ``stride(-2)``：4D 张量的 ``stride(-2)`` 是 head 的步长 K，不是 token 步长。
    """
    return int(x.stride(1)) if x.dim() >= 3 else int(x.stride(0))


# 核里 ``A_log``/``dt_bias`` 都是硬编码的 ``const float*``。bf16 权重下
# ``layer.dt_bias`` 是 **bf16 [HV]**（实测 48 个元素），按 float* 读就是
# 48×4=192 字节读一个 96 字节的 buffer：前一半 head 读到相邻内存、后一半
# 直接越界，decay = exp(-exp(A_log)*softplus(a+dt)) 每个 head 全错，状态链
# 从第一步就崩。aiter 的 Triton 核按张量 dtype 取指针，所以只有我们中招。
# 权重是常量，cast 一次挂在张量上，热路径不再重复（避免每次 verify 多一个
# kernel launch，也避免在 CUDA graph capture 里新分配）。
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
    # ``use_initial_state`` 和 ``varlen`` 是两个独立开关：纠偏重放没有 cu_seqlens
    # （varlen 关、按 D 定长跑）但**必须**读 h0。默认两者都开，与改动前一致。
    if use_h0 is None:
        use_h0 = True
    getattr(lib(), fn)(
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
        # a 和 b 各走各的行距：b 常是更宽 buffer 的切片（实测 96 vs a 的 48），
        # 共用一个 stride_b 会让 a 读到别的 token、越界后读到 NaN。
        _tok_stride(q), _tok_stride(k), _tok_stride(v),
        _tok_stride(a), _tok_stride(b),
        1.0 / (kdim ** 0.5), 1.0, 20.0,
        # (use_initial_state, varlen, disable_state_update) —— 只有这三个 int。
        # h0 读不读由第一个决定，和 varlen 无关；纠偏两样都是「读 h0、不定长」。
        1 if use_h0 else 0,
        # cu_seqlens 为空时必须把 varlen 关掉：否则内核去读 cu_seqlens[i_n]，
        # 传进去的是 0 指针，直接段错误。
        0 if cu_seqlens is None else 1,
        1,
        torch.cuda.current_stream().cuda_stream,
    )


def verify(q, k, v, a, b, A_log, dt_bias, cu_seqlens, ssm_states, cache_indices,
           snap_flat, snap_idx, cache_steps, draft_token_num, o, prev_step):
    """verify：全跑完的那一份快照写到第 ``(prev_step+1) % cache_steps`` 行。

    不能写死第 ``draft_token_num - 1`` 行：全接受之后提交指针就是它，下一拍
    verify 的 h0 正读这一行。若那一拍又被截断，纠偏要重放同一个 h0，而 verify
    已经把它覆盖成本拍全跑完的状态 —— 纠偏从错的 h0 出发（accept 看不出来）。
    每槽只有 ``prev_step`` 那一行是活的，``(prev+1) % D`` 必然是死行，且不等于
    h0 行，所以这一次写入不动任何还在被读的数据；提交侧在全接受时把 prev_step
    指到这一行（见 ``gdn_backend.gdn_hip_compensate``）。
    """
    n = int(snap_idx.shape[0])
    store = torch.full((n,), draft_token_num - 1, dtype=torch.int32,
                       device=snap_idx.device)
    # 槽号走 snap_idx：`cache_indices` 在这个调用里是 h0 的扁平行号
    # （slot*D+prev，最大 ~slot_cnt*D），拿它索引 prev_step 会越界读；
    # 而且 CUDA graph 里没有边界检查，越界就是 VMFault。上界也 clamp 一次，
    # 这条路径宁可读成 0 也绝不越界。
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
    """**只用于对拍**：同一个内核、同一份输入，走全量快照（每步都写）。

    B+ 相对它只该少写 T-1 份快照，写出来的那一行必须 ``torch.equal``。
    行号就是「第几步」，所以最后一步在第 ``draft_token_num-1`` 行。
    """
    n = int(snap_idx.shape[0])
    getattr(lib(), "gdn_recurrent_update_nofma")(
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
    """纠偏：只推 state，不算 o；h0_rows < 0 的槽整块退出。"""
    _call("gdn_recurrent_update_bplus_state", q, k, v, a, b, A_log, dt_bias,
          None, ssm_states, h0_rows, o, snap_flat, snap_idx, cache_steps,
          store_step, None, n_seq, tok_per_seq, v.shape[-2], q.shape[-2],
          k.shape[-1], v.shape[-1], use_h0=True)
