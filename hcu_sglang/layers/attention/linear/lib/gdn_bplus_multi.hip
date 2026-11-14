// GDN 递推步的手写核（实验版，不进货线，先在 experiments/ 里对拍）。
//
// 目标：替换 aiter 的 `fused_sigmoid_gating_delta_rule_update_kernel` 里
// 那个占 20.8% 的 per-token 循环。
//
// 布局是本文件的全部要点。wave64 处理 (slot, hv, 32 个 v)：
//   G = 8   每个 k 组占 8 个 lane
//   KL = 16 每 lane 持 16 个 k
//   VL = 4  每 lane 持 4 个 v
//   lane l: g = l & 7 -> k ∈ [16g, 16g+16)；t1 = l >> 3 -> v ∈ {vb+4*t1+r}
//   状态寄存器 = KL*VL = 64 个 f32（与 Triton 的 64 同量级，125 VGPR / 0 溢出）。
//
// 为什么是 G=8：寄存器数恒等于 KL*VL = (128/G)*(32G/64) = 64，与 G 无关；
// 但每次归约的 shuffle 数 = (v/lane)*log2(G) = (G/2)*log2(G)：
//   G=32 -> 80，G=16 -> 32，G=8 -> 12，G=4 -> 4。
// 原来照 Triton 用 G=32，每 token 170 条 ds_bpermute 串在依赖链上，实测
// 每多一个 token 只加 ~70us（状态搬运量不变），是延迟而不是带宽。
//
// 试过但不可用：DPP 内联汇编。`v_add_f32_dpp` 带 bank_mask 的「shr 写高 bank +
// shl 写低 bank」形式**孤立测试 64/64 正确**，但接在任何一条 DPP 之后结果就错
// （只有 bank2 对），`"+v"` 的读改写形式更是全 0。这套工具链上不可靠，故用
// `__shfl_xor`（宽 8，组内连续 8 lane）。
//
// 构建：scripts/build.sh 同一套编译器与 flag，
//   experiments/gdn/build_gdn.sh -> build/gdn_exp.so

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#define BK 128
#define BV 32
// G 是唯一自由参数：一个 k 组占 G 个 lane。
// 其余全部导出，寄存器数恒为 KL*VL = (BK/G)*(BV*G/64) = BK*BV/64 = 64。
#define G 8
#define KL (BK / G)          // 每 lane 的 k 个数
#define VL (BV * G / 64)     // 每 lane 的 v 个数
#define NT1 (64 / G)         // 每 block 的 v 组数

// VL 个 bf16 的打包搬运类型（2 个 -> 4 B，4 个 -> 8 B）
template <int N> struct vpack_of;
template <> struct vpack_of<2> { using type = uint32_t; };
template <> struct vpack_of<4> { using type = uint2; };
template <> struct vpack_of<8> { using type = uint4; };
typedef vpack_of<VL>::type vpack_t;
// 打包类型必须正好覆盖 VL 个 bf16，否则只会搬一半（G=32/VL=16 踩过）。
static_assert(sizeof(vpack_t) == VL * sizeof(hip_bfloat16),
              "vpack_of<VL> 的字节数必须等于 VL*2");

// 组内 G lane 的归约（lane 连续 G 个为一组）
__device__ __forceinline__ float grp_sum(float p) {
#pragma unroll
  for (int m = G / 2; m >= 1; m >>= 1) p = __fadd_rn(p, __shfl_xor(p, m, G));
  return p;
}

// BPLUS：只写 store_step[i_n] 那一份快照（-1 = 该槽不写），并让无效槽（h0_idx<0）
//        立即退出，使纠偏那一趟的成本正比于真正要补的槽数。默认 false，
//        原有两个导出符号的行为逐字节不变。
//
//        写哪一行由 store_row[i_n] 决定（空 = 写 store_step[i_n] 那一行）。
//        verify 必须传 (prev_step+1)%D，不能写死第 T-1 行：
//        全接受之后提交指针 prev_step 就是 T-1，下一拍 verify 的 h0 正读第 T-1 行；
//        若这一拍又被截断，纠偏要重放同一行 h0，而 verify 已经把它覆盖成本拍
//        全跑完的状态 → 纠偏从错的 h0 出发，写出垃圾提交态（accept 看不出来）。
//        每槽只有 prev_step 那一行是活的，其余行都是死行，所以 (prev+1)%D 一定安全，
//        且和 store_step 那一行不冲突（store_row 只在全接受时被提交指针采纳）。
// STATE_ONLY：纠偏用。跳过 q 的 l2norm 与 o = h·q 那段，只推 state。
// MULTI: blockIdx.z selects a layer from a meta table and rebinds the
// per-layer pointers below. One launch then covers all GDN layers; the table
// row is 10 pointers + 5 token strides, all int64 (see gdn_hip_bplus.py).
#define GDN_MULTI_COLS 15

template <bool FMA_OUTER, bool BPLUS = false, bool STATE_ONLY = false,
          bool MULTI = false>
// 64 线程/块，最少 2 waves/SIMD。默认的 4 waves 会把 VGPR 压到 64 并溢出 334 个；
// 放到 2 waves 后是 125 VGPR / 0 溢出，与 Triton 的 119 寄存器同量级。
__global__ __launch_bounds__(64, 2) void gdn_recurrent_update_kernel(
    const void* const* layer_ptrs,          // MULTI: [L, GDN_MULTI_COLS]
    const hip_bfloat16* __restrict__ q,     // [tokens, H,  K]
    const hip_bfloat16* __restrict__ k,     // [tokens, H,  K]
    const hip_bfloat16* __restrict__ v,     // [tokens, HV, V]
    const hip_bfloat16* __restrict__ a,     // [tokens, HV]
    const hip_bfloat16* __restrict__ b,     // [tokens, HV]
    const float* __restrict__ A_log,        // [HV]
    const float* __restrict__ dt_bias,      // [HV]
    const int* __restrict__ cu_seqlens,     // [N+1] or null
    float* __restrict__ h0_source,          // [slots, HV, K, V]
    const int* __restrict__ h0_indices,     // [N]
    hip_bfloat16* __restrict__ o,           // [tokens, HV, V]
    // target_verify 的中间态快照：[slots*D, HV, K, V]（per-layer 视图已摊平），
    // 第 step 份在 cache_idx*cache_steps + step。为 null 时不写（普通 decode）。
    float* __restrict__ snap_buf,
    const int* __restrict__ snap_idx,       // [N]
    const int cache_steps,
    const int* __restrict__ store_step,     // [N] B+：只写 step == store_step[i_n]
    // [N] B+：写第几行快照。-1/空 = 写 store_step 那一行本身（纠偏那一趟）。
    const int* __restrict__ store_row,
    const int T, const int HV, const int K, const int V, const int hratio,
    // a 和 b 的行距**不一定相同**：b 常常是某块更宽 buffer 的切片（实测 96），
    // a 是 48。共用一个 stride_b 会让 a[:t] 读到别的 token、越界后读到 NaN。
    int stride_q, int stride_k, int stride_v,
    int stride_a, int stride_b,
    const float scale, const float softplus_beta, const float softplus_threshold,
    const int use_initial_state, const int varlen, const int disable_state_update) {
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int lane = threadIdx.x;
  const int g = lane & (G - 1);
  const int t1 = lane / G;

  if (MULTI) {
    const long* m = (const long*)layer_ptrs + (long)blockIdx.z * GDN_MULTI_COLS;
    q = (const hip_bfloat16*)m[0];
    k = (const hip_bfloat16*)m[1];
    v = (const hip_bfloat16*)m[2];
    a = (const hip_bfloat16*)m[3];
    b = (const hip_bfloat16*)m[4];
    A_log = (const float*)m[5];
    dt_bias = (const float*)m[6];
    h0_source = (float*)m[7];
    o = (hip_bfloat16*)m[8];
    snap_buf = (float*)m[9];
    stride_q = (int)m[10];
    stride_k = (int)m[11];
    stride_v = (int)m[12];
    stride_a = (int)m[13];
    stride_b = (int)m[14];
  }

  const int i_n = i_nh / HV;
  const int i_hv = i_nh % HV;
  const int i_h = i_hv / hratio;
  // k 的归属用「步长 32 交错」：k = 4g + 32c（c=0..KL/4-1）。
  // 这样一条 float4 里 8 个 lane 正好覆盖 128 B 连续（4g 步长 16 B），
  // 而每 lane 仍只跨 8 个 lane 归约。用「连续 16 个 k」时同样一条指令
  // 8 个 lane 相隔 64 B，只用到 16 B，状态搬运掉到 800 GB/s。
  const int kbase = 4 * g;
  const int kstep = BK / (KL / 4);   // = 32，块间 k 步长
  const int vbase = i_v * BV + VL * t1;

  int bos, eos;
  if (varlen) {
    bos = cu_seqlens[i_n];
    eos = cu_seqlens[i_n + 1];
  } else {
    bos = i_n * T;
    eos = bos + T;
  }

  // state: h[j][r] = h0[k = kbase + j, v = vbase + r]
  float h[KL][VL];
  const int sidx = use_initial_state ? h0_indices[i_n] : -1;
  const int snap_i =
      (snap_buf != nullptr && snap_idx != nullptr) ? snap_idx[i_n] : -1;
  // B+ 纠偏：无效槽（h0_idx < 0）整块退出，让纠偏成本 ∝ 真正要补的槽数。
  if (BPLUS && use_initial_state && sidx < 0) return;
  const long stoff = (long)i_hv * K * V + vbase * K + kbase;
  if (sidx >= 0) {
    const float* p = h0_source + (long)sidx * HV * K * V + stoff;
#pragma unroll
    for (int r = 0; r < VL; ++r)
#pragma unroll
      for (int c = 0; c < KL / 4; ++c) {
        const float4 t = *reinterpret_cast<const float4*>(p + r * K + kstep * c);
        h[4 * c + 0][r] = t.x;
        h[4 * c + 1][r] = t.y;
        h[4 * c + 2][r] = t.z;
        h[4 * c + 3][r] = t.w;
      }
  } else {
#pragma unroll
    for (int j = 0; j < KL; ++j)
#pragma unroll
      for (int r = 0; r < VL; ++r) h[j][r] = 0.0f;
  }

  const float bA = __expf(A_log[i_hv]);
  const float dtb = dt_bias[i_hv];
  const float inv_beta = 1.0f / softplus_beta;

  for (int t = bos; t < eos; ++t) {
    // ---- k (pre-l2norm, bf16)，16 个 = 32 B -------------------------------
    hip_bfloat16 kb[KL];
    {
      const hip_bfloat16* pk = k + (long)t * stride_k + i_h * K + kbase;
#pragma unroll
      for (int c = 0; c < KL / 4; ++c)
        *reinterpret_cast<uint2*>(kb + 4 * c) =
            *reinterpret_cast<const uint2*>(pk + kstep * c);
    }
    float kk[KL];
#pragma unroll
    for (int j = 0; j < KL; ++j) kk[j] = float(kb[j]);

    // ---- gating: x = a + dt_bias; softplus; g = -exp(A_log) * x ----------
    const float av = float(a[(long)t * stride_a + i_hv]);
    float x = __fadd_rn(av, dtb);
    const float xs = __fmul_rn(softplus_beta, x);
    if (xs <= softplus_threshold) {
      x = __fmul_rn(inv_beta, __logf(__fadd_rn(1.0f, __expf(xs))));
    }
    const float g_ = __fmul_rn(-bA, x);

    // ---- l2norm(k) ------------------------------------------------------
    float ss = __fmul_rn(kk[0], kk[0]);
#pragma unroll
    for (int j = 1; j < KL; ++j) ss = __fadd_rn(ss, __fmul_rn(kk[j], kk[j]));
    ss = grp_sum(ss);
    const float invk = rsqrtf(__fadd_rn(ss, 1e-6f));
#pragma unroll
    for (int j = 0; j < KL; ++j) kk[j] = __fmul_rn(kk[j], invk);

    // ---- h *= exp(g) ----------------------------------------------------
    const float eg = __expf(g_);
#pragma unroll
    for (int j = 0; j < KL; ++j)
#pragma unroll
      for (int r = 0; r < VL; ++r) h[j][r] = __fmul_rn(h[j][r], eg);

    // ---- v (bf16 -> fp32)，本 lane 独占 4 个 -------------------------------
    float vv_[VL];
    {
      const hip_bfloat16* pv = v + (long)t * stride_v + i_hv * V + vbase;
      hip_bfloat16 packed[VL];
      *reinterpret_cast<vpack_t*>(packed) = *reinterpret_cast<const vpack_t*>(pv);
#pragma unroll
      for (int r = 0; r < VL; ++r) vv_[r] = float(packed[r]);
    }

    // ---- delta: s[r] = sum_k h[k,r] * k[k] -------------------------------
    float part[VL];
#pragma unroll
    for (int r = 0; r < VL; ++r) {
      float s = __fmul_rn(h[0][r], kk[0]);
#pragma unroll
      for (int j = 1; j < KL; ++j) s = __fadd_rn(s, __fmul_rn(h[j][r], kk[j]));
      part[r] = grp_sum(s);
    }

    // ---- v = (v - s) * sigmoid(b) ---------------------------------------
    const float bv = float(b[(long)t * stride_b + i_hv]);
    const float beta = 1.0f / (1.0f + __expf(-bv));
#pragma unroll
    for (int r = 0; r < VL; ++r)
      vv_[r] = __fmul_rn(__fadd_rn(vv_[r], -part[r]), beta);

    // ---- h += k (x) v ----------------------------------------------------
    if (FMA_OUTER) {
#pragma unroll
      for (int j = 0; j < KL; ++j)
#pragma unroll
        for (int r = 0; r < VL; ++r)
          h[j][r] = __fmaf_rn(kk[j], vv_[r], h[j][r]);
    } else {
#pragma unroll
      for (int j = 0; j < KL; ++j)
#pragma unroll
        for (int r = 0; r < VL; ++r)
          h[j][r] = __fadd_rn(h[j][r], __fmul_rn(kk[j], vv_[r]));
    }

    // ---- 中间态快照（target_verify）：每 token 一份完整 state --------------
    // 产线 T=5，即每槽每层写 5 份 3 MB 的 fp32 状态；这一步的带宽不可省，
    // 但可以藏进算的后面（store 无依赖）。
    if (snap_buf != nullptr && snap_i >= 0 &&
        (!BPLUS || (store_step != nullptr && store_step[i_n] == (t - bos)))) {
      // 写哪一行：默认就是「第几步」（纠偏那一趟重放后写提交行）；verify 传
      // store_row=(prev_step+1)%D，故意避开 h0 那一行——见文件头 B+ 注释。
      const int row = (BPLUS && store_row != nullptr) ? store_row[i_n] : (t - bos);
      if (row >= 0) {
      float* ps = snap_buf + ((long)snap_i * cache_steps + row) * HV * K * V
                  + stoff;
#pragma unroll
      for (int r = 0; r < VL; ++r)
#pragma unroll
        for (int c = 0; c < KL / 4; ++c) {
          float4 s4;
          s4.x = h[4 * c + 0][r];
          s4.y = h[4 * c + 1][r];
          s4.z = h[4 * c + 2][r];
          s4.w = h[4 * c + 3][r];
          *reinterpret_cast<float4*>(ps + r * K + kstep * c) = s4;
        }
      }
    }

    if (!STATE_ONLY) {
    // ---- q，同一套归一化 ------------------------------------------------
    hip_bfloat16 qb[KL];
    {
      const hip_bfloat16* pq = q + (long)t * stride_q + i_h * K + kbase;
#pragma unroll
      for (int c = 0; c < KL / 4; ++c)
        *reinterpret_cast<uint2*>(qb + 4 * c) =
            *reinterpret_cast<const uint2*>(pq + kstep * c);
    }
    float qq[KL];
#pragma unroll
    for (int j = 0; j < KL; ++j) qq[j] = float(qb[j]);
    float sq = __fmul_rn(qq[0], qq[0]);
#pragma unroll
    for (int j = 1; j < KL; ++j) sq = __fadd_rn(sq, __fmul_rn(qq[j], qq[j]));
    sq = grp_sum(sq);
    const float invq = rsqrtf(__fadd_rn(sq, 1e-6f));
#pragma unroll
    for (int j = 0; j < KL; ++j) qq[j] = __fmul_rn(__fmul_rn(qq[j], invq), scale);

    // ---- o[r] = sum_k h[k,r] * q[k] --------------------------------------
    float oacc[VL];
#pragma unroll
    for (int r = 0; r < VL; ++r) {
      float s = __fmul_rn(h[0][r], qq[0]);
#pragma unroll
      for (int j = 1; j < KL; ++j) s = __fadd_rn(s, __fmul_rn(h[j][r], qq[j]));
      oacc[r] = grp_sum(s);
    }
    {
      hip_bfloat16 packed[VL];
#pragma unroll
      for (int r = 0; r < VL; ++r) packed[r] = hip_bfloat16(oacc[r]);
      hip_bfloat16* po = o + (long)t * HV * V + i_hv * V + vbase;
      *reinterpret_cast<vpack_t*>(po) = *reinterpret_cast<vpack_t*>(packed);
    }
    }  // !STATE_ONLY
  }

  if (!disable_state_update && sidx >= 0) {
    float* p = h0_source + (long)sidx * HV * K * V + stoff;
#pragma unroll
    for (int r = 0; r < VL; ++r)
#pragma unroll
      for (int c = 0; c < KL / 4; ++c) {
        float4 t;
        t.x = h[4 * c + 0][r];
        t.y = h[4 * c + 1][r];
        t.z = h[4 * c + 2][r];
        t.w = h[4 * c + 3][r];
        *reinterpret_cast<float4*>(p + r * K + kstep * c) = t;
      }
  }
}

// ---------------------------------------------------------------------------
// host 入口（ctypes 直接调）
// ---------------------------------------------------------------------------
template <bool BPLUS, bool STATE_ONLY>
static void launch(const void* q, const void* k, const void* v, const void* a,
                   const void* b, const void* A_log, const void* dt_bias,
                   const void* cu_seqlens, void* h0_source, const void* h0_indices,
                   void* o, void* snap_buf, const void* snap_idx,
                   int cache_steps, const void* store_step,
                   const void* store_row,
                   int N, int T, int HV, int K, int V, int hratio,
                   int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
                   float scale, float softplus_beta, float softplus_threshold,
                   int use_initial_state, int varlen, int disable_state_update,
                   int fma_outer, hipStream_t stream) {
  dim3 grid(V / BV, N * HV);
  dim3 block(64);
  if (fma_outer) {
    gdn_recurrent_update_kernel<true, BPLUS, STATE_ONLY><<<grid, block, 0, stream>>>(
        nullptr,
        (const hip_bfloat16*)q, (const hip_bfloat16*)k, (const hip_bfloat16*)v,
        (const hip_bfloat16*)a, (const hip_bfloat16*)b, (const float*)A_log,
        (const float*)dt_bias, (const int*)cu_seqlens, (float*)h0_source,
        (const int*)h0_indices, (hip_bfloat16*)o, (float*)snap_buf,
        (const int*)snap_idx, cache_steps, (const int*)store_step,
        (const int*)store_row,
        T, HV, K, V, hratio,
        stride_q, stride_k, stride_v, stride_a, stride_b, scale, softplus_beta,
        softplus_threshold, use_initial_state, varlen, disable_state_update);
  } else {
    gdn_recurrent_update_kernel<false, BPLUS, STATE_ONLY><<<grid, block, 0, stream>>>(
        nullptr,
        (const hip_bfloat16*)q, (const hip_bfloat16*)k, (const hip_bfloat16*)v,
        (const hip_bfloat16*)a, (const hip_bfloat16*)b, (const float*)A_log,
        (const float*)dt_bias, (const int*)cu_seqlens, (float*)h0_source,
        (const int*)h0_indices, (hip_bfloat16*)o, (float*)snap_buf,
        (const int*)snap_idx, cache_steps, (const int*)store_step,
        (const int*)store_row,
        T, HV, K, V, hratio,
        stride_q, stride_k, stride_v, stride_a, stride_b, scale, softplus_beta,
        softplus_threshold, use_initial_state, varlen, disable_state_update);
  }
}

// MULTI: one launch for L layers, layer chosen by blockIdx.z. Per-layer
// pointers and token strides come from the int64 meta table.
template <bool STATE_ONLY>
static void launch_multi(const void* layer_ptrs, const void* h0_indices,
                         const void* snap_idx, int cache_steps,
                         const void* store_step, int L,
                         int N, int T, int HV, int K, int V, int hratio,
                         float scale, float softplus_beta,
                         float softplus_threshold, int use_initial_state,
                         int varlen, int disable_state_update,
                         hipStream_t stream) {
  dim3 grid(V / BV, N * HV, L);
  dim3 block(64);
  gdn_recurrent_update_kernel<true, true, STATE_ONLY, true>
      <<<grid, block, 0, stream>>>(
      (const void* const*)layer_ptrs,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, (const int*)h0_indices, nullptr, nullptr,
      (const int*)snap_idx,
      cache_steps, (const int*)store_step, nullptr,
      T, HV, K, V, hratio, 0, 0, 0, 0, 0,
      scale, softplus_beta, softplus_threshold,
      use_initial_state, varlen, disable_state_update);
}

extern "C" void gdn_recurrent_update(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const void* A_log, const void* dt_bias, const void* cu_seqlens,
    void* h0_source, const void* h0_indices, void* o, void* snap_buf,
    const void* snap_idx, int cache_steps, int N, int T, int HV, int K, int V,
    int hratio, int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch<false, false>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
         h0_indices, o, snap_buf, snap_idx, cache_steps, nullptr, nullptr,
         N, T, HV, K, V, hratio,
         stride_q, stride_k, stride_v, stride_a, stride_b, scale,
         softplus_beta, softplus_threshold, use_initial_state, varlen,
         disable_state_update, 1, (hipStream_t)stream);
}

// ---- B+：只写 store_step 指定那一份快照 -----------------------------------
// store_step[N] 为 device 指针：第 i 槽只写 step == store_step[i]，-1 = 不写。
// h0_idx[i] < 0 的槽整块退出（纠偏时用它跳过全接受的槽）。
extern "C" void gdn_recurrent_update_bplus(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const void* A_log, const void* dt_bias, const void* cu_seqlens,
    void* h0_source, const void* h0_indices, void* o, void* snap_buf,
    const void* snap_idx, int cache_steps, const void* store_step,
    const void* store_row,
    int N, int T, int HV, int K, int V,
    int hratio, int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch<true, false>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
         h0_indices, o, snap_buf, snap_idx, cache_steps, store_step, store_row,
         N, T, HV, K, V, hratio,
         stride_q, stride_k, stride_v, stride_a, stride_b, scale,
         softplus_beta, softplus_threshold, use_initial_state, varlen,
         disable_state_update, 1, (hipStream_t)stream);
}

// 纠偏专用：同样的 store_step 语义，但不算 q 的 l2norm 和 o = h·q。
extern "C" void gdn_recurrent_update_bplus_state(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const void* A_log, const void* dt_bias, const void* cu_seqlens,
    void* h0_source, const void* h0_indices, void* o, void* snap_buf,
    const void* snap_idx, int cache_steps, const void* store_step,
    const void* store_row,
    int N, int T, int HV, int K, int V,
    int hratio, int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch<true, true>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
         h0_indices, o, snap_buf, snap_idx, cache_steps, store_step, store_row,
         N, T, HV, K, V, hratio,
         stride_q, stride_k, stride_v, stride_a, stride_b, scale,
         softplus_beta, softplus_threshold, use_initial_state, varlen,
         disable_state_update, 1, (hipStream_t)stream);
}

extern "C" void gdn_recurrent_update_nofma(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    const void* A_log, const void* dt_bias, const void* cu_seqlens,
    void* h0_source, const void* h0_indices, void* o, void* snap_buf,
    const void* snap_idx, int cache_steps, int N, int T, int HV, int K, int V,
    int hratio, int stride_q, int stride_k, int stride_v,
                   int stride_a, int stride_b,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch<false, false>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
         h0_indices, o, snap_buf, snap_idx, cache_steps, nullptr, nullptr,
         N, T, HV, K, V, hratio,
         stride_q, stride_k, stride_v, stride_a, stride_b, scale,
         softplus_beta, softplus_threshold, use_initial_state, varlen,
         disable_state_update, 0, (hipStream_t)stream);
}

// B+ 纠偏的多层入口：一次 launch 覆盖全部 GDN 层，层号 = blockIdx.z。
extern "C" void gdn_recurrent_update_bplus_state_multi(
    const void* layer_ptrs, const void* h0_indices, const void* snap_idx,
    int cache_steps, const void* store_step, int L,
    int N, int T, int HV, int K, int V, int hratio,
    float scale, float softplus_beta, float softplus_threshold,
    int use_initial_state, int varlen, int disable_state_update, void* stream) {
  launch_multi<true>(layer_ptrs, h0_indices, snap_idx, cache_steps, store_step,
                     L, N, T, HV, K, V, hratio, scale, softplus_beta,
                     softplus_threshold, use_initial_state, varlen,
                     disable_state_update, (hipStream_t)stream);
}
