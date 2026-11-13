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

// h += k*v 用不用 FMA。Triton 侧是两条独立指令（fmul + fadd，LLIR 里没有
// fmuladd），所以逐位对齐必须关掉 FMA；留开关只为 A/B。
#ifndef BPLUS_FMA
#define BPLUS_FMA 1
#endif
// l2norm 的倒数：0=rsqrtf(HW v_rsq_f32 近似) 1=__frsqrt_rn(正确舍入)
// 2=1/sqrtf(IEEE 除)。Triton tl.rsqrt 在 AMD 上未必是 HW 近似，留开关对拍。
#ifndef BPLUS_RSQRT
#define BPLUS_RSQRT 0
#endif

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

// ---- 与 Triton 完全同序的归约树 -------------------------------------------
// aiter 的 tl.sum 在 BK=128 上按 32 个「k 组」（每组 4 个连续 k）归约：
//   组内 d=0..3 顺序累加；32 组的 xor 树是 16,8,4,2,1。
// 本核的 k 归属 k = 4g + 32c 正好把 Triton 线程 tk = g + 8c 的 4 个组放进
// 同一 lane，所以 16/8 两级是 lane 内配对 —— (c0,c2) 与 (c1,c3) —— 只有
// 4/2/1 三级是 8-lane shuffle，shuffle 数与原来的平铺链完全相同。
static_assert(KL == 16, "bitwise 归约树按 KL=16（G=8）展开");
// Triton 的 delta/o 归约里乘积是**独立**的 v_mul_f32，不能被后端折进 v_fma
// （逐位对拍已证实：折了就 1 ULP 漂移，约 3/4 行中招）。
// 两种挡法：1=inline asm（一定有效，但丢掉 v_pk_mul_f32 打包）；
//           0=照常写 __fmul_rn，靠 -mllvm=-not-combine-fma 关掉后端的折 fma。
#ifndef BPLUS_NOFOLD_ASM
#define BPLUS_NOFOLD_ASM 1
#endif
__device__ __forceinline__ float mul_nofold(float a, float b) {
#if BPLUS_NOFOLD_ASM
  float d;
  asm volatile("v_mul_f32 %0, %1, %2" : "=v"(d) : "v"(a), "v"(b));
  return d;
#else
  return __fmul_rn(a, b);
#endif
}
__device__ __forceinline__ float sum4_seq(float x0, float x1, float x2, float x3) {
  return __fadd_rn(__fadd_rn(__fadd_rn(x0, x1), x2), x3);
}
__device__ __forceinline__ float grp_tree(float g0, float g1, float g2, float g3) {
  float s = __fadd_rn(__fadd_rn(g0, g2), __fadd_rn(g1, g3));
  s = __fadd_rn(s, __shfl_xor(s, 4, 8));
  s = __fadd_rn(s, __shfl_xor(s, 2, 8));
  s = __fadd_rn(s, __shfl_xor(s, 1, 8));
  return s;
}

// l2norm 的归约布局和 state 不一样：TTGIR 里是 #blocked2 -> reshape(#blocked1)
// 再 reduce，每线程先配相邻的 (2t,2t+1)，然后按 t 做 xor 32,16,8,4,2,1。
// 映射到 k=4g+32c+d：xor32/16 -> c^2/c^1（lane 内），xor8/4/2 -> g^4/2/1，
// xor1 是 d 的高位、排在 shuffle 之后。所以带 2 份部分和做 3 级 shuffle。
// l2norm 的归约树：从缓存的 .llir 逐条还原（与 state 的树不同）
//   叶子 q_i = x[2i] + x[2i+1]           —— 同一线程相邻两个 k
//   组内 row_shr 8,4,2,1 的移位链      —— 每 16 个 q 一组，取组末 lane
//   组间 (T0+T1)+(T2+T3)
// 映射到 k = 4g+32c+d：Triton lane = 2g+16c+d/2，所以
//   c ↔ 组，d/2 ↔ 组内 e；移位链在 g 方向是 g-4 / g-2 / g-1，
//   最后一级 row_shr:1 是 e 翻转（同一个 lane 内）。
// 只有 g==7 的 lane 拿到 T_c，再广播给 8-lane 组。
__device__ __forceinline__ float norm_grp(float q0, float q1, int g) {
  float t;
  t = __shfl_up(q0, 4, 8); q0 = g >= 4 ? __fadd_rn(q0, t) : q0;
  t = __shfl_up(q1, 4, 8); q1 = g >= 4 ? __fadd_rn(q1, t) : q1;
  t = __shfl_up(q0, 2, 8); q0 = g >= 2 ? __fadd_rn(q0, t) : q0;
  t = __shfl_up(q1, 2, 8); q1 = g >= 2 ? __fadd_rn(q1, t) : q1;
  t = __shfl_up(q0, 1, 8); q0 = g >= 1 ? __fadd_rn(q0, t) : q0;
  t = __shfl_up(q1, 1, 8); q1 = g >= 1 ? __fadd_rn(q1, t) : q1;
  return __shfl(__fadd_rn(q1, q0), 7, 8);
}

#define NORM_SQ(x, y) __fadd_rn(__fmul_rn(x, x), __fmul_rn(y, y))

__device__ __forceinline__ float norm_inv(float x) {
#if BPLUS_RSQRT == 0
  return rsqrtf(x);
#elif BPLUS_RSQRT == 1
  return __frsqrt_rn(x);
#else
  return 1.0f / sqrtf(x);
#endif
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
template <bool FMA_OUTER, bool BPLUS = false, bool STATE_ONLY = false>
// 64 线程/块，最少 2 waves/SIMD。默认的 4 waves 会把 VGPR 压到 64 并溢出 334 个；
// 放到 2 waves 后是 125 VGPR / 0 溢出，与 Triton 的 119 寄存器同量级。
__global__ __launch_bounds__(64, 2) void gdn_recurrent_update_kernel(
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
    const int stride_q, const int stride_k, const int stride_v,
    const int stride_a, const int stride_b,
    const float scale, const float softplus_beta, const float softplus_threshold,
    const int use_initial_state, const int varlen, const int disable_state_update) {
  const int i_v = blockIdx.x;
  const int i_nh = blockIdx.y;
  const int lane = threadIdx.x;
  const int g = lane & (G - 1);
  const int t1 = lane / G;

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

    // ---- l2norm(k)：Triton 的相邻配对 + 64-lane 树 -------------------------
    // p_d = (kk[d]² + kk[8+d]²) + (kk[4+d]² + kk[12+d]²) = (c0+c2)+(c1+c3)
    const float t0 = norm_grp(NORM_SQ(kk[0], kk[1]), NORM_SQ(kk[2], kk[3]), g);
    const float t1 = norm_grp(NORM_SQ(kk[4], kk[5]), NORM_SQ(kk[6], kk[7]), g);
    const float t2 = norm_grp(NORM_SQ(kk[8], kk[9]), NORM_SQ(kk[10], kk[11]), g);
    const float t3 = norm_grp(NORM_SQ(kk[12], kk[13]), NORM_SQ(kk[14], kk[15]), g);
    const float ss = __fadd_rn(__fadd_rn(t0, t1), __fadd_rn(t2, t3));
    const float invk = norm_inv(__fadd_rn(ss, 1e-6f));
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
      part[r] = grp_tree(
          sum4_seq(mul_nofold(h[0][r], kk[0]), mul_nofold(h[1][r], kk[1]),
                   mul_nofold(h[2][r], kk[2]), mul_nofold(h[3][r], kk[3])),
          sum4_seq(mul_nofold(h[4][r], kk[4]), mul_nofold(h[5][r], kk[5]),
                   mul_nofold(h[6][r], kk[6]), mul_nofold(h[7][r], kk[7])),
          sum4_seq(mul_nofold(h[8][r], kk[8]), mul_nofold(h[9][r], kk[9]),
                   mul_nofold(h[10][r], kk[10]), mul_nofold(h[11][r], kk[11])),
          sum4_seq(mul_nofold(h[12][r], kk[12]), mul_nofold(h[13][r], kk[13]),
                   mul_nofold(h[14][r], kk[14]), mul_nofold(h[15][r], kk[15])));
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
    const float t0 = norm_grp(NORM_SQ(qq[0], qq[1]), NORM_SQ(qq[2], qq[3]), g);
    const float t1 = norm_grp(NORM_SQ(qq[4], qq[5]), NORM_SQ(qq[6], qq[7]), g);
    const float t2 = norm_grp(NORM_SQ(qq[8], qq[9]), NORM_SQ(qq[10], qq[11]), g);
    const float t3 = norm_grp(NORM_SQ(qq[12], qq[13]), NORM_SQ(qq[14], qq[15]), g);
    const float sq = __fadd_rn(__fadd_rn(t0, t1), __fadd_rn(t2, t3));
    const float invq = norm_inv(__fadd_rn(sq, 1e-6f));
#pragma unroll
    for (int j = 0; j < KL; ++j) qq[j] = __fmul_rn(__fmul_rn(qq[j], invq), scale);

    // ---- o[r] = sum_k h[k,r] * q[k] --------------------------------------
    float oacc[VL];
#pragma unroll
    for (int r = 0; r < VL; ++r) {
      oacc[r] = grp_tree(
          sum4_seq(mul_nofold(h[0][r], qq[0]), mul_nofold(h[1][r], qq[1]),
                   mul_nofold(h[2][r], qq[2]), mul_nofold(h[3][r], qq[3])),
          sum4_seq(mul_nofold(h[4][r], qq[4]), mul_nofold(h[5][r], qq[5]),
                   mul_nofold(h[6][r], qq[6]), mul_nofold(h[7][r], qq[7])),
          sum4_seq(mul_nofold(h[8][r], qq[8]), mul_nofold(h[9][r], qq[9]),
                   mul_nofold(h[10][r], qq[10]), mul_nofold(h[11][r], qq[11])),
          sum4_seq(mul_nofold(h[12][r], qq[12]), mul_nofold(h[13][r], qq[13]),
                   mul_nofold(h[14][r], qq[14]), mul_nofold(h[15][r], qq[15])));
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
  launch<(BPLUS_FMA != 0), false>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
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
  launch<(BPLUS_FMA != 0), true>(q, k, v, a, b, A_log, dt_bias, cu_seqlens, h0_source,
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
