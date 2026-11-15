# HCU sglang plugin — 现状（2026-11-15）

只写**当前生效**的开关、已验证范围与已知未验证项。历史过程与逐日数据在
`docs/history/hcu-sglang-development-2026-10-11.md`，排查记录在 `docs/investigations/`。

## 平台语义

- HCU 以 **CUDA 语义**对外：`is_hip()` 恒 False（`hcu_adapter/platform_seam.py`）。
  因此上游所有 "ROCm 专用" 分支在 HCU 上不会自动触发，插件必须显式接管。

## 默认开启（生产等价）

| 开关 | 默认 | 作用 | 验证 |
|---|---|---|---|
| `HBIRD_GEMM_SGLANG` + `HBIRD_GEMM_ROLE` | 外部脚本开启 | bf16 linear / lm_head 走 hbird-gemm；`hcu_sglang/layers/gemm/` 是 provider 层 | 20: target 256 层打包、45.31 GiB 释放、skipped 48；T=0 sha 不变 |
| `SGLANG_GDN_HIP_BPLUS` + `_HANDOFF` + `_HANDOFF_SKIP_TRACK` | 脚本开启 | GDN verify 只写提交行，commit 只移指针 | 空答案 5/130 → 1/164；accept len 回到 2.50（T=0.2） |
| `SGLANG_GDN_HIP_BPLUS_SKIP_FULL` | 1 | compensate 只重放截断槽，全接受槽跳过 | 受控 probe 100.5 → 93.2 ms/步 |
| `SGLANG_GDN_HIP_BPLUS_MULTI` | 1 | 48 层 compensate 合并一次 launch；旧 .so 自动回退 | c40 上中性（<0.3 ms） |
| HCU 自动 rejection sampling（无开关，`speculative_compat.py`） | 开 | 非贪心请求避免走缺头的 CUDA JIT；`SGLANG_HCU_AUTO_REJECTION_SAMPLING=0` 可关 | T=0.2 全量门可跑完 |

## 默认关闭（实验/A-B 用）

| 开关 | 默认 | 状态 |
|---|---|---|
| `SGLANG_HCU_FUSE_QKVZBA_VERIFY` | 0 | verify 零拷贝视图；单波 TPOT 26.18→26.06，逐位不变 |
| `SGLANG_GDN_ZBA_STRIDED` | 0 | z/b 保持 strided；与上项同开才不亏 |
| `SGLANG_HCU_FUSE_SILU` | 0 | Triton `_silu_mul_kernel` 取代 lightop `silu_and_mul_kernel_opt2`（保留两次 bf16 舍入）；逐位对拍待 GPU 验证 |
| `SGLANG_HCU_FUSE_ADD_RMS` | 0 | 单遍 Gemma add-RMSNorm（一次读 x/r、一次写 residual/x），语义与 adapter 两遍 Triton 核逐位相同；待 GPU 验证 |
| `SGLANG_GDN_HIP_KERNEL` | 0 | kernel-only 线（非 B+）；默认关 |
| `SGLANG_FORCE_MM_FAST_PROCESSOR_GPU` | 0 | 打开会让 tokenizer 进程建第二个 CUDA context（HCU 驱动 runlist 超订），**不要开** |

## 已确认的坑（会崩/会错，不是性能问题）

1. **tokenizer 进程不能建 CUDA context**：HCU 驱动按卡统计 compute queue（30 > 28）后
   全卡慢 1.55–1.8× 且状态粘滞，只有重启进程才恢复。sglang 侧修复 + hbird 守卫已落地。
2. **T>0 必须走 rejection sampling**：否则 `tree_speculative_sampling_target_only` 的
   CUDA JIT 缺 flashinfer 头，scheduler 直接死。已由 plugin hook 自动开启。
3. **投机 verify 的 top-p 归一化**：adapter 侧用阈值式实现（避免 int64 索引张量 OOM），
   并列值语义与"最小前缀"略有差异。
4. **B+ handoff 的状态提交必须用 live batch 数据**：capture 期记录的 `snap_idx`/`T` 在
   bs>1 时会让截断槽状态不重放 → 空最终答案。

## 当前水位（同卡、40 请求单波、2k/2k、TEMP=0、双方 40/40）

| | 产线 0.5.18 | clean 0.5.22 |
|---|---:|---:|
| 吞吐 | 1236.93 tok/s | 1026.55 |
| Mean TPOT | 24.20 ms | 27.07 |
| Mean TTFT | 12.29 s | 12.39 |
| Mean ITL | 94.44 ms | 104.09 |

差距分解：verify +30.6 ms/步、draft_extend +15.0 ms/步，主机侧我们快 12 ms/步。
精度：投机 12w 全量门 161/163 = 98.77%（空答案 0）。详见 `/workspace/clean-stack/PERF_RECORD.md`。

## 未验证

- 非 handoff（`HANDOFF=0`）分支未做硬件 A/B。
- EAGLE topk>1 / deterministic inference 下 rejection sampling 保持关闭（未测）。
- `SGLANG_ENABLE_PP_SPEC` 的 stable-rows 分支未验证。
- 多卡 / TP>1 未验证。
