# HCU sglang plugin — 现状（2026-11-15）

只写**当前生效**的开关、已验证范围与已知未验证项。历史过程与逐日数据在
`docs/history/`，排障入口见 [故障诊断](troubleshooting.md)，交付证据见
[交付验证记录](delivery-validation.md)。

## 平台语义

- HCU 以 **CUDA 语义**对外：`is_hip()` 恒 False（adapter 仓 `adapter-das` 的
  `hcu_adapter/platform_seam.py`）。
  因此上游所有 "ROCm 专用" 分支在 HCU 上不会自动触发，插件必须显式接管。

## 默认开启（生产等价）

| 开关 | 默认 | 作用 | 验证 |
|---|---|---|---|
| `HBIRD_GEMM_SGLANG` + `HBIRD_GEMM_ROLE` | 外部脚本开启 | bf16 linear / lm_head 走 hbird-gemm；`hcu_sglang/layers/gemm/` 是 provider 层 | target 打包 256 层、释放 45.31 GiB、skipped 48，draft 打包 4 层；T=0 sha 不变 |
| `SGLANG_GDN_HIP_BPLUS` + `_HANDOFF` + `_HANDOFF_SKIP_TRACK` | 脚本开启 | GDN verify 只写提交行，commit 只移指针 | 空答案 5/130 → 1/164；accept len 回到 2.50（T=0.2） |
| `SGLANG_GDN_HIP_BPLUS_SKIP_FULL` | 1 | compensate 只重放截断槽，全接受槽跳过；关（=0）则全接受槽也重放 | 受控 probe 100.5 → 93.3 ms/步（=1）；=0 路径在 plain 分支曾静默不重放，已修并加回归测试 |
| `SGLANG_GDN_HIP_BPLUS_MULTI` | 1 | 48 层 compensate 合并一次 launch；旧 .so 自动回退 | c40 上中性（<0.3 ms） |
| B+ verify 每步共享索引（无开关，`gdn_bplus.shared_step_values`） | 开 | verify 的 `store`/`store_row`/h0/prev 只依赖 batch，不依赖层：48 层各算一遍改成整步算一次并复用持久 buffer；同时删掉只在写死字段、无人读的 `remember_step`/`tdev.fill_` | 离线 200 组随机逐位全等（`tests/test_gdn_bplus_step_shared.py`，另有一次仓外 200 组随机对拍）；GPU6 c40 单波 40/40、Mean TPOT 26.05、sha 与基线逐位相同；eager 2622 → 1677 kernel/步（B+ 零碎核 993 → 34/步，5.21 → 0.20 ms/步） |
| HCU 自动 rejection sampling（无开关，`speculative_compat.py`） | 开 | 非贪心请求避免走缺头的 CUDA JIT；`SGLANG_HCU_AUTO_REJECTION_SAMPLING=0` 可关 | T=0.2 全量门可跑完 |
| `SGLANG_HCU_FUSE_QKVZBA_VERIFY` + `SGLANG_GDN_ZBA_STRIDED` | 1（HCU 默认） | verify 零拷贝 qkvzba 视图 + z/b 保持 strided，split 核消失 | 40/40 TPOT 27.01→26.67，T=0 sha 不变 |
| `SGLANG_HCU_FUSE_SILU` | 1（HCU 默认） | Triton silu，保留两次 bf16 舍入，取代 lightop 编译核 | 全 shape 逐位对拍 0 mismatch；核 12.99→11.14 µs/call；40/40 TPOT 在噪声内 |
| `SGLANG_HCU_FA_PAGED_BSHD` | 脚本开启（代码默认 0） | decode/MTP-verify 走 wheel `layout="bshd"` 直连 paged 核（NHD 池无需换布局）。**改变 T=0 输出**：与 splitkv 不是同一数值实现 | 40/40 TPOT 26.90→26.10；与产线 custom paged 逐位一致（`logs/pa_layout_equiv.log`）；精度门只挂 #47/#145，两者基线也红 |
| `SGLANG_MTP_LMHEAD_FP8`（`_BACKEND`/`_CHUNK`） | 脚本开启 | 只替换 draft 模型的 lm_head（target head 不动），e4m3 + hbird fp8 整词表一次 launch | fp8+bshd 40/40 TPOT 25.03、`logs/bench_fp8_bshd.log`。**依赖 `--mem-fraction-static` ≤0.86**：0.88 下 verify 采样 0 bytes free 直接 OOM；0.86 起精度门 19/27、9/9 |
| `SGLANG_HCU_FUSE_ADD_RMS` | 1（HCU 默认） | 单遍 Gemma add-RMSNorm（x/r 各读一次、residual/x 各写一次） | 全 shape 逐位对拍 0 mismatch；核 18.88→17.14 µs/call（含两次拷贝）；全开 40/40 TPOT 26.58，默认配置（无 env）40/40 TPOT 26.51，T=0 sha 不变 |

## 默认关闭（实验/A-B 用）

| 开关 | 默认 | 状态 |
|---|---|---|
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

最近一轮（`qwen3.8-27b-perf` 合并后，fp8 草稿头 + bshd + `mem-fraction-static 0.87`）：

| | 产线 0.5.18 | clean 0.5.22（合并后） |
|---|---:|---:|
| 吞吐 | 1236.93 tok/s | 1204.33 |
| Mean TPOT | 24.20 ms | 24.87 |
| Mean ITL | 94.44 ms | 97.42 |
| Median ITL | 81.61 ms | 84.53 |

差 TPOT +2.8%、吞吐 −2.6%。精度门 20/27 + CONTROLS 9/9，与 bf16 基线持平；
投机 12w 全量门 161/163 = 98.77%（空答案 0）。日志 `logs/bench_merged.log`。

历史对照（B/C 补丁前，同一口径）：1026.55 tok/s / TPOT 27.07 / ITL 104.09，
差距分解为 verify +30.6 ms/步、draft_extend +15.0 ms/步、主机侧我们快 12 ms/步。
详见 `/workspace/clean-stack/PERF_RECORD.md`。

## decode/verify paged attention（2026-11-15，`SGLANG_HCU_FA_PAGED_BSHD` 之上）

> 本节与"当前水位"里的 `logs/*` 都指 `/workspace/clean-stack/logs/`（仓外实测目录）。

口径：Qwen3.8-27B、40 请求单波 2k/2k、TEMP=0、40/40、重启后第一轮、CUDA graph。

- **产线 0.5.18 的 decode/verify 也用 `paged_attention_combine_parallel`**。`logs/fk_kernels_prod.tsv`
  里 595 次 `paged_attention_kernel_fp8_opt` 203.78 µs 配 595 次
  `paged_attention_combine_parallel` 30.03 µs；单次 `paged_attention_combine` 9.51 µs 只与
  非 opt 的 `paged_attention_kernel_fp8`（166.23 µs）配对。parallel/单次跟随主核族，
  `num_splits` 与 `max_seqlen_k` 都改不了它（wheel 的两个入口都不把 `num_splits` 传给
  `paged_attention`；launcher 把 ≤10 或 >8192 的 `max_seqlen_k` 归一到 meanseq）。
- graph 真实负载下用时间戳对拍（绕开 trace 时长错位一格，每次主核到 combine 后一个事件
  的 span）：产线 0.5.18 `ab_prod518_clean`（40/40）231.1 µs、我们 bshd 251.8 µs、
  bhsd(HND) 259.4 µs，splitkv 350.9 µs。产线 custom 布局仍快我们 20–28 µs/次；
  microbench 的 144–226 µs（`logs/pa_real_pool_bench.log`）是紧凑池/窄页表的乐观值。
- 四臂（同一 worktree、重启后第一轮、均 40/40）：

  | 臂 | tok/s | Mean TPOT | Mean ITL | 主核 |
  |---|---:|---:|---:|---|
  | baseline splitkv（NHD） | 1075.97 | 26.85 | 103.88 | `flash_fwd_splitkv_fp8_gfx938` |
  | bshd（NHD） | 1161.57 | 26.10 | 102.42 | `paged_attention_kernel_bshd_fp8_opt` |
  | HND（bhsd） | 1087.02 | 26.36 | 101.98 | `paged_attention_kernel_bhsd_fp8_opt` |
  | custom（plugin HCU 池） | 962.88 | 31.54 | 122.24 | `paged_attention_kernel_fp8_opt` |

  attention 换核把 ITL 降了 ~1.9 ms/step（与 19 次 × ~90 µs 的核差一致），TPOT 只降 ~0.5：
  TPOT = step_time/(1+accept)，实测 ITL/TPOT 在四臂为 3.87–3.92，即每步约 3.9 个 token，
  ~1.9 ms 的步时收益只能折成 ~0.5 ms TPOT。
- `SGLANG_USE_HND_KVCACHE=1` 用官方 page-major 池；插件按 stride 识别
  `paged_kv_view` 的 HND 视图，还原物理 `[pages, heads, page, dim]` 后按 `layout="bhsd"`
  调 wheel（prefill/decode/verify 全走 bhsd）。decode 与 torch 参考对拍 max_abs 4.3e-3，
  三布局互相逐位相同（`logs/pa_layout_equiv.log`、`logs/pa_prefill_equiv.log`）。
- plugin 的 HCU page-major 池（`SGLANG_KV_LAYOUT_HCU_FA=1`，V 转置布局）仍慢：TPOT 31.54，
  与 08:10 的 31.90 一致，**继续默认关**。它走的是产线同款 custom 布局入口（未单独抓核
  trace），代价在换池后的写路径/其它链路上，换 attention 核本身不是瓶颈。
- T=0 400-token 贪心：splitkv 基线 sha `33bcb427…`、HND `18d075b4…`，臂内 2/2 稳定、文本连贯
  （`logs/t0_base.log`、`logs/t0_hnd.log`）。换 paged 核与产线 custom paged 逐位一致、
  与 splitkv 不同，这是允许的精度门口径。
- 精度门（bshd 臂，`he_probe.py 9140 bshd 1`）：FAILS 7/9、CONTROLS 3/3，只挂
  #47/#145（bf16 基线 3 轮 20/27、8/9）；#47 是题面陷阱、#145 基线也 0/3。
  日志 `logs/he_bshd.log`。

## 未验证

- 非 handoff（`HANDOFF=0`）分支未做硬件 A/B。
- EAGLE topk>1 / deterministic inference 下 rejection sampling 保持关闭（未测）。
- `SGLANG_ENABLE_PP_SPEC`（官方 SGLang 树的 `sglang/srt/environ.py`，非 plugin 开关）的 stable-rows 分支未验证。
- 多卡 / TP>1 未验证。
- B+ 每步共享索引（2026-11-15）：c40 graph bench（40 桶全部捕获）40/40 且 sha 逐位相同，剩余未验证为非 handoff 分支的硬件 A/B 与 PP/TP 场景；窗口脚本在 `clean-stack/gdnk_window.sh`（仓外）。
- `SGLANG_MTP_LMHEAD_FP8`：0.86 以下验证通过且精度门通过，但**尚未切到常驻配置**；
  0.88 下会 OOM，切换前需重跑吞吐与精度门。
