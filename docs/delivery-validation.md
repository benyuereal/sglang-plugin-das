# 交付验证记录：版本、范围与证据

> 命名说明：本页历史产物保留交付时的 `sglang_dcu_plugin` / `dcu` 名称、版本及 SHA256；当前插件使用 `hcu_sglang` / `hcu`，没有对历史 wheel 重新构建或改名。


> **本文用途**：记录每次交付实际改变了什么、做了哪些检查、结果适用于什么环境。用于验收和交接，不代替 [CI 规则](ci.md) 或 [验证方法](validation.md)。历史结果保持原版本含义。

## 1. V2 文档与架构修订（2026-10-21）

输入需求为 `agent-work/plugin-v2-1021.txt`；修改基线为本仓 `331f10c`。参考 adapter `dac7de1` 的治理/软件分层与 plugin-das `ce870e2` 的接入说明。

| 交付范围 | 内容 |
|---|---|
| 架构 | 两仓独立与组合运行、加载/fallback 契约、现有与未来目录、协作闭环 |
| 文档 | README 扩充；文档中心/用途/导航；配置与协作专篇；操作和验证手册重整 |
| skill 与模板 | 四个任务入口和参考同步组合契约，验证模板补目标所有权与运行组合 |
| 图文导览 | 根 architecture.html，离线可读的 SVG 关系图、目录与流程卡片 |
| 实现边界 | 不改工具、测试、门禁代码；不增加运行时业务包或外部服务 |

### V2 实际检查

| 检查 | 环境与结果 |
|---|---|
| L0 | 本地 Python 3.11、历史验证节点 Python 3.10.12 均 passed；10 个 Python 文件、4 个 skill |
| 文档链接与标题锚点 | 235 个文件链接、15 个片段锚点检查通过，含 HTML 导览链接 |
| Mermaid | 11 段图使用 Mermaid 11 解析通过 |
| skill 元信息 | skill-creator quick_validate：4/4 passed |
| 浏览器导览 | 实际打开检查关系图、目录/协作卡片和页内导航，正常显示 |
| 工具回归 | 历史验证节点 Python 3.10.12：14 passed |
| 变更范围 | tools/ci/tests、.gitignore 与两个 V1 来源 JSON 的字节内容保持基线不变 |

远端复核使用已有 历史工具验证容器 容器、源码所属用户 与 `PYTHONDONTWRITEBYTECODE=1`，只运行 CPU gate/临时 Git fixture。本次未启动模型、调用 GPU 或测试真实 adapter/plugin 运行组合；图文描述的是设计契约，不是新运行时支持结论。

本地执行日志、链接与图语法检查汇总及远端结果留在 `artifacts/v2-20261021/`（Git 忽略）。Mermaid/DOM/锚点检查依赖仅位于维护者临时环境，没有加入本库执行依赖。旧版来源 JSON 保留快照，不改写捕获日期。

V2 修订基于 `331f10c`，包含 26 个已有文件修改和 4 个新文件；于 2026-10-22 按用户要求创建 Git 提交，未推送。

## 2. V1 方法与工具抽取（2026-10-20）
日期：2026-10-20。对象为方法库与工具契约，不是 plugin 运行时。完整测试日志位于当前 checkout 的 `artifacts/delivery-20261020/`（该目录已 Git 忽略，保留作本地执行证据）。

| 检查 | 环境 | 结果 |
|---|---|---|
| L0：10 个 Python AST、4 个 skill 与资源/文件链接、业务代码排除 | 本地 Python 3.11；历史验证节点 Python 3.10.12 | passed |
| 工具行为测试 | 本地 Python 3.11；远端 Python 3.10.12 | 14 passed |
| skill-creator quick_validate | 本地隔离校验环境，PyYAML | 4/4 passed |
| agents/openai.yaml YAML 与描述长度 | 本地隔离校验环境 | 4/4 passed |
| 工具 --help | 本地 Python 3.11；远端 Python 3.10.12 | 7/7 passed |

远端使用历史工具验证容器，以源码所属用户执行，设置 `PYTHONDONTWRITEBYTECODE=1`，从 `${REPO}` 读取交付树；测试仅在临时目录创建小型 Git fixture，未启动模型或调用 GPU。节点默认 Python 3.6 不满足工具要求。

14 项测试覆盖：正常回溯与 provenance、现有未知输出保护、合法生成输出显式覆盖、脏输入拒绝、anchor 漂移时不产生输出、输入输出父/子目录重叠保护、SHA256 修改/缺失、清单路径越界、校验 CLI 失败码、审计保留未跟踪文件及解析错误、merge-base 清单包含继承硬件漂移和 WORKTREE、AST/runtime-export 不误晋级 implemented、脏删除符号记录、official-ref 与 worktree HEAD 不一致时拒绝。

本次没有执行外部 plugin 的真实注册/目标解析/签名兼容、DeepSeek 方法同步、模型服务、GPU 数值/graph/TP 或性能 lane。这些验证属于后续实现任务，不继承历史 214 项 plugin 测试或 MiniMax TP4 数字作为本次结果。`--help` 成功不证明外部依赖可用。

## vLLM 发版 skill 方法迁入（2026-10-22）

范围：将用户指定 `upgrading-vllm-hcu-upstream-first` 的适用方法重写进本库；详情见 [迁移分析](vllm-skill-migration.md)。新增 [升级发版 skill](../skills/upgrading-sglang-hcu-upstream-first/SKILL.md)、5 份按需参考、UI 元信息、43 个来源文件的哈希/取舍快照，以及 [升级记录模板](../templates/upgrade-release-record.md)；四个既有 skill 与文档/导览补充入口。

来源处置：17 个文件抽取方法，1 个元信息文件重写，3 个 vLLM 专属脚本/测试不迁移，22 个历史/专题材料保留在来源。文件枚举不代表重验历史记录；未复制运行时、固定 vLLM 配置、历史成绩或私有 CI。

| 本次检查 | 实际环境与结果 |
|---|---|
| L0 | 本地 Python 3.11.16，13 个 Python 文件、5 个 skill，passed |
| 现有工具测试 | 本地 Python 3.11.16，22 项全部通过；本次未修改工具实现 |
| skill frontmatter/名称/资源 | skill-creator quick_validate，5 个 skill 全部通过 |
| Markdown/HTML 文件链接、锚点、Mermaid | JSDOM + github-slugger + Mermaid 11 检查通过；未改原图/布局 |
| 来源快照与目的地 | 43 个源文件 SHA256 与目的地存在性检查通过；V1 来源 JSON 保留原字节 |

本次是 skill/方法交付，不包含新的模型/GPU、wheel/image 发版或外部发布验证；前述 V1/V2 与独立 V3 的运行成绩不属于本次执行。本库原有未提交 V3 修改保留。

## V3 KV 修复阶段记录（6984176，后续候选见下一节）

输入需求为 `agent-work/plugin-v2-1022.txt`（内容定义 V3）。方法库保持运行时外置，具体步骤与问题见 [V3 实践](v3-runtime-validation.md)。

| 需求 | 当前交付与证据 | 状态 |
|---|---|---|
| 1. 用给定社区/HCU 输入完善 plugin | `output-code/sglang-plugin`，干净提交 `69841761fcda60931685e7e4ce835d4250222726`；0.3.0/dcu，206 目标与 3 个具名 seam | 代码与 CPU 接线已交付，完整模型仍需后续验收 |
| 2. 两输入回溯为 sglang-das | `output-code/sglang-das`，generator v2、440 vendored、446 文件 SHA256；来源 official baa9befa38 + plugin 6984176 | passed（生成/来源/哈希与隔离接线） |
| 3. 指定容器空闲 GPU 测两份代码 | 修复前外部/内嵌 GPU 各 3 passed；最终 6984176 两份 CPU 各 231 passed/3 GPU skipped，包含真实 KV pool 构造 | 最终 GPU/完整服务 pending；之前 GPU 结果按其原提交范围保留 |
| 4. 原 d.sh 服务 | 首次完成权重加载后 KV descriptor ValueError、未 ready，配方退出 137；6984176 已修复；attempt2 准入因 GPU 忙退出 75，未重复加载 | pending：需要完整 8 卡空闲窗口重跑原配方 |
| 5. 反哺 skill/docs | lazy bootstrap、严格 loader、msgspec/具名 seam、完整 native resources、同步预检、KV 真实构造与服务恢复方法 | 本轮已更新，后续真实模型问题继续迭代 |

### 6984176 阶段代码验证

证据相对方法库 `artifacts/v3-20261022/`；这些文件 ignored，长期留存须另保存。测试环境为 历史验证容器 Python 3.10.12、torch 2.11 DTK。

| 检查 | 实际结果与证据文件 |
|---|---|
| 方法库工具回归 | `method-final-tests.log`：指定容器 Python 3.10，22 项通过；L0/文档检查另执行 |
| 外部 CPU 全套 | `plugin-kvfix-cpu-tests.log`：231 passed、3 skipped、18 warnings |
| 回溯 CPU 全套 | `embedded-kvfix-cpu-tests.log`：231 passed、3 skipped、18 warnings；importlib 模式防外部源码注入 |
| 外部最终接线 | `plugin-runtime-final.json`：206 registered/applied、3 具名 seam、errors 空 |
| 内嵌最终接线 | `embedded-runtime-final.json`：禁用全部外部 entry points，206 applied，框架和插件来源均为输出 python |
| 回溯保护与哈希 | 替换前旧清单 446 哈希通过，全部 Git 变化属于允许生成集合；重建后 446 哈希通过 |
| wheel | `wheels/6984176/sglang_dcu_plugin-0.3.0-py3-none-any.whl`，SHA256 `c41b4d2ef01759b814d257060ccc515fb316193804a009f15eed5a6b036a260d` |
| wheel 内容 | `wheel-kvfix-resources.json`：283 native + 1 许可资源哈希通过，KV 修复源码与工作树一致 |
| 输入与原配方 | official baa9befa38、HCU 5f67ce01c3 与 plugin 6984176 均干净；d.sh SHA256 `127a6a1f2822a1087126ce45a3c973f285f408b0ced00cad9e1df023ea8c83d4` 未改 |

首次失败保留 `plugin-service-result.json`、`plugin-d-sh-service.log` 和 controller log；下一次结果独立留在 `plugin-kvfix-attempt2/`。不能把退出码 137 单独解释为 OOM；已发现的 rank 异常由真实日志定位。当前 8 卡各约 116000–118000 MiB 驻留，不按 0% 利用率判空闲，不停止其他服务。pending 不更新为完整模型 last-good，也不宣称全 fork/全模型等价；初始 1802 pending 台账仍为审计起点。

## V3 Graph/RMS 候选阶段（57f4f84，2026-10-22）

该阶段修复 Graph keyword 和 RMSNorm quant_linear。外部、回溯、隔离 wheel 各 246 CPU passed/3 skipped，各 206 targets applied；回溯 440 vendored/446 哈希通过。wheel SHA256 `0b76eddf69fd4a87f27339169cf81e3265138040eef8a0a369beb39523f67f70`，全包 440 文件及 usercustomize 与提交一致。

资源恢复后 `final-window-contractfix/plugin-gpu.log` 与 `embedded-gpu.log` 各 3 passed。随后 `plugin-contractfix-attempt4/` 的原始服务在 Graph 捕获前向出现 hcu_skip_layernorm 分发错误，结果 failed、cleanup_remaining 空、own_service_cleaned_up true；该等待工作流已终止失败。历史失败与产物保留，不能作为完整模型 last-good。静态 AROUND 审查 `around-contract-review-57f4f84.json` 有 9 个目标/13 种调用形态待审查，涉及 DSA/DeepSeek V4 等，不以 bind/apply 推断全模型兼容。

## V3 StageBoundary 候选阶段（e87f911，2026-10-22）

外部插件提交 `e87f911e2d5cbdf76e09ce4a33a63a8eafc14def`，在公共 StageBoundary.prepare 消费 fused norm handoff 参数；保持每次调用的 plan 副本和社区残差执行链。两份输入、原始 d.sh 不变。范围为指定输入上的插件完善与请求的 MiniMax 配方，不声明全模型/全 fork 等价。

| 原需求 | 当前证据 | 完成边界 |
|---|---|---|
| 1. 完善插件并输出 | output-code/sglang-plugin，e87f911；254 CPU passed/3 GPU skipped，206 targets applied/3 具名 seam | 代码已交付；完整模型服务待续验 |
| 2. 回溯仓库 | official baa9befa38 + plugin e87f911；440 vendored/446 哈希通过；隔离 206 applied、CPU 254 passed/3 skipped | 生成、接线、CPU 通过 |
| 3. 指定容器 GPU 测试 | e87f911 外部/内嵌各 3 passed，使用实际空闲 card 0，详见 single-gpu-boundaryfix-e87f911/ | 当前候选两个形态 GPU 用例通过；不替代完整模型服务 |
| 4. 服务验收（用户授权 TP4） | 基于原 d.sh 派生 TP4，只改 TP 数及四卡 NUMA；e87f911 已启动 plugin-tp4-attempt1 | 仍需 ready、真实生成和清理；不宣称原 TP8 通过 |
| 5. 方法迭代 | V3 实践 §9 与 test-contract 记录真实公共分发、有效 residual/entry 链及 per-call plan | 本轮完成，实际服务后继续补结论 |

证据相对 artifacts/v3-20261022/：

| 当前候选检查 | 结果与文件 |
|---|---|
| 外部完整 CPU | plugin-boundaryfix-cpu-tests.log：254 passed/3 skipped，18 warnings |
| 回溯完整 CPU | embedded-boundaryfix-cpu-tests.log：254 passed/3 skipped，18 warnings；importlib 隔离 |
| 两形态严格接线 | plugin-boundaryfix-runtime.json / embedded-boundaryfix-runtime.json：各 206 registered/applied、errors 空；内嵌禁用外部 entry points |
| 回溯哈希 | reconstruction-boundaryfix-verify.json：446 checked/passed；旧清单哈希和 Git 变化集合已核验后替换 |
| 两形态 GPU | single-gpu-boundaryfix-e87f911/plugin-gpu.log、embedded-gpu.log：各 3 passed/4 warnings；state.json 记录 e87f911、exit 0、card 0 和两个 visibility 为 0 |
| wheel 内容 | wheel-boundaryfix-content.json：440 tracked package 文件及 usercustomize 与当前提交一致；SHA256 `45bb39d1fa56b9bc28cca5d2f78040009c277638f7669b9b156e9db8f84e3668` |
| wheel 隔离检查 | wheel-boundaryfix-runtime.json：206 applied、来源为该轮隔离安装目录；wheel-boundaryfix-cpu-tests.log：254 passed/3 skipped、18 warnings。 |

此前 2026-10-23（北京时间）曾调整准入顺序（以下等待记录已被用户 TP4 授权替代）：原 wait_boundaryfix_validation.py 经实际进程句柄核对、暂停并确认仍为 waiting_for_idle 且无 active_pid 后才终止，状态为 superseded_before_service，原服务尚未启动。新 validate_boundaryfix_idle_card.py 先串行执行两份单卡 GPU 用例，再等完整 TP8；状态 final-window-boundaryfix-services/state.json。原截止时间不延长：2026-10-22 16:18:32 UTC / 2026-10-23 00:18:32 北京时间，初始窗口共 1800 秒。续接核对新实际 PID/原工具会话；第 5 次外部服务目录 plugin-boundaryfix-attempt5，通过清理后执行 embedded-boundaryfix-attempt1。资源不足记 pending_busy；不停止其他服务，不凭旧状态文件重复拉起服务。原配方 SHA256 `127a6a1f2822a1087126ce45a3c973f285f408b0ced00cad9e1df023ea8c83d4`。


### 用户授权 TP4 后的当前服务配置

用户明确允许 TP4 验证，需求 4 按该更新接受条件执行。原始脚本不改，派生 minimax-tp4-d.sh 仅修改 TP 8→4、NUMA 列表八项→0 0 1 1；其他配置不改。源 SHA256 `127a6a1f2822a1087126ce45a3c973f285f408b0ced00cad9e1df023ea8c83d4`，派生 SHA256 `9d6bdd367d316606b81ffdddcbdcab6d601caa01bb2088d9ad1b5bca9d7396a7`，详见 minimax-tp4-recipe-manifest.json 和 V3 实践 §10。

validate_minimax_tp4.py 在指定容器中固定卡 0–3，记录并核对可见 GPU 数为 4。TP8 等待器确认尚未启动服务后结束，状态 superseded_by_user_tp4；当前流程状态 tp4-validation-e87f911/state.json。首轮 plugin-tp4-attempt1 在 Graph 捕获的 RoPE dtype 契约处失败并已清理。后续修复与当前候选见下一节，不把该轮记为 passed。


## V3 RoPE 候选阶段（62b5b7f，2026-10-23）

当前插件 commit 62b5b7f2b8cc818115ad4d152a6239342f2ef91d，原输入未改。用户已明确授权将需求 4 的服务验证改为 TP4；原始 TP8 不是 last-good。沿用 TP4 派生配方及原/派生 SHA256，仅调整 TP 和对应 NUMA；其余模型/FP8/FA3/Graph 参数保持。

| 原需求 | 当前证据 | 完成边界 |
|---|---|---|
| 1. 完善插件 | output-code/sglang-plugin，62b5b7f；263 CPU passed/6 skipped，207 registered/applied、3 具名 seam | 完整 TP4 服务仍在续验 |
| 2. 回溯输出 | official baa9befa38 + plugin 62b5b7f；441 vendored/447 哈希通过，隔离接线和 CPU 通过 | 源码生成完成；完整内嵌服务单独验证 |
| 3. 指定容器 GPU | 外部/回溯各 6 passed、exit 0，card 0 | 算子正确性及变输入 replay 通过，不外推全模型 |
| 4. 授权 TP4 服务 | plugin-tp4-attempt1 失败清理后修复 RoPE cache 契约；plugin-tp4-attempt2 启动中 | 必须 ready、三次生成与清理，不称原 TP8 已通过 |
| 5. 方法迭代 | V3 实践 §10/§11、test-contract 与本记录 | 本轮问题和验证边界已反哺，实际服务后补结论 |

证据相对 artifacts/v3-20261022/：

| 检查 | 文件与结果 |
|---|---|
| 外部完整 CPU | plugin-rotaryfix-cpu-tests.log：263 passed/6 skipped，18 warnings |
| 内嵌完整 CPU | tp4-validation-rotaryfix/embedded-cpu.log：263 passed/6 skipped，18 warnings，importlib 隔离 |
| wheel 完整 CPU | wheel-rotaryfix-cpu-tests.log：263 passed/6 skipped，18 warnings，来自该轮隔离安装目录 |
| 三形态严格接线 | plugin-rotaryfix-runtime.json、embedded-rotaryfix-runtime.json、wheel-rotaryfix-runtime.json：各 207 applied、errors 空；内嵌屏蔽外部 entry points |
| 两形态 GPU | tp4-validation-rotaryfix/plugin-gpu.log、embedded-gpu.log：各 6 passed/4 warnings、exit 0，显式 card 0 与两个 visibility=0 |
| 回溯保护/哈希 | reconstruction-rotaryfix-verify.json：447 checked/passed；旧 446 哈希与全部 Git 变化集合先核验 |
| wheel 内容 | wheel-rotaryfix-content.json：441 tracked package 文件及 usercustomize 字节一致；wheels/62b5b7f/sglang_dcu_plugin-0.3.0-py3-none-any.whl，SHA256 8c2f533e7454b6bf130bc8cfb8135461fa60500b3fdcf287d3cf84bf69febaa6 |
| RoPE 定向回归 | rotaryfix-targeted-cpu.log：10 passed；rotaryfix-native-gpu.log：真实算子/Graph 3 passed |

服务工作流 validate_minimax_tp4_rotaryfix.py，状态 tp4-validation-rotaryfix/state.json。仅使用卡 0–3，controller 核对实际可见 GPU=4；外部 plugin-tp4-attempt2 通过并清理后，内嵌 embedded-tp4-attempt1 才启动。续接须核对当前实际 PID/工具会话，不根据旧 JSON 重复启动。原 TP8 等待流程已由用户 TP4 授权替代，不能继续把旧截止时间当成当前服务状态。


## V3 KV 视图候选阶段（309ddfa，2026-10-23）

当前 commit 309ddfadd32e14a80cb9dc60b44f0661d90604e1，修复 HCU pool→社区 FA getter→插件 interface 的物理布局交接。第二次 TP4 在 RoPE 后遇到 V 视图 shape assertion，plugin-tp4-attempt2 保留 failed，清理完成、remaining 空。当前第三次服务按用户授权的同一 TP4 配方执行，不改其他参数，不宣称原始 TP8 通过。

| 原需求/门禁 | 当前证据与状态 |
|---|---|
| 1. 完善插件 | output-code/sglang-plugin，309ddfa 干净输入；267 CPU passed/8 skipped，208 targets applied/3 具名 seam；完整模型服务待续验 |
| 2. 回溯输出 | official baa9befa38 + plugin 309ddfa；441 vendored/447 哈希通过，内嵌隔离接线与完整 CPU passed |
| 3. 指定容器 GPU | tp4-validation-fa-viewfix/plugin-gpu.log、embedded-gpu.log，各 8 passed/18 warnings、exit 0，显式空闲 card 0；只证明实际算子/Graph lane |
| 4. 用户授权 TP4 服务 | plugin-tp4-attempt3 启动中，通过并清理后运行 embedded-tp4-attempt1；仍需 owned health、真实模型名、三次请求及清理 |
| 5. 方法迭代 | V3 实践 §12、test-contract、当前记录与任务汇总已补齐布局消费者链和 FP8 基线边界 |
| 三形态完整 CPU | plugin-fa-viewfix-cpu-tests.log、tp4-validation-fa-viewfix/embedded-cpu.log、wheel-fa-viewfix-cpu-tests.log；各 267 passed/8 skipped、18 warnings |
| 三形态严格接线 | plugin-fa-viewfix-runtime.json、embedded-fa-viewfix-runtime.json、wheel-fa-viewfix-runtime.json，各 208 applied、errors 空；内嵌禁用外部 entry points |
| 回溯保护 | reconstruction-fa-viewfix-verify.json：447 checked/passed；替换前旧 447 哈希与 Git 变化集合已验证 |
| wheel 内容 | wheel-fa-viewfix-content.json：441 tracked package 文件及 usercustomize 字节一致，SHA256 0e3be4becef2a8aefe02a5ce8e723dc8df05347275f69c6431df7c82c24272ab |
| 定向回归 | fa-viewfix-targeted-cpu.log：10 passed；fa-viewfix-native-equivalence-gpu.log：2 passed，BF16/FP8 原生对照与 replay |

当前服务工作流 validate_minimax_tp4_faviewfix.py，状态 tp4-validation-fa-viewfix/state.json。固定卡 0–3并核对可见 GPU=4，派生 recipe SHA256 9d6bdd367d316606b81ffdddcbdcab6d601caa01bb2088d9ad1b5bca9d7396a7；原始 recipe SHA256 127a6a1f2822a1087126ce45a3c973f285f408b0ced00cad9e1df023ea8c83d4 保持。续接时核对新实际 PID/工具会话，不重用任何旧失败目录或等待器状态。当前未记录完整服务 last-good。


## V3 当前候选与 TP4 续验（0688960，2026-10-23）

当前 commit 0688960dec7db9f2ca416e22cd48690f2e62d68b，修复 NormQuantReadout 三参数 norm 交接。第三次 TP4 已通过 FA3 后因 HcuSkipNorm 第三个位置参数失败并清理。第四次仍用同一用户授权 TP4 派生配方，原/派生脚本及 SHA 不变，不能解释为原 TP8 已通过。

| 需求/门禁 | 当前结果 |
|---|---|
| 插件输出 | output-code/sglang-plugin，0688960 干净输入；plugin-readoutfix-cpu-tests.log：276 passed/8 skipped，18 warnings |
| 回溯输出 | official baa9befa38 + plugin 0688960；441 vendored/447 哈希通过；tp4-validation-readoutfix/embedded-cpu.log：276 passed/8 skipped，18 warnings |
| 严格接线 | plugin-readoutfix-runtime.json、embedded-readoutfix-runtime.json：208 applied、3 具名 seam、errors 空；内嵌禁用外部 entry points |
| GPU | tp4-validation-readoutfix/plugin-gpu.log、embedded-gpu.log：各 8 passed、exit 0，card 0 |
| 授权 TP4 服务 | 第四次四个 rank 完成全部 20 个 decode Graph 后，Cargo 1.75 构建失败并清理；Rust 1.92 两形态预编译已通过。第五次模型启动前 pending_busy；本轮有时限等待已到期，未启动第六次 |
| 方法迭代 | V3 实践 §13、test-contract 补真实 readout 变体、第三参数及额外张量语义 |
| 当前 wheel | wheels/0688960/sglang_dcu_plugin-0.3.0-py3-none-any.whl；441 包文件及 usercustomize 与源码一致，SHA256 b69c19066b99ed5437f21ee3c614df2f5715574b842271e0062b302ea83c4de5 |
| wheel 隔离门禁 | wheel-readoutfix-isolated-runtime.json：208 applied、plugin 来自该轮隔离安装目录；wheel-readoutfix-isolated-cpu-tests.log：276 passed/8 skipped、18 warnings |

完整服务仍未验收，不能将上述 CPU/GPU 门禁当作服务 ready/输出正确的证明。第四次的 owned health、三次实际请求、完整响应和清理结果须独立保留。


### 当前环境就绪与资源等待

源代码、回溯、隔离 wheel 当前各 276 CPU passed/8 skipped，两个运行形态各 8 GPU passed。Rust 1.92 的选择链、Python 别名环境与真实 bindings 预编译见 V3 实践 §14、rust192-runtime-manifest.json、两份 rust192-*-alias-prebuild.log。整个构建保留源码/lock，未切换 Rust tree cache 实现，也未改模型配方。

wheel-readoutfix-runtime.json 是初次隔离助手错误引用外部源码时的 failed 记录，门禁正确识别路径污染；助手已修正并保留备份。有效结果只使用带 isolated 的两份文件，不能把旧 failed 报告改写成通过。

本轮服务队列 validate_minimax_tp4_rust192_window.py，状态 tp4-validation-rust192-window/state.json；截止北京时间 2026-10-23 02:11:59。plugin-tp4-attempt5 的 pending_busy 未分配模型；下一次 plugin-tp4-attempt6 准入后执行，成功清理后再做 embedded-tp4-attempt1。完整服务仍为 pending，20 个 decode Graph 不等于 ready 或生成正确。续接核对实际 PID/工具会话，不能凭旧状态重复拉起。


### 02:12 窗口结束状态

北京时间 2026-10-23 02:12:01，队列以 pending_busy/exit 75 结束，实际 PID 不存在，未创建 plugin-tp4-attempt6，内嵌服务也未启动。全部 8 卡仅 card2/card3 满足空闲准入，目标容器无存活模型 worker。当前没有后台复验任务；源代码、CPU/GPU、wheel 和 Rust 预编译结果保持，健康与真实生成仍待四卡可用后继续。


### 后四卡续验窗口（10:34）

用户明确指定后四卡 TP4。当前物理 4–7、NUMA 2 2 3 3，派生脚本 SHA256 bb0f42be5018ffa7eec6944d447c947c0af540b28cffedadbeeb89d9904f6982；其余参数及运行时 0688960 不改。10:30 首次准入 pending_busy，尚未加载模型；10:34:17 启动新的 30 分钟准入窗口，实际 PID 1113315 已核对，截止 11:04:17 北京时间。状态 tp4-validation-last4-rust192-window/state.json；外部 plugin-tp4-last4-attempt2 通过并清理后再运行 embedded-tp4-last4-attempt1。完整服务仍待验收，旧 02:12 窗口已结束的事实保留；配置与证据见 V3 实践 §15。


### 后四卡本轮终态与停止权限阻塞（10:51）

| 项目 | 实际结果 |
|---|---|
| 外部 TP4 | plugin-tp4-last4-attempt2：权重加载完成后，KV cache 可用显存检查失败，未 health/生成；脚本退出 137 的具体根因在 scheduler ValueError 日志 |
| 并发占用 | 其他用户的容器（完整身份见原始证据）；四个外部 PID/cgroup 与后四卡 KFD 显存关联，每卡停止前 77258903552 字节 |
| 用户指令 | 明确允许停掉占后四卡的其他容器，已执行 docker stop --time 30 |
| 停止结果 | 节点权限策略拒绝，exit 1：当前执行用户无权停止其他用户创建的容器；inspect 仍 running，未宣称释放成功 |
| 自有清理 | own_service_cleaned_up=true，remaining=[]；目标容器仅基础 bash，30001 空闲，窗口 PID 已消失 |
| 内嵌服务 | 未启动，不能记作通过 |
| 当前产物 | 0688960、三形态 CPU 276/8 skipped、两形态 GPU 8 passed、208 targets 与 447 哈希证据保持；无新运行时代码变更 |
| 下一步 | 由创建者/管理员停止冲突容器或保证后四卡稳定空闲，再以新 attempt 继续同一 TP4 配方 |

当前无后台验证任务，完整目标未完成。停止操作的精确证据见 last4-conflicting-container-stop-20261023.json，行为与经验已反哺 V3 实践 §16 和 test-contract。原始 TP8 仍未通过，不改配置以迎合空闲显存。


### 12:47 续验：服务启动通过，实际生成失败

后四卡空闲后，新流程 tp4-validation-last4-resume-1247 已运行 plugin-tp4-last4-attempt3。当前 0688960 配方完成权重/KV/Graph/服务初始化，owned health=200，真实模型名正确；实际三份响应为乱码/length=512，最终 failed，清理完成。旧算术数字匹配过宽的误判已由独立语义复核否定；严格新控制器对三份实际错误响应全部拒绝。未启动内嵌服务，不记为模型正常性通过。

仅关闭 Graph 的诊断对照同样 health=200 后生成失败并清理，不能作为最终配方通过。Attention、实际形状 FP8 MoE 与 Graph 的进一步对照见 V3 实践 §17；当前没有依据这些小样本宣称整个模型正确。线性量化探针已启动，SSH 中断后其终态待恢复连接确认。当前仍需定位乱码、修复、重新生成/验收受影响产物及完整两形态 TP4 服务。


13:18 连接恢复，原线性探针终态与四个真实形状输出已核对，relative L2 约 0.00165。目标容器无遗留模型进程；仍需定位完整模型乱码，详见实践 §17。


## V3 当前结果（45e67e0，2026-10-23）

重复归约错误已由真实 FFN 边界回归复现并修复，当前 commit 45e67e0bfe2d64b79f15398b4c0b61e46ae63584。三形态完整 CPU 各 280 passed/8 skipped、严格 loader 各 208 applied；两源码 GPU 各 8 passed。旧清单受保护后重新生成，441 vendored/447 哈希通过，新 wheel 441 包文件及 usercustomize 字节一致，SHA256 876aeb6f24aad056855e3a3ee55c314c8033c18d22a9e3501182e0952184b1d4。

外部 plugin-tp4-last4-attempt4 已在原授权 TP4 后四卡配方下 health=200，并通过三次实际生成：最终回答 2、2、Paris，finish_reason=stop，清理完成、remaining=[]。重构版首轮切换时仅因端口准入 pending_busy，未启动模型；端口恢复可绑定后以新目录 embedded-tp4-last4-attempt2 复验通过，同样 health=200、三次正常 stop，最终回答 2/2/Paris，清理完成、remaining=[]。当前两份完整服务证据均已齐备；详细合同、命令和验证边界见 V3 实践 §18。


当前五项交付的最终证据索引为 requirements-audit-final-45e67e0.json。外层切换工作流的 pending_busy 记录保留；后续内嵌独立运行的 passed 原始结果单独记录，不改写旧状态。测试服务均已清理，目标容器仅基础 bash、30001 空闲。方法库文档/skill 已更新，工作区原有未提交变更保留，未自动提交或推送方法仓。

## V3 开发方法论整理（2026-10-23）

本轮把已完成的 V3 开发与修复记录整理为可复用流程，没有改动外部运行时实现或重新启动 GPU 模型。固定历史模型证据仍属于对应的 45e67e0 运行，不能当成本轮重跑。

| 改动 | 结果与边界 |
|---|---|
| 方法入口 | 新增 [开发方法论](development-methodology.md)：接受配置→实际消费者/归约 owner→修复前反例→原配置复验→产物/证据闭环；按影响面选测，不固化 TP4、后四卡或 Rust 1.92 |
| 现有 5 个 skill | 开发、排障、迁移、回溯、升级发版入口连接到方法论；开发测试与排障参考按决策重整，保留权限、共享资源、实际子进程和未覆盖模式的边界 |
| 可执行生成验收 | 新增 [离线工具](../tools/validate_generation_response.py)，完整答案匹配＋正常 stop＋显式思考协议＋可选模型身份＋原响应哈希；不替代 HTTP/监听者、开放题、数据集、cache hit 或清理证明 |
| CPU 行为回归 | 16 个新增用例覆盖原误判类型、协议、截断、损坏输入与原文件保护；和现有工具合计 38 项，候选目录在指定容器 Python 3.10 全部通过 |
| 真实记录回放 | 新工具接受 V3 的六份正确响应，拒绝三份原始乱码响应，九份原文件 SHA256 前后不变。报告在 artifacts/methodology-20261023/real-response-replay.json；这是历史响应的离线重验，未宣称重跑模型 |
| 文档与导览 | 更新验证记录模板、导航、CI/工具边界与 9 工具清单；HTML 桌面/手机渲染无脚本错误、无页面横向溢出，修复代码卡片的最小宽度撑开问题 |

本轮证据放 `artifacts/methodology-20261023/`，包含候选/部署检查、真实响应回放及变更哈希清单。V1 来源 JSON、旧 V3 原始失败/成功报告和 vLLM skill 来源快照保留原义；未自动提交或推送仓库。


## 新人、安装与排障入口重整（2026-10-23）

本次按使用者和开发者任务重写入口；方法库继续只维护 skill、文档与方法工具，没有引入 plugin 包、entry point 或打包定义。保留已有未提交工作，没有提交/推送。

| 改动 | 现在如何使用 |
|---|---|
| 根 README / 文档中心 / 阅读指南 | 首屏选安装、首次开发、已有问题三条路线；五个 skill 按场景选一个主入口，并提供输入、产出与可复制请求 |
| [安装手册](installation.md) | 明确本库与运行时仓；wheel / editable 隔离安装、metadata、HCU 访问、严格加载、模型验收与停用分开 |
| [最小案例](first-plugin.md) | 文档代码块创建临时独立 demo 包，真实 pinned HookRegistry 接通 add_prefix；实跑文件在容器 /tmp，不纳入本方法库 |
| [排障入口](troubleshooting.md) | 按症状找首个失败层；TP4 乱码案例展示 triage→最小复现→dev 修复→原配置复验 |
| [经验索引](development-methodology.md) | 保留旧路径，将此前总流程定位改成历史症状、查法、修复经验和边界，后续遇到相似问题再查 |
| skill / AGENTS / 静态导览 | 同步任务路由，明确不要求依次执行全部 skill；业务代码继续留在外部实现仓 |

本次检查：方法库 L0、38 项 CPU 工具测试、5 个 skill 元信息、422 个文件链接、23 个锚点、10 段 Mermaid 通过；静态导览 1440 / 390 宽度渲染通过。最小案例三个真实启停对照通过，保留原返回值与异常。

实际安装验证使用原容器与运行时 45e67e0，在临时 target 分别完成 wheel / editable 安装和 metadata 检查；严格加载报告各为 208 registered/applied、三个具名 seam、errors 为空。本次发现安装 UID 与 HCU 运行用户能力不同，以及 AITER 独立缓存目录要求，已写入安装手册。冷缓存尝试在原有 180 秒外层限制下退出 124，保留日志与已产出的报告；后继热缓存命令结果单独记录，不覆盖旧尝试，也不把冷缓存外层超时写成整条命令成功。

证据位于 `artifacts/onboarding-20261023/`，方法库部署/页面检查与源哈希见其中 `review/`。本次没有启动模型服务、执行整模型精度或性能验证；先前 V3 结果仍属于原历史记录。


## 文档术语规范化与 Skill 执行入口完善（2026-10-23）

本次将导航、标题和任务描述调整为技术文档表述，统一“任务类型、操作指南、对应 Skill、验收标准”。补充文档维护规范，明确操作文档与执行 Skill 的输入、调用、输出及职责边界。

新增 `hcu-plugin-scaffold`（插件工程初始化）和 `hcu-plugin-install`（插件安装与加载验证）两个 Skill，均包含 SKILL.md 与 Agent 元信息；未增加运行时代码或执行脚本。现有开发、诊断、迁移、升级与回溯 Skill 保持职责，并同步与新入口的衔接。

七项功能文档提供可复制调用示例；新增 `docs/upgrading.md`，将版本升级操作指南与 vLLM 方法来源分析区分。标题更新对应的内部锚点同步迁移，原路径保持不变。原有安装和初始化命令示例、全部 Python 工具/测试及 JSON 历史快照字节不变。

本次验证：L0 通过（15 个 Python 文件、7 个 Skill）；7 份 Skill 元信息校验通过；文件链接、标题锚点、Mermaid 解析及 1440 / 390 宽度静态导览检查通过。证据见 `artifacts/standardization-20261023/`。本次未重新执行模型、GPU、安装或工具行为测试；此前实跑结论保留历史范围。

本次仅更新方法库的文档、Skill 和静态导览；未提交或推送，原有工作区修改保持原状。

## V3 2026-10-26 需求最终交付（2026-10-27）

需求为 `agent-work/plugin-v3-1026.txt`。正式运行时输出为 `output-code/sglang-plugin`（v0.3.2.dev5，`afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`）和由 official/plugin 两输入生成的 `output-code/sglang-das`；源码、wheel 与内嵌产物仍在方法库之外。旧 v0.3.1 已保留回退位置。

两种运行时形态分别通过 435 项 CPU、37 项 GPU、363 项加载合同及四卡 native Graph 重放；各保留一个 CPU skip 和一个 GPU skip。独立 wheel 完成 435 项 CPU、入口/启动隔离及 363 项加载；未运行 wheel GPU/模型服务。两种形态的 MiniMax TP4 原配方派生服务均完成自有 health 200、三次最终答案 `2`、`2`、`Paris`、正常结束和清理。正式输出与测试提交/包内容相同，并另验模块来源。

模块清单、真实测试位置、所有日志、原配方差异、HCU 逻辑对照及未覆盖范围统一见 [最终实施与需求验收](hcu-sglang-development.md#2026-10-26-8-最终交付与需求验收)。完整 HCU fork 等价、TP8、性能和全模型数据集精度均未宣称通过。方法库保持现有 HEAD/暂存区，变更仅保存为工作区文档、Skill 与此前工具修正；没有推送或外部发布。

## 2026-10-27 方法库提交前检查

本轮整理七个任务 Skill、初始化/安装/诊断/升级操作指南、V3 方法经验和离线验收工具；方法同步保留源方法语义，回溯生成器版本 3 不额外写入说明注释或生成标记。根目录 NOTICE 的来源声明合并到[来源地图](extraction-map.md#来源与许可声明)，本地配置改为按[配置指南](configuration.md#1-当前工具的路径配置)直接填写 `.env`，不再引用已删除的模板。

| 检查 | 本次结果与范围 |
|---|---|
| `python3 ci/l0_gate.py` | 通过：16 个 Python 文件、7 个 Skill、相对文件链接及阶段边界 |
| `python3 -m unittest discover -s tests -v` | 48 项通过；在任务容器内使用 Python 3.10.12，覆盖离线工具与隔离 Git fixture |
| Skill `quick_validate.py` | 7 个入口通过；维护端 Python 3.12 环境 |
| Markdown 标题锚点 | 本次扫描无失效的本地 Markdown 锚点 |
| `git diff --check` | 通过 |

本段只记录方法库与离线工具检查，不新增模型精度、GPU/Graph 或性能通过声明。已有运行案例保留各自日期、版本与验证边界。生成仓库、原始日志和运行依赖不纳入本次方法库提交。

## V3 方法库导航与统一入口（2026-10-27）

新增 `hcu-plugin-assistant` 作为检索、任务匹配与必要输入补全入口；原 7 个专项 Skill 继续按任务执行。README 补全文件树、功能、开发/安装/诊断案例及直接使用方法工具的路线。网页加入离线阅读器、内部路由、全文检索和导航状态处理，由 `tools/build_guide.py` 与 `templates/architecture.html` 维护。

本轮 L0、53 项工具测试、8 个 Skill 元信息、网页快照及 DOM 交互检查通过。实际浏览器安全策略拒绝本地 file URL，未执行截图或真实布局验收，未以 DOM 状态代替视觉验证。未启动外部插件或模型服务。详细范围与证据见 [方法库入口实施记录](hcu-sglang-development.md#2026-10-27-v3-方法库导览与统一任务入口实施记录)。

## DAS 模块盘点与插件开发计划（2026-10-28）

新增 [研发迁移规划网页](../migration-plan.html)，按 26 个领域盘点三份固定源码，并给出 78 条待评审建议工作项。来源为 HCU `15c7204b`、官方 `baa9befa`、插件 `afc6205`；当前树比较清单 11,497 个路径，运行时来源含 4,714 个 Python 文件。合同目标、来源哈希和同字节 native 资源作为静态线索，不记为迁移完成率。

新增只读采集/渲染工具、保存的数据快照、可维护计划定义与网页模板。源码明细支持路径/顶层符号检索、关系筛选、详情和 CSV 导出；计划按阶段列出职责、依赖与验收标准。维护流程与口径见 [规划指南](sglang-das-migration-plan.md)。

本轮本地 60 项工具测试、L0、8 个 Skill 元信息与文档/路由检查通过；16 项规划页 DOM 检查包含全部 46,322 个 Python 顶层定义的源行位置交叉核对。浏览器通过仅提供新规划页的回环 HTTP 预览，已检查默认桌面及 390px 窄屏、模块详情、阶段与源码筛选及真实 CSV 导出，窄屏整页无横向溢出。节点采集复验、完整测试与来源保护结果以 `artifacts/migration-plan-20261028/` 最终报告为准。本轮未运行 GPU、模型或 Hook apply。


## V3 分阶段实现与当前包交付工具（2026-10-29）

本轮本地代码提交 `af5c635`、`42fe693`：迁移/审计/内嵌工具支持 `hcu_sglang`，保留旧包兼容；新增 HCU 标准分发的无效专家 ID 保留逻辑、当前目标合同及全差异执行台账。详细来源、命令与边界见 [开发迁移记录](hcu-sglang-development.md#6-2026-10-29-分阶段继续迁移)。未推送。

70 项离线测试、4 项真实公共 dispatcher CPU 张量回归、L0、3 个修改 Skill 元信息及导览检查通过。源码、隔离 wheel、禁用外部 entry point 的实际 embedded 均为 364/364 项 Hook 和 3 个具名 seam；wheel 527 项包文件与源码相同，embedded 532 项哈希通过。桌面和 390px 窄屏阅读器、导航开关已实际检查。

本记录是本轮阶段交付，**全量 DAS 迁移尚未完成**。DeepSeek V4 旧抽取 profile 因平台类拆分正确拒绝预检；完整 GPU 数值、Graph、多进程 EP 与模型服务未执行。未审核路径保留 pending，不能据此推断现有实现缺失或全部 fork 等价。


## 2026-11-05 模型侧阶段记录

本轮在插件 main 起始 `1eb4abaa3a3ad5bb7bbc9a8b758488639f4db1c7` 的工作树中修复 V4 public forward 的 `_forward_impl` 接线，新增 current_v4 同步 profile，并迁入 GLM5-next（含 vision、NextN、shared visual）和 Muse Spark text/omni 的完整 owned 源码及按需严格注册。DAS 来源固定 `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`、官方固定 `baa9befa381511cd4d56f83980951e5a45fffc6d`；未提交或推送。文件、对应表、命令与所有验证边界见 [V4 记录](hcu-sglang-development.md#2026-11-05-v4-平台类与方法同步模型侧) 和 [模型迁入记录](hcu-sglang-development.md#2026-11-05-glm5-next-与-muse-spark-owned-模型迁入)。

宿主 Python 3.10.19 全量 unittest 为 133 项、120 通过、13 skipped；current_v4 --check 两份 facade unchanged。容器 Python 3.10.12 禁 GPU 后新增 11 项隔离 CPU 定义测试通过，CUDA 未初始化；这不替代整模块导入或真实 collective。先前 V4 九项 CPU 和旧 598/598 loader 是各自快照证据，不升级为新增后的通过结论。

GLM 五个专用 runtime 模块与 Muse kernels/layers/config/processor 未齐，已审七个模型键保持 in_progress。最新禁 GPU 的 599-target 严格 loader 退出 1、0 registered / 0 applied；完整模型导入与 V4 真实路由回归也在官方依赖的导入期设备查询处受阻，Muse 还报专用 kernel 包缺失。原始成功/失败日志均保存在 ignored `artifacts/migration-20261105/`；长期统一测试表由测试负责人追加，不将报告器退出 0 当模型导入成功。模型 GPU、精度/性能 A/B、CP/TP/Graph、wheel 与导览浏览器视觉验收未执行。本轮是部分迁移与证据交接，阶段一未完成。

## 2026-11-06 阶段二验收：47 精度与性能对齐、26 环境阻塞

**精度（47 / gfx936 / BF16 KV / Qwen3.6-27B-INT8-w8a8 / TP2）**

- l2norm 迁入后 A/B 逐层 dump：layer 0/1/4/8/16/32/48/63 全部 `max_abs = 0.000e+00`（|1081 键）；A 侧两次独立启动逐位相同（`4f33666a`），旧基线 `7919f29e` 作废。
- 确定性文本：r02、r08 在新代码下 A/B 逐字一致（8/8 请求；独立复算 id/text/usage 哈希 r02 两侧 `d3582051242f`、r08 两侧 `d372fb258587`）。
- 结论：在该配方上等价性成立；不外推至 26。

**性能（同上配方）**

- 每侧 3 次独立启动（新进程、各 5 轮），轮内算分位数后取轮次中位数；R19 门禁剔除无效轮后源为 `perf-quantitative-v3.json`。
- 8 项指标中位数差 **−0.926% ~ +1.440%**；方向为 **B 在 7 项上数值略优，A 仅 TPOT p99 一项略优**（不得写“A 略优”）。
- 同侧极差实测 **A 0.52–2.70 pp / B 2.97–5.30 pp**。
- 结论：**未观测到可分辨差异**，不给差异上界。不得表述为“性能等价”或“差异 <1%”。
- 口径变更：旧 `≤3% pass` 判据因噪声地板大于 3% 而失效；旧“2.3–6.8pp 噪声地板”由两个不同窗口的两轮对比得出、含跨窗口环境差异，作废；现行尺度改为同侧多次独立启动的极差。

**26（gfx938 / fp8 KV，目标环境）——未验证 · 环境阻塞**

- **精度已取得**：26 上 A1/A2/B1 三次运行 8 条请求，`A1==A2 8/8`、`A==B 8/8`（text + prompt/completion token 哈希）；`dump26/{A,B}/tensors.pt` 45 个键形状一致且 `torch.equal` 全等。
- **性能未验证（用户要求停止）**：仅有一轮标定，且该轮并发 token 集与 47 不同，不构成性能结论。
- 27B TP2 起不来：`rye_sglang_dpsk-v41` 占 119.5–120.8 GB/卡（余 26.6–28.3 GB），`rest_memory` 为负 → 容量不足，非参数问题。
- 0.6B 替代路径不可用：B 侧 fused rotary dtype 缺陷（`expected scalar type BFloat16 but found Float`），47/26 均可复现；调用栈见开发记录第十一节。
- 遗留：26 的 `layer.json`/`fa.json` 存在首调用捕获缺陷，两份快照不能用于 A/B 比对（已记工具 todo）；A/B 结论以 `tensors.pt` 为准。
- R18 口径：26 串行 r02=54 / 并发 42、47 为 33/54 —— 批组成影响输出，两侧一致、各机自身稳定，不写成跨机差异证据。

**已知不透明点：插件不是透明替换**

- 两侧在 HCU 上都走 `rotary_embedding/base.py` 的 fallback 分支，差别在被调 kernel：A 由 `HcuRotaryFallback`（`hcu_sglang/layers/rotary.py:85-107`）在调用前把 fp32 `cos_sin_cache` 转成 `query.dtype`（带 memo、不改原张量），B 无此包装。
- 证据：插件代码 + `tests/test_rotary_fallback_dtype.py`（3 项 CPU 通过）；**运行期分支命中日志未打印**，属代码级证据而非运行期取证。
- 含义：27B 逐位一致只说明该模型上该转换无副作用，不等于插件对所有模型是 no-op。

## 2026-11-06 拉取远端：origin/main 被第三方 force push 与变基记录

**事实**：拉取时 `origin/main` 发生强制更新 `+ 1eb4aba...548e677`（forced update），**远端历史被第三方改写，非本次操作所致**。我们未推送过任何内容。

**为什么不直接 `git pull --rebase`**：本地 17 个 "ahead" 中只有 2 个是本仓提交，另外 15 个是远端本次改写掉的旧提交；`git diff --stat 1eb4aba 900b7fc` 为空，说明远端只重写历史、内容未变。直接 rebase 会尝试重放那 15 个重复提交，结果不可预测。实际采用 `git rebase --onto origin/main 1eb4aba main`，只重放本仓的 2 个提交。

**冲突处理**：唯一冲突文件 `architecture.html`（`tools/build_guide.py` 的生成产物），两处差异均为“生成时输入不同”的自然结果（Skill/工具计数、快照哈希与内嵌 catalog JSON），无语义冲突。按方案 A 逐字执行（rebase 期间 `--ours` 为基线侧、本仓提交为 `--theirs`），随后用容器 Python 3.10 重跑 `build_guide.py` 重建并以 `--check` 验证 `Guide is up to date`，生成物最终内容由生成器决定，未手工合并。

**变基前后 SHA 对照**

```text
远端改写   1eb4aba → 900b7fc（内容等价，git diff --stat 为空）
本仓提交   6005118 → a765f62（feat: 阶段一/二迁移收口）
            c053168 → 4ad9b96（docs: 阶段二等价性验证报告与口径同步）
新增       81d82d4（chore: 变基后重建导览快照，仅 architecture.html）
基线       548e677（远端新 main）
退回点     backup/pre-rebase-c053168 = c053168（推送完成后清理）
```

**变基后状态与门禁（容器 Python 3.10.12）**

```text
HEAD 81d82d4，main ahead 3 / behind 0，git status 空
ci/l0_gate.py → passed（673 Python / 8 Skill / 0 errors）
python3 -m unittest discover -s tests → Ran 190 tests，OK（skipped=13）
python3 tools/build_guide.py --check → Guide is up to date
```

**重叠文件复核**（远端新 main 与本地改动都触碰的 4 个文件，均未丢本地改动）：`docs/ci.md` 的「L0 迁移文件头」行在；`hcu_sglang/environ.py` 的 `SGLANG_REQUIRE_KV_CACHE_SCALES` 在；`hcu_sglang/layers/attention/flashattention_interface.py` 的 `q_4d` 构造在 varlen 分支之后；`architecture.html` 由生成器重建。远端新增的 `.github/workflows/quality-gate.yml` 亦在。

**推送状态**：未执行。推送前须重新 `git fetch` 确认 `behind` 仍为 0（远端可能再次被 force push），之后 `git push origin main` 并用 `git ls-remote` 与 `rev-list` 核对远端新提交。
## 2026-11-04 HCU KV 与 GDN 单流策略

范围为现有 `HCUMHATokenToKVPool` 的构造参数策略与 Qwen3.5 GDN 构造 AROUND Hook；不修改官方、adapter 或数学 kernel。基线、职责/回退、MTP 受控 A/B 与限制见 [开发记录](hcu-sglang-development.md#2026-11-04-kv-与-gdn-单流提交验证)，评审入口见 [任务记录](board.md#6-hcu-kv-与-gdn-单流策略2026-11-04)。

| 检查 | 实际结果与范围 |
|---|---|
| CPU 离线 | Python 3.10 无第三方启动注入的隔离环境，`python -m unittest discover -s tests -v`：117 项，104 passed / 13 条件 skipped。 |
| HCU KV GPU | `tests/runtime/hcu_mha_single_stream_gpu.py`：3 passed；真实父类构造、位置/关键字/重复参数、公共写入、不同 K/V 维度、空输入/页边界/padding、变输入 Graph replay。 |
| GDN GPU | `tests/runtime/hcu_gdn_single_stream_gpu.py`：4 passed；真实 HookRegistry、HCU/非 HCU、有效 original 异常保留、真实 BF16 投影、同流 capture 与变输入 replay。 |
| 完整入口与目标合同 | 严格入口检查 registered 598 / applied 584；新增 GDN 构造与既有三个 HCU pool builder Hook 均 applied，注册集合与更新后目标合同一致。14 项多模态 Hook 未应用，严格检查记 failed，不能记为全量接线通过。未修改基线为 registered 597 / applied 583；前后 14 项 unapplied 及错误列表完全一致，仅新增 GDN 目标并成功应用。 |
| 真实 TP2 服务与性能 | 本次先前四次 MTP 独立启动均通过 ready、模型身份及三道完整问答；320 次无 profiler 测量、八份完整 trace。提交整理新增的构造参数绑定另由真实 GPU 构造测试复验；性能数据不是最终提交后的新一轮服务测量。 |

导览重建与 `--check`、L0、`git diff --check` 及桌面/390px 窄屏导览渲染和搜索/跳转交互纳入本次提交验收，日志与截图保留于本批次产物。当前复验日志位于 `${PLUGIN_ROOT}/artifacts/single-stream-pr-20261104`，原始服务/性能证据为独立批次 `qwen38-tp2-mtp-gdn-ab-20261104`，ignored 产物不随 Git 推送；共享统计、固定版本与解释已写入开发记录。

收益限 gfx936、BF16 Qwen3.8-27B、TP2、batch1、指定 NEXTN/Graph/后端配置。1024→32 的 E2E 中位数下降 1.06%，decode-after-first 下降 1.41%；4096→1 prefill 增加 0.29%，每模式仅两个独立启动，未作全场景或严格零影响承诺。量化、其他模型/架构、短 prefill Graph、PD、并发及 wheel/embedded 未验收。提交对齐上游重写后的 `9755678`；与测量基线 `0aa4114` 相比仅协作规范变化，运行时代码一致。本轮提交评审，不包含合入或发布。
