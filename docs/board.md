# 后续任务与实施路线

> **本文用途**：列出方法库之后的建设方向、启动条件与接受标准，并约定任务如何登记。它不是运行时已支持能力清单；实际完成记录见 [交付验证](delivery-validation.md)，参与方式见 [协作流程](contributing.md)。

## 1. 当前阶段

V1 已建立方法、Skill、审计/回溯工具和 CPU 测试；V2 重整架构、文档组织、使用与协作规范。2026-10-28 起本仓同时维护 `hcu_sglang/`，迁入、提交与实际验证见 [开发迁移记录](hcu-sglang-development.md)；外部运行时 V3 的实现与指定 MiniMax TP4 验证已完成，见 [最终交付](hcu-sglang-development.md#2026-10-26-8-最终交付与需求验收)。托管 CI、其他模型及 adapter 组合验收仍按独立任务推进。

## 2. 建设路线

| 阶段/事项 | 启动前输入 | 最小交付与验收 | 状态 |
|---|---|---|---|
| 当前运行时维护 | 包名、entry point、官方版本与依赖范围 | 在本仓 `hcu_sglang/` 修改实现；同步注册、测试和开发迁移记录 | 已迁入，验证范围以开发迁移记录为准 |
| adapter/plugin 组合契约 | 两仓目标、加载时序、最低接口版本 | 目标所有权表、包装链与 fallback 集成用例 | 规划 |
| 历史 DAS 能力审计 | official/fork/plugin SHA、来源历史 | 差异清单、台账、逐项实现归属及所需运行组合 | V3 已完成固定输入对照；全 fork 等价未证明 |
| 首个新增能力 | 窄 seam、支持条件、基线与接受标准 | 可达注册、数值/状态/graph、适用服务 A/B | 待选能力 |
| 托管 CI 接入 | 平台、runner、触发方式与日志位置 | 真实触发并保留结果；CPU 与 GPU lane 分离 | 规划 |
| 回溯适配新上游 | 新 loader/bootstrap/package-data 格式 | fixture 回归、新输出 provenance 核验、明确运行边界 | 按版本需要启动 |

这些不是已经分配的任务，不预填负责人、开工日期或“已完成”。选择任务后按下一节建立具体记录。

## 3. 任务记录格式

| 字段 | 要记录的内容 |
|---|---|
| ID / 标题 / 负责人 / 状态 | 可稳定引用的事项和实际负责者 |
| 行为与接受条件 | 触发、支持范围、断言与预期结果 |
| 基线与分支 | 每个涉及仓库的 SHA/dirty 和工作分支 |
| 跨仓依赖 | 前置任务/接口、兼容版本、哪个仓拥有哪个目标 |
| 验证与产物 | 命令、环境、证据路径、未执行项 |
| 阻塞与下一步 | 缺少什么、解除条件、下一项动作 |

状态使用 planned / in-progress / blocked / done，并在 done 时链接实际结果。迁移台账的 implemented/verified 有更细证据要求，见 [治理](governance.md#4-处置台账与状态)，不要混成同一种完成口径。

## 4. 完成与交接

任务完成后把规则沉淀到相应设计/操作文档，把结果写入 [交付记录](delivery-validation.md)，更新 Skill 路由和兼容范围。看板保留可追踪入口，不把已经失效的日志摘要当成后续版本的通过结论。


## 5. plugin-v3-1028 全量迁移执行

状态：in-progress。固定来源、阶段与实际结果见 [开发迁移记录](hcu-sglang-development.md#6-2026-10-29-分阶段继续迁移)，逐路径范围见 [执行台账](../plans/sglang-das-execution.json)。2026-10-29 已完成当前包工具支持、HCU 标准分发哨兵修复及源码/wheel/embedded 加载复验；全量源码与行为处置仍未闭合。

下一阶段先审查 DeepSeek V4 平台类拆分和当前 `current_v4` profile，随后按领域继续审阅 Attention/KV、量化/通信、GLM5-next/Muse Spark 与扩展功能。每项补充行为、实现、支持/回退及 CPU/GPU/Graph/模型证据；只有实际满足该阶段退出条件才更新状态。保持用户要求：提交本地仓库，不推送远程。


## 6. 2026-11-05 模型侧迁移收口

| 字段 | 本轮记录 |
|---|---|
| ID / 负责人 / 状态 | models-20261105 / 模型领域负责人 / blocked（完整 GLM/Muse 依赖与加载）；V4 接线和同步工具的已审改动已完成 CPU 自测 |
| 行为与接受条件 | V4 public forward 安装缺失 helper；current_v4 保留方法类上下文；GLM5-next 与 Muse 迁入完整 owned body、正式架构注册和来源合同；完整模型验收需解除运行时依赖阻塞 |
| 基线与工作树 | main 起始 `1eb4abaa3a3ad5bb7bbc9a8b758488639f4db1c7`，插件工作树含本轮及并行运行时修改，未提交；DAS `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`、官方 `baa9befa381511cd4d56f83980951e5a45fffc6d`，来源只读 |
| 依赖与下一步 | 运行时领域补 GLM 五个专用模块及 Muse kernels/layers/config/processor 后，模型侧更新 import 对应并复验完整导入；599 项严格 loader 尚无本轮通过证据；GPU/模型 A/B 由独立测试负责人统一执行 |
| 证据与交接 | [模型开发记录](hcu-sglang-development.md#2026-11-05-glm5-next-与-muse-spark-owned-模型迁入)、[V4 修复记录](hcu-sglang-development.md#2026-11-05-v4-平台类与方法同步模型侧)、[交付边界](delivery-validation.md#2026-11-05-模型侧阶段记录)；来源及状态见执行台账，不宣称全模块迁移完成 |
## 6. HCU KV 与 GDN 单流策略（2026-11-04）

ID：hcu-single-stream-kv-gdn；负责人：本次提交作者；状态：in-progress（提交评审，未合入）。用户本轮明确授权推送并创建 PR；第 5 节“不推送”仅适用于其历史迁移任务。

行为与接受条件：保留 HCU KV 布局、有效 original 链和数学实现；KV 写入与 Qwen3.5 GDN 投影使用当前流；验证构造签名、非 HCU 回退、数值/变输入 Graph、TP2 真实输出及 MTP 预热 A/B。基线、依赖、分支及未覆盖范围见 [本次开发记录](hcu-sglang-development.md#2026-11-04-kv-与-gdn-单流提交验证)，实际复验见 [交付记录](delivery-validation.md#2026-11-04-hcu-kv-与-gdn-单流策略)。后续：评审后按仓库 squash-only 策略合入；本轮只创建 PR。
