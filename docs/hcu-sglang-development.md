# hcu_sglang 开发迁移记录

> **本文用途**：持续记录本仓 `hcu_sglang` 每次代码迁移或开发完成的步骤、提交范围、实际验证与未完成条件。2026-10-28 起，本仓从仅维护方法推进到同时维护运行时代码；早期方法库限制不再禁止当前包落地。

## 时间索引

开发与验证记录按日期从早到晚排列；同一天沿用原记录顺序，未提供具体时刻的条目不推定先后。跨日执行的历史记录按起始日期归档，保留原文中的实际执行日期与验证边界。

- [2026-10-26：V3 实施与验证](#2026-10-26-v3-2026-10-26-实施与验证记录)
- [2026-10-27：方法库导览与统一入口](#2026-10-27-v3-方法库导览与统一任务入口实施记录)
- [2026-10-28：运行时代码迁入](#2026-10-28-运行时代码迁入与入口验证)
- [2026-10-29：继续迁移与四卡验证](#6-2026-10-29-分阶段继续迁移)
- [2026-10-30：后端与模型路径同步](#20-2026-10-30-dsa-indexer-当前-hcu-producerconsumer-同步)
- [2026-11-01：分支迁移与八卡验证](#35-2026-11-01-rmsnormgemma-hcu-分支继续迁移)
- [2026-11-02：DAS 分支逻辑对齐](#das-分支逻辑对齐2026-11-02)
- [2026-11-12：PD prefill inflight 队列 rank 一致性](#2026-11-12-pd-prefill-inflight-队列-rank-一致性)
- [附录：布局、操作方式与汇总](#附录布局操作方式与汇总)

两份早期独立开发记录已全文收录。历史结果不代表当前代码已重新验证。

## 2026-10-26 V3 2026-10-26 实施与验证记录

> **用途**：记录 `plugin-v3-1026.txt` 指定输入下的插件开发、内嵌源码生成、HCU 逻辑对照及本轮验证。执行跨 2026-10-26 至 2026-10-27（Asia/Shanghai）。方法库保留 Skill、工具和记录，运行时代码位于独立 `output-code` 仓库。

> **当前交付**：以下 §1–7 保留各阶段快照；正式输出、最终验证和限制统一见 [§8](#2026-10-26-8-最终交付与需求验收)。

### 2026-10-26 1. 固定输入与产物

工作根目录以 `${BASE}` 表示；下表保留历史相对布局，不要求当前部署沿用。`${BASE}` 仅是历史位置标识，执行任务使用 [路径配置](configuration.md#1-当前工具的路径配置) 中的本地变量。

| 对象 | 路径及版本 | 状态与用途 |
|---|---|---|
| 社区输入 | `input-code/sglang`；`baa9befa381511cd4d56f83980951e5a45fffc6d` | 干净来源，提供当前框架和正式插件接口 |
| HCU 输入 | `input-code/sglang-das`；`15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e` | 干净来源；较上轮增加 139 个文件改动 |
| 插件输出 | `output-code/sglang-plugin`；v0.3.1，`39038d5394c23e6b918e6e50911ee5292a1e4d34` | 独立包 `sglang_dcu_plugin`，入口 `dcu` |
| 回溯输出 | `output-code/sglang-das` | 由上述社区与插件两个干净输入生成，清单记录来源 SHA |
| 方法库 | `hcu-sglang-plugin` | 本文、Skill、抽取工具修正与验证记录 |
| 执行容器 | 历史验证容器 | Python 3.10.12；宿主 Python 3.6 不用于工具执行 |
| 证据目录 | `hcu-sglang-plugin/artifacts/v3-20261026` | 原始日志、资源准入、响应、哈希与逻辑对照，Git 忽略 |

开始执行时 `output-code` 不存在。插件起点从保留的源码快照恢复，运行时逐文件与历史 `45e67e0` wheel 一致，记录见 `baseline-recovery.json`；新仓库恢复提交为 `bb76e1b`，没有冒充恢复原仓库已移除的 Git 历史。历史验证结果没有计入本轮通过数量。

### 2026-10-26 2. 已实现模块与证据边界

| 模块 / 实现位置（插件仓库内） | 本轮建设内容 | 本轮证据与限制 |
|---|---|---|
| 注册与参数解析：`__init__.py`、`server_args*.py` | 保留正式入口，新增配置与通信目标，目标合同更新为 210 项 | 严格 loader 检查非空、预期集合、全量 apply、实际来源与三个 resolution seam；注册不代表所有分支均已执行 |
| V4.1 配置：`configs.py` | HCU Channel FP8 转为 compressed-tensors channel-weight / token-activation 配置，接在官方规范化入口之后 | 通过真实 `DeepseekV41Config` 构造入口；其他量化格式和非 HCU 保留基线。未运行完整 V4.1 模型 |
| 通信：`distributed.py` | HCU native all-reduce 在 Graph capture 中使用 copy-in，保留 disabled/shape 准入和非 capture 的有效基线 | CPU 分支断言及真实 TP4：每 rank 两个 Graph 分配，三轮按 rank 交叉选择重放；不外推 AITER Fabric、DP 模型或全部拓扑 |
| 方法抽取：`models/deepseek/**/methods` | 七个 V2/V4/NextN 构造函数显式绑定拥有类；修复八处二次常量限定 | 修复前八项执行失败，修复后通过；覆盖构造父类边界和 MHC norm 分支，不能替代整模型构造与生成 |
| 常量别名：`models/deepseek/hook_utils.py` | 覆盖前保存旧引用，仅刷新已存在的 DeepSeek 消费者别名；缺失属性不视为 `None` 别名 | 两项修复前失败 / 修复后通过；不向无关 vendor 模块注入属性 |
| JIT 原生资源：`jit_overlay` | 更新 `c1.cuh`、`c2.cuh`、`c_plan.cuh`、`fp4_indexer_rope.cuh`、`fp4_utils.cuh` | 283 个 native 文件及一项许可资源，共 284 项与当前 HCU / embedded 哈希一致。新增 FP4/压缩算子的完整数值验证未执行 |
| MiniMax 与层边界：`models/minimax_m2*`、`layers/layer_boundary.py` 等 | 保留 residual-stream 与 FFN 出口归约契约；没有重新引入旧生产者重复 all-reduce | 当前 CPU、GPU 与本轮 TP4 服务结果。该兼容实现与历史 fork 文本可不同，需按最终输出与归约所有权比较 |
| 内嵌集成 | 复制 442 个 plugin 文件，注入官方 bootstrap、loader、依赖、许可及 package-data | 448 项生成清单通过；SGLang 入口 bootstrap 替代外部 `usercustomize.py`，属于显式集成差异 |

### 2026-10-26 3. 故障复现与工具修正

#### 2026-10-26 3.1 常量重写与类闭包

旧正则先把 `_is_hcu` 改成 `C._is_hcu`，又在下一轮替换 `_is_hcu`，产生 `C.C._is_hcu`。类方法抽成普通函数后，原来的 `super()` 也失去隐式 `__class__` 闭包。仅检查 AST、注册和 apply 均无法发现这些执行错误。

当前 [方法同步工具](../tools/sync_deepseek_methods.py) 根据 AST Name 的 UTF-8 字节位置修改引用：不改变属性名、字符串和注释；重复执行不会再次限定；遇到本地同名变量拒绝猜测。构造方法使用明确拥有类和 receiver；保留普通装饰器，遇到 classmethod/staticmethod/property 描述符时要求先声明绑定契约。该改进不代表历史抽取 profile 已适配当前 HCU 的类拆分，`--check` 仍需完整预检。

回归包括真实函数执行、父类调用、重复变换、中文文本偏移、局部变量冲突、装饰器行为以及来源符号漂移时不写输出。方法库完整门禁结果单独记录，不计入插件测试数。

#### 2026-10-26 3.2 预导入别名

旧 patcher 先修改 owner 常量，再调用传播函数；传播函数读到的已是新对象，因而跳过旧消费者。修正后先捕获旧值并限制作用域，只更新明确存在、仍引用旧对象的 DeepSeek 消费者。测试同时检查无关模块和缺失属性，避免修复变成全局副作用。

#### 2026-10-26 3.3 执行身份

容器 root 执行只读 Git 审计时触发仓库所有权检查。审计、源码构建及回溯改由源码 UID 执行；运行服务沿用容器已有合法设备用户。未修改全局 Git safe.directory、HOME、系统 Rust 或设备权限。

### 2026-10-26 4. 源码、内嵌与 HCU 逻辑对照

本轮执行了三个不同层次的比较，原始结果为 `fork-logic-comparison.json`：

1. **产物保真**：442 个内嵌插件文件逐字节等于本轮插件；448 项生成清单核验通过。启动集成使用生成器定义的 bootstrap 映射，不能把不生成全局 `usercustomize.py` 误报为丢文件，也不能忽略入口顺序。
2. **来源保真**：284 项 JIT 资源逐项比较当前 HCU、插件、embedded 三方；Channel FP8 转换 helper 的 AST 与当前 HCU 相同。
3. **行为与版本差异**：检查整个 `python/sglang` 当前树，社区与 HCU 有 1206 项内容不同、387 项 HCU 独有、20 项社区独有。抽查的 24 个方法中，21 项需要适配/陈旧行为复核，3 项源拥有类或方法已变更。不能把文件名、签名、AST 相似或 210 个目标全部 apply 当成整个 fork 逻辑等价。

**对照结论：本轮交付形态和明确验证的契约已有证据；整个 HCU fork 的全逻辑等价尚未证明。** 当前 HCU 的 CP LayerSplit 全链路、候选/分页 FP4 索引、PD/DSpark 辅助状态、V4.1 中间 chunk decoder、EnGram、W4A8/DeepEP，以及 AOT `rope_fp4` 构建导出没有在本轮形成完整运行验收。这些能力不得标记为已迁移或已验证。

对旧源码差异的处理遵循当前 caller/consumer 契约。例如 MiniMax 的框架 `FfnExit` 已负责归约，保留 fork 旧生产者归约会重复求和；此处应保留有回归与服务证据的适配差异，不能为了文本相等恢复旧操作。

### 2026-10-26 5. 验证矩阵

| 检查 | 插件形态 | 内嵌形态 | 原始证据 |
|---|---|---|---|
| CPU 与契约 | 304 passed / 8 skipped | 304 passed / 8 skipped | `plugin-cpu-final.log`、`embedded-cpu.log` |
| 严格加载 | 210/210，三个 resolution seam | 210/210，屏蔽外部入口 | `plugin-runtime-final.json`、`embedded-runtime.json` |
| 单设备 GPU / Graph | 8 passed | 8 passed | `plugin-gpu-final.log`、`embedded-gpu.log` |
| TP4 native collective Graph | 四 rank × 三轮通过 | 四 rank × 三轮通过 | `plugin-collective-graph-final/rank-*.json`、`embedded-collective-graph/rank-*.json` |
| MiniMax TP4 服务 | health 200；`2、2、Paris`；正常 stop；自有进程清理 | health 200；`2、2、Paris`；正常 stop；自有进程清理 | `*-service-attempt1/*-service-result.json` 与原始响应 |
| 方法库门禁 | L0 通过；45 项工具单测通过 | 同一方法库，不重复计数 | `method-l0-final.json`、`method-tests-final.log` |

原始配方为上级 `minimax-debug/d.sh`；本轮按已有授权使用物理 4–7 卡 TP4，仅修改 TP 数和 NUMA 列表，保留 FA3、FP8 权重/KV、Graph、context 131072、chunk/prefill 和服务端口。`recipe-manifest.json` 保存原文与派生脚本哈希、精确差异及可见设备映射。

确定性响应通过 [响应校验工具](../tools/validate_generation_response.py) 校验实际模型、完整最终答案、正常 stop 和控制字符。该 checkpoint 在本次响应 content 中返回闭合前导 think 段，使用本次明确允许的 `allow_think_prefix`，未将 `enable_thinking=False` 宣称为全局停用思考的保证。重复请求记为 repeat，不作为 cache hit 证据；本轮不宣称 TP8、数据集精度或性能已通过。

### 2026-10-26 6. 复现与 Skill 入口

- 三方输入/差异审计：按 [迁移指南](migration.md) 调用 [迁移 Skill](../skills/sglang-hcu-plugin-migration/SKILL.md)，固定本轮 SHA 后生成新的审计和台账。
- 修改能力或抽取方法：按 [插件开发指南](plugins.md) 调用 [开发 Skill](../skills/hcu-plugin-dev/SKILL.md)，先保留真实构造、装饰器、状态与出口边界，再执行对应回归。
- 生成新 embedded：按 [回溯指南](backtracking.md) 调用 [回溯 Skill](../skills/hcu-plugin-backtrack/SKILL.md)。两个生成输入为 official/plugin；提供的 HCU 输入用于独立逻辑对照，不替换生成来源。
- 安装交付：按 [安装指南](installation.md) 使用已有源代码或 wheel，先验证实际解释器和模块来源，再运行目标服务。

证据脚本在本轮 ignored artifacts 中固定路径与 profile，只作为可复现实验记录；未来版本按对应操作指南重新固定输入和执行范围，不复用本页通过状态。

### 2026-10-26 7. 逻辑对齐后续工作

以下保留 v0.3.1 起的检查点与后续开发过程；各阶段的 pending 状态按当时范围解释，当前需求验收见 §8。完整 HCU fork 的所有模型与分支等价不属于这些局部通过结论。

后续开发位于独立的 `output-code/.sglang-plugin-v3-parity`（分支 `v3-hcu-parity`），以本页插件提交为起点。初始提交 `f2394ab` 迁移了 CP LayerSplit 的十个模块及分页/状态/描述符回归；后续 `0.3.2.dev1` 已接入父池、容量预算、工厂和握手规范化，进展见 §7.2。完整模型前向与 PD 网络字节传输尚未闭合，不能标记为可启用的完整 LayerSplit 服务。


#### 2026-10-26 7.1 CP 索引重排的设备准入修复

候选第二次局部回归为 23 项通过、1 项失败：HCU 主机上的 CPU 张量进入 LightOP 的 device-only remap，报 `indices must be device int32 or int64`。设备能力与输入 tensor 的实际设备不是同一个条件；当前候选同时检查 HCU 与 `indices.is_cuda`，CPU 张量使用原 PyTorch 路径。

修复后，候选局部测试为 **32 passed / 136 subtests passed**，证据 `cp-layer-split-candidate-attempt3.log`。新增对照包含真实 LightOP 的 int32/int64、乱序页、静态负值 padding、缺页、负 token ID，以及三轮变输入 Graph 重放；CPU 路径另检查不导入设备专用依赖。状态机测试中的部分边界沿用 HCU 的 AST 节点提取方式，不能代替完整模块导入、pool 构造或模型运行。方法来源及适配哈希在候选 `source-provenance.json`，能力接线状态在 `PARITY-CANDIDATE.md`。


#### 2026-10-26 7.2 CP 父池、容量与握手接线

候选 `0.3.2.dev1` 的实现提交为 `fe45284b9f17e0fde572d72a0751e9b8cddebb56`，显式 GPU 测试准入提交为 `1f1ff9d75d0857e5914b8fa328bce8acafd9cf87`。与最初仅搬迁模块的检查点相比，本次保留当前 HCU 的完整缓存池类，复用 AST 相同的官方值类型和独立 unified pool，通过正式 class REPLACE 接入五个缓存池类。同步迁移 DSV4 预算器、CP 工厂、容量 MIN 及 DSpark 全局描述符规范化。原有六个局部池 Hook 由一致的池构造/状态契约替代，候选的 HCU 目标集合变为 214 项。

声明的父池、INT8 helper、masked store、预算类、工厂及握手共 37 个类/函数节点，在明确 namespace 和插件进程选项映射后与当前 HCU AST 一致，证据 `hcu-pool-source-comparison.json`。该对照覆盖声明的节点与复用边界，不能替代其余模型/网络链路审阅。

本次增加了真实父池和四个 CP rank 的构造、公共工厂、分配缓冲与传输描述符、masked BF16 Graph，以及 packed FP4 threshold / RNE 舍入和变位置 Graph 读写。四卡验证使用真实 PyNCCL、真实 `ServerArgs` 投影和池分配，覆盖三轮 SWA/低压缩比 KV、非 owner 未预取拒绝、空查询 rank 参与、共享 staging 重用、容量最小值和真实分片/未分片池握手；没有启动 DeepSeek 模型引擎。

| 检查 | 候选源码 | 候选内嵌 | 原始证据 |
|---|---|---|---|
| 原有 CPU 回归 | 304 passed | 本阶段未重复全量 CPU；按改动执行下列池/设备验证 | `cp-parent-baseline-cpu.log` |
| CP CPU 合约 | 40 passed / 1 skipped，138 subtests；排除 11 个 GPU 用例 | 局部合约 38 passed / 2 skipped，2 subtests；排除读取候选源码的 AST 测试，包含实际池和设备边界 | `candidate-cp-cpu-final.log`、`candidate-embedded-cp-contracts.log` |
| 显式 GPU lane | 18 passed / 1 skipped | 18 passed / 1 skipped | `candidate-gpu-optin-final.log`、`candidate-embedded-gpu-optin-final.log` |
| CP4 真实池/通信 | 四 rank 全部通过 | 四 rank 全部通过 | `cp-tp4-pool-attempt2/rank-*.json`、`candidate-embedded-tp4-pool/rank-*.json` |
| 严格加载 | 214/214，正确源码来源 | 214/214，屏蔽外部入口并固定内嵌来源 | `candidate-pool-runtime-final.json`、`candidate-embedded-runtime.json` |

GPU lane 包含原有八项和新增十项实际通过用例。最初未设置 opt-in 的原有 GPU 运行仅产生八项 skip，该日志不计通过；随后明确设置 `HCU_PLUGIN_RUN_GPU_TESTS=1` 执行。CPU 与内嵌局部 lane 范围不同，不按通过数直接比较覆盖率。

回溯产物位于 `output-code/.sglang-das-v3-parity`，由干净候选和官方两输入生成。466 项清单核验通过，含 460 个插件文件。内嵌检查屏蔽外部入口，并确认 SGLang 与插件模块均来自该输出树。AST 提取测试若通过测试路径定位候选源码，就不能计作 embedded 覆盖；因此内嵌局部回归排除了这类测试，另以实际导入、构造和设备行为验收。

首轮 CP4 失败是验证器误把 PyNCCL 初始化后的默认 `disabled=True` 当作不可用。直接探针确认 RCCL 2.22.3 与基础符号正常；当前先验证 `available`，再在 `change_state(enable=True)` 作用域内断言真实归约数值，随后执行池广播。初始失败日志保留，没有把跳过 collective 当作通过。

两项 INT8 池用例要求源实现支持的 gfx936；当前节点为 gfx938，按 gate 跳过。此处不声明该特性已在本节点验证。候选 wheel 已构建并核对 460 个插件文件及新增 provenance 资源；尚未把它作为隔离 wheel 安装形态运行，不计作 wheel 运行验收。

CP 父池阶段结束时仍待完成正式启用入口、模型 prefetch/write/read 消费者、prefill/decode 描述符生产、Mooncake 追加字段及匹配复制和 DSpark 辅助状态。后续 PD 协议进展见 §7.3；整体集成完成后再合入正式输出并重新验收服务。`HCU_PLUGIN_CP_CACHE_LAYER_SPLIT` 目前仅供候选池/通信契约使用，不是完整模型能力承诺。


#### 2026-10-26 7.3 PD 描述符生产、协商与复制

候选升级为 `0.3.2.dev2`，提交 `f9a838c0de11e369bee70125c9703d06636c6763`，仍位于独立 `output-code/.sglang-plugin-v3-parity`。本阶段从公开 `setup_state_kv_args` 补齐 V4 描述符生产，将 CP 拓扑、KV 布局、类型信息接入 HTTP bootstrap，连接 decode 真实 receiver 构造、ZMQ 注册与 prefill peer 发布。注册消息保留旧 0–19 字段（ZMQ 帧），20/21/22 分别追加 state formats、state types 和 V4 描述符；短旧消息仍可解析。

复制入口按照 `(family, global layer)` 匹配指针和每项字节跨度。对来源实现保留的两处问题，本次补充了失败反例与适配：

- 描述符合法但目标顺序不同，位置式 stride 前检可能先返回 `-1`；函数级四项反例覆盖未缓存 verdict、单批/分批/线程路径。现在描述符路径由身份匹配检查 ABI，位置式检查继续用于原有路径。
- 状态组件已按类型和重复次序解析目标组件，目标描述符却仍用原索引；一项重排反例在修复前拒绝合法布局。现在指针、长度、索引与描述符都沿解析后的目标组件读取。

peer 发布前检查描述符与指针/stride 数量、二进制字段与 JSON 中 KV 大小的一致性；坏的重新注册会撤销旧有效 verdict。上游 worker 保留原控制流，只增加其当前注册快照的描述符参数，保证与该次指针列表一致，未改成重新读取可变 peer 表。

32 个 HCU 来源节点及一个上游 worker 调用适配完成明确映射后的 AST 对照，包含 async HTTP handler、classmethod 恢复、显式 super owner、文档字符串缩进及上述功能修复。`pd/pd-source-comparison.json` 和插件内 `disagg/source-provenance.json` 保存范围与哈希。该结论不覆盖未迁移模型及流式状态分支。

| 验证 | 源码形态 | 内嵌形态 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU 基础与合约 | 369 passed / 1 skipped；11 GPU deselected，138 subtests | 本阶段按 PD 改动面验证下列项 | `pd/full-cpu-attempt1.log` |
| 新增写入 drain / 公开链路 | 3 passed（2 个 drain、1 个隔离公开入口） | PD 单测及公开链路共 28 passed | `pd/drain-public-integration.log`、`pd/embedded-contracts.log` |
| 严格加载 | 241/241，正确来源 | 241/241，外部入口屏蔽 | `pd/runtime-final.json`、`pd/embedded-runtime.json` |
| 原生 Mooncake TCP 公开链路 | 通过 | 通过 | `pd/native-public-path-final.json`、`pd/embedded-native-public.json` |
| 显式 GPU lane | 18 passed / 1 skipped | 18 passed / 1 skipped | `pd/gpu-optin.log`、`pd/embedded-gpu.log` |

公开链路采用真实 CPU 池分配、HTTP/ZMQ 协商、真实 receiver 构造和原生 Mooncake TCP session / memory registration。通过公开 KV 与状态发送方法后，逐字节比较目标缓冲，包括未应写入区域。独立的有界 CPU engine 模式保留作回归，它与原生 TCP 证据分开记录；两者均明确不是模型服务或 RDMA 硬件验收。恢复旧 peer 前检的正常顺序组合对照也通过，因此不将函数级失败反例描述为所有基线调用都会失败。

新增写入完成测试覆盖一个 future 异常或返回错误、另一写入仍在执行的情况：确认 helper 等待已启动写入结束后才返回或抛出，防止提前释放 buffer。CPU 失败处理证据不等同于完整流式源事件/ACK 生命周期已闭合。

回溯使用干净的官方 `baa9befa...` 与候选 `f9a838c...` 两个输入，475 项清单核验通过，其中 469 个插件文件；证据 `pd/embedded-manifest-verify.json`。内嵌加载屏蔽外部入口，原生 TCP 验证器同时固定 SGLang 与插件的实际导入根路径。

已验证的 v0.3.1 主输出与其 448 项生成清单保留。本阶段没有把新候选的协议/设备结果写成重新通过 MiniMax 模型服务。

下一阶段仍需模型 prefetch/write/read 消费者、DSpark hidden-state 流式生产与 ACK、source event / 在途 buffer 生命周期、候选索引/稀疏 prefill、W4A8/DeepEP 和其余模型 caller 差异。整体目标仍在进行，不将本阶段协议路径通过写作完整 HCU fork 等价。


#### 2026-10-26 7.4 MQA 类图、缓存读取与原生边界

候选 `0.3.2.dev3` 的实现提交为 `5b182d7f8e39cd1637ca2c10bafb033c9e619911`。从当前 HCU 保留完整 `MqaAttentionBase`、`MQALayer` 与 `DSparkAttention` 类，替代四个旧 MQA 方法 Hook；DSpark 显式继承候选基类，避免仅更新模块别名仍留下旧继承关系。后端新增 metadata prepare、tail reset、普通/sparse-prefill 缓存读取及原生 FlashMLA 接口，HCU 严格目标集合为 246 项。

实际池与 ForwardContext 回归覆盖 CP 空查询的 prepare→SWA/extra prefetch→wait→返回顺序；旧 forward 在同一边界的反例为 1 failed / 1 passed，保留于 `attention/empty-cp-before.log`。普通 attention 的页索引和 sparse-prefill 独立 flat indices 分别重映射，检查共享 metadata 未被改写。昂贵计算和通信载荷可替换，但不绕过决定调用顺序的控制流。

真实 GPU 验证中发现并修复三项问题：

- 独立无 residual 的 RMS 进入当前环境缺少的 vLLM `_C.rms_norm`；HCU 设备路径改用实际可用 LightOp，连续/非连续输入及变输入 Graph 有独立参考。原 residual 和 batch-invariant 路径按现有契约保留。
- overlay 哈希和 extra include 正确，但 loader 的 `DEFAULT_INCLUDE + extra` 使官方同名头文件仍优先，真实 JIT 报缺少 HIP warp 宏。注册时在 BuildSpec 冻结前调整共享默认列表的有效优先级，验证引用、幂等和非 HCU 分支；没有修改输入源码或全局工具链。
- 当前 FlashMLA dtype matcher 拒绝 FP8 FNUZ typed carrier；在 HCU native 入参边界使用 uint8 view，保持 pointer、shape、stride 与 bytes。BF16 和非 HCU 保持各自载体。

24 个来源节点/构造器后缀、9 个实际复用的官方 HIP helper 完成限定 AST 对照，证据 `attention/source-comparison.json`；插件内 `models/deepseek/current_v4/source-provenance.json` 记录源/目标哈希、装饰器映射、carrier 适配及命名空间边界。构造器沿官方初始化后追加 HCU 选项，不把整个 HCU 后端构造器标记为搬迁完成。

小型 MQA 使用真实 GPU 权重、norm、RoPE、KV store、FlashMLA 和输出投影，分别执行 FP8 与 BF16 gather。Q 对独立 PyTorch 参考、输出对直接原生 attention 加独立投影参考均为最大差值 0。backend metadata 由显式 fixture 提供；这不是完整模型引擎，也不是原生 attention 算法的独立 FP32 精度证明。

当前回溯由干净官方与 `5b182d7...` 两输入生成，482 项清单通过，其中 476 个插件文件。wheel 的全部 476 个插件文件与候选逐字节一致，含新 provenance；wheel 资源验证不计作隔离安装运行验证。验证结果及边界如下表。

| 验证 | 候选源码 | 候选内嵌 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU 基础与合约 | 384 passed / 1 skipped，16 GPU deselected，138 subtests | attention 与 CP 导入/构造合约 40 passed / 1 skipped，16 GPU deselected，2 subtests；排除从候选测试路径读源码的 AST 用例 | `attention/full-cpu-corrected.log`、`attention/embedded-contracts.log` |
| 严格加载 | 246/246 | 246/246，屏蔽外部入口 | `attention/runtime-final.json`、`attention/embedded-runtime.json` |
| 显式 GPU / Graph | 23 passed / 1 skipped | 23 passed / 1 skipped | `attention/gpu-regression.log`、`attention/embedded-gpu-attempt2.log` |
| 小型 MQA FP8 / BF16 gather | 两种模式均通过 | 两种模式均通过，实际后端来自内嵌树 | `attention/mqa-gpu-attempt5.json`、`attention/mqa-bf16-gpu.json`、`attention/embedded-mqa-*.json` |

GPU lane 新增 packed KV/upconvert、变选择与长度 Graph、超过 int32 的 4GiB output stride，以及独立 RMS；gfx936 专用 INT8 用例在当前 gfx938 继续跳过。首次 CPU 回归的唯一失败是旧注册测试同时要求并禁止 class REPLACE，修正为按平台检查目标集合后全量通过。

内嵌 GPU 首轮为 22 passed / 1 failed / 1 skipped：MiniMax 测试的 PyTorch 参考在 `hipblasCreate` 报分配失败。该例隔离运行通过，同一完整 GPU 组重跑为 23 passed / 1 skipped；未改生产代码，也未确定首次失败的根因，保留 `attention/embedded-gpu.log` 与 `attention/embedded-router-isolated.log`。

完整 CP compressor/indexer、DecoderReplayTail runner/模型循环、DSpark streaming source-event/ACK、W4A8/DeepEP 和完整 DeepSeek 服务仍待集成。本阶段不替换原 v0.3.1 主输出，也未将局部 attention 结果记作重新通过 MiniMax 服务。相关经验与开发 Skill 使用方式见 [类图与空查询](development-methodology.md#14-attention-类图与空查询通信顺序)、[JIT 头文件](development-methodology.md#15-jit-实际头文件优先级) 和 [原生接口](development-methodology.md#16-原生算子-provider-与-fp8-carrier)。


本阶段保留性审计还发现官方索引文件 SHA 从 `606b51f2...` 变为 `acca1b00...`，mtime 对应回溯来源检查；HEAD 和源码工作区均未变，当前 index tree 与 HEAD tree 同为 `27762040...`，暂存 diff 为空。原始索引字节未备份，故不声称原索引字节保持不变，细节见 `attention/official-index-refresh.json`。生成器现对子进程设置 `GIT_OPTIONAL_LOCKS=0`，新增真实双来源 mtime 反例；旧实现两项子检查失败，修复后方法工具 46 项单测通过。后续实际回溯再次检查前后输入索引哈希，结果保存在 `attention/index-preservation-regeneration.json`。这一修复不修改插件实现，内嵌运行结论仍限定于已核验的相同产物字节。


#### 2026-10-26 7.5 低压缩比缓存的持久写入边界

候选 `0.3.2.dev4` 已提交为 `02e11c96418310bd3a69fcdfc606c988aea29791`：正式接入 C1/C2 decode、C2 prefill 的 Python wrapper 与 backend `_low_ratio_compress_fused`，新增四个目标，HCU 合同总数为 250。该调用点使用 `get_extra_key_buffer_for_write`，不再把读取用紧凑 staging 当持久 KV；非 owner 取得 None，由 Python 零页载体与既有 native no-store 分支配合，继续计算 latent、维护 C2 ring 并发布复制的索引 K。

7 个 HCU 来源函数完成 AST 对照，仅调整模块级相对 import 路径；使用的 `c1.cuh`、`c2.cuh` 与固定 HCU 来源逐字节一致。证据为 `compressor/source-comparison.json` 和插件内 `compressor-provenance.json`，不动态导入 fork。

反例分为两层：旧 wrapper 的三个无主 KV 场景在 `.view` 处失败；旧 backend 的六个池边界对照为 3 failed / 3 passed，失败均为非 owner 错把读 staging 传作写目标。新的六个真实 CP 池边界均通过，断言 no-store 的同时仍发布索引 latent，且共享 staging 字节不变。首版测试误以为读 getter 在所有阶段均应抛错，改为按实际池生命周期验证写入所有权，没有为迎合测试修改生产池行为。

原生六项 GPU 验证覆盖 C1、C2 decode、C2 prefill 的 owner/no-store，每项三轮变输入 Graph；对照独立 PyTorch latent 与 pair-state 参考，另检查 V4 页内 payload、scale 和保留区的写入范围。正式 loader 的函数 REPLACE 使用 `functools.wraps` 保留原 metadata，首次 provider 断言因把 `__module__` 当实现来源而失败；后续从正式公开函数调用验证实际行为，由严格加载报告核对目标。

| 验证 | 源码形态 | 内嵌形态 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU | 390 passed / 1 skipped，22 GPU deselected，138 subtests | 本阶段按 attention/写入改动验证，18 passed，11 GPU deselected | `compressor/full-cpu.log`、`compressor/embedded-contracts.log` |
| 严格加载 | 250/250 | 250/250，屏蔽外部入口且固定内嵌来源 | `compressor/runtime-final.json`、`compressor/embedded-runtime.json` |
| 显式 GPU / Graph | 29 passed / 1 skipped | 29 passed / 1 skipped | `compressor/gpu-full.log`、`compressor/embedded-gpu.log` |

回溯的 485 项文件哈希通过，包含 479 个插件文件；输入索引前后相同，见 `compressor/index-preservation-reconstruction.json`。wheel 内全部 479 个插件文件与当前候选相同，含新增来源记录；仅做资源核验，尚未作隔离 wheel 运行验收。

完整 CP query-lens、candidate/top-k indexer、模型循环和流式 ACK 生命周期仍待集成；本节不构成完整 CP 模型服务声明。新增经验与开发 Skill 的检查入口见 [持久写入边界案例](development-methodology.md#18-缓存读取-staging-与持久写入目标)。


#### 2026-10-26 7.6 完整后端依赖图与 CP 入口集成

候选进入 `0.3.2.dev5`，在此前逐方法接线后改为完整 `DeepseekV4AttnBackend` 类及相关 metadata、C1/C2/C4 indexer、compressor、候选与 CP strategy 依赖图。固定 HCU 来源的 42 个模块共 482 个顶层类/函数节点、9 个额外桥接函数节点完成声明映射后的 AST 对照；`backend/source-comparison.json` 与插件内 `hcu_backend/source-provenance.json` 分开记录原始哈希、导入路由、来源缺口和有意修正。旧后端成员 Hook 已退出，避免 class REPLACE 后再由旧方法覆盖。新增状态与 API 后 HCU 严格目标集合为 363 项。

新增十个 ForwardBatch 字段采用继承官方记录的 dataclass 声明，保留原有方法，使构造、复制及 pickle 保留新增状态。ServerArgs 继续使用固定官方 schema；`--enable-cp-cache-layer-split`（别名 `--hcu-cp-cache-layer-split`）和 `--disable-cp-cache-layer-split` 转为进程选项并由 worker 继承。CLI 解析及完整 ModelConfig 派生分别验证，不将仅解析成功当作运行时配置完整。

正式启用检查复用当前 HCU LayerSplit 约束：DeepSeek V4.1、compression ratio 0/1/2、prefill PD、interleave CP、CP>1、PP1、Mooncake 与 Compressor V2，且 prefill Graph disabled。HiSparse/HiCache、encoder SWA replay、单独 staging transport、Q8 sparse prefill 和不支持的 speculative 组合仍被拒绝。这些是来源的支持边界，不为通过测试删除约束。

本阶段复现并处理以下数据接口问题：

- Torch 候选源在宽度 259 时给出未对齐 FP32 行，原生 selector 拒绝。对齐 carrier 仅在该 caller 准备，低层 selector 的非法 dtype/offset/stride 拒绝保持不变；三种布局的变输入 Graph 与独立 mask/ID oracle 验证实际 copy 和只读合同。
- legacy/interleave 的 Python list/tuple 被当成 Tensor 使用 range 高级索引；interleave 四 rank 的反例均失败，修复为 slice 后通过，Tensor 路径保持来源行为。

BF16 KV 对照还复现旧 compressor Hook 读取环境变量的接口错误：代码导入 `environ` 模块，却按 `envs` 对象访问 descriptor，旧单测又向错误模块添加虚构属性。Hook 改为读取实际 `envs`，两项回归改为注册实际 descriptor 并通过环境变量驱动分支；CPU 两项通过后继续执行真实 BF16 设备复验，见 [配置对象案例](development-methodology.md#22-环境变量对象与测试替身的接口一致性)。

输入 HCU 源码还保留 graph selector 对缺失 `glm5_next` 模块的无条件 HCU 导入，节点工作区与固定 HEAD 均确认文件不存在。候选移除该失效分支并保留可用模型的官方选择行为；该条作为来源缺口记录，不声明迁移了 GLM5Next HCU 优化。

真实 GPU 层链路采用实际 pool、SWA allocator、声明的 ForwardBatch、公共 metadata 初始化与两层随机权重 MQA。验证包括 ratio 1、ratio 2、264-token 候选源/消费者，以及 1030-token 跨页 top-k（从 1025 个历史位置选择 512 项）。确认命中原生 dense logits、候选 logits 与 paged sparse consumer；原生评分对同一 packed K 的独立 FP32 运算，输出对 FP32 评分的参考路径，已执行场景最终输出最大差为 0。该参考不构成独立 FP32 attention 算法或模型准确率验证。

真实四 rank 使用公开模型并行初始化，新增第二个 CP communicator 沿来源分区创建；验证 equal/variable all-to-all、peer 零计数、整 rank 空输入/空接收、两个通信器的双 stream 归约及销毁。`backend/collectives-tp4/rank-*.json` 保存逐 rank 结果；不是模型引擎或 RDMA 传输验收。

| 验证 | 源码形态 | 内嵌形态 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU 基础与合约 | 最后完整回归 425 passed / 1 skipped，147 subtests；配置对象 2 passed；池视图相关 37 passed | 65 passed，9 subtests | `backend/full-cpu-final.log`、`backend/bf16-view-after.log`、`backend/embedded-contracts-final.log` |
| 严格加载 | 363/363 passed | 363/363 passed | `backend/runtime-final.json` |
| GPU / Graph | 37 passed / 1 skipped，13 subtests | 待记录 | `backend/gpu-regression-attempt1.log` |
| 实际 allocator / metadata / MQA | ratio 1/2、候选/paged consumer、跨页 top-k 均通过 | 待记录 | `backend/gpu-public-*.json` |
| CP4 独立通信器与交换 | 四 rank 均通过 | 待记录 | `backend/collectives-tp4/rank-*.json` |
| CP4 完整后端/层链路 | FP8；显式匹配批次参考，四 rank 最大差 0 | BF16；四 rank 最大差 0，实际来源为内嵌目录 | `backend/cp-mqa-tp4-*` |

新增的完整模型配置与四卡层探针暴露了合成 HF fixture 缺失 `sliding_window` / `index_head_dim`；补齐实际 ModelConfig 所需字段后再验证，不修改生产代码绕过配置派生。原始失败与每轮后续结果均保留。

设备验证前按已有授权释放了 `debug_lzg` 在物理 4–7 卡的四个调优进程，核对 cmdline、容器与进程 start-time 后发送 SIGTERM；保留该容器在前四卡的服务。`backend/stopped-tuners.json` 和设备/PID 快照记录实际范围，未将整容器停止记作执行结果。

本阶段仍未宣称完整 decoder/model loop、DSpark streaming source-event/ACK、W4A8/DeepEP、AOT FP4 或完整 HCU fork 等价。最终集成后仍须生成正式输出并重验原 MiniMax 配方。当前使用方法按 [插件开发指南](plugins.md#使用-skill-开发插件能力) 调用开发 Skill，选择 [前向记录](development-methodology.md#19-前向批次-schema-与复制生命周期)、[评分行对齐](development-methodology.md#20-候选评分的行对齐与消费边界) 或 [序列分片](development-methodology.md#21-python-序列分片与-tensor-高级索引) 对应回归，历史案例不构成全部任务的固定检查清单。

候选后端集成提交为 `19ec9e554a0cc27ab112bc143ba205f278692f46`。本次生成包含 532 个插件文件、538 项清单文件，逐项哈希核验通过；生成前后两个来源的索引字节哈希保持不变。wheel 的 532 个插件文件逐字节相符；尚未执行隔离 wheel 运行验收。

CP4 与单卡原生完整批次的对照仍未通过原数值阈值（`rtol=0.02`、`atol=0.003`）：四层最大绝对差约 0.01733。首层 Q 与单卡输入相同，首层注意力输出已有差异；该结果不能写成完整 CP 模型验收通过。关闭参考 compressor 融合后偏差未消失，继续检查实际原生输入与独立数学参考。相关原始失败保存在 `backend/cp-mqa-tp4-*.log`。

网络恢复后，后四卡被其他用户容器的 TP8 MiniMax 服务占用。按已有授权尝试停止容器，但节点权限以创建者不是当前执行用户而拒绝，未绕过权限或记录为停止成功。随后诊断在每卡约 7.6–8.0 GB 剩余显存下执行，探针设置约 3.9 GB 的 PyTorch 单进程分配上限；这不是设备空闲或独占的声明，也不构成性能测试。完整服务复验与需要较大显存的 GPU 回归仍需满足资源条件。

首层原生输入捕获与独立对照已完成：15 个查询的 Q、解码 KV 集合和 FP32 数学输出一致（`backend/cp-native-input-comparison.json`）。单卡重放相同输入时，完整批次与 CP 批次形状原生输出差为 0.0078125，按相同批次形状执行则与 CP 输出差为 0（`backend/native-shape-replay.json`）。验证器新增显式 `--match-reference-native-batch`，用于隔离这一因素；完整批次结果与匹配批次结果分别记录，未修改原阈值或生产注意力实现。见 [数值参考案例](development-methodology.md#23-原生计算批次形状与数值参考)。

BF16 设备复验在修复配置读取后进一步暴露 SWA 视图的单位错误：父池 getter 继承了来源中 packed FP8 的 584 字节宽度，但实际 BF16 分配的最后轴应为 512 个元素。修复前，真实父池及四个 CP rank 的五项 BF16 consumer view 均失败，五项 FP8 对照通过；修复后后端与父池合计 37 项 CPU 测试通过。该来源修正保留 FP8 和 uniform-FP8 行为，单独保存为 `backend/bf16-view-source-comparison.json`，不纳入 42 个后端模块原样映射的声明。修复提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，前一个验证器提交为 `6db84b4f27ce0c1da1a4068c6d3633f724b02032`。

最终内嵌 BF16 四卡复验通过：`backend/embedded-cp-mqa-tp4-bf16/rank-*.json` 四项均为 passed，四层 Q、注意力输出和最终层输出与参考逐元素一致，最大差为 0；实际 SGLang/plugin 路径均位于候选内嵌目录。使用显式匹配原生批次形状及 Torch compressor 参考；本次 BF16 逐层完整原生批次与匹配批次的差也均为 0。该结果限定于真实 allocator、metadata、CP 池与小型随机权重 MQA 层链路，不是完整模型引擎或 PD 服务验收。内嵌 CPU 最终回归为 65 passed、9 subtests；源池相关回归为 37 passed。

本检查点最终候选为 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，由该干净提交重新生成内嵌源码并重建 wheel。方法库仅保存文档、Skill 和工具，未放入运行时包；原 v0.3.1 主输出与其 448 项内嵌清单继续保留。候选尚未替换主输出，完整 MiniMax 配方的新候选服务复验仍需可用设备，历史通过结果不继承为本候选通过。

### 2026-10-26 8. 最终交付与需求验收

2026-10-27 在历史验证容器完成本轮验证，并将已验证提交同步到正式输出。运行时代码继续位于方法库之外；方法库的 HEAD 和暂存区保留原状，文档及 Skill 更新保存在工作区。

| 产物 | 正式位置（工作根目录下） | 来源与状态 |
|---|---|---|
| 源码插件 | `output-code/sglang-plugin` | v0.3.2.dev5，干净提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c` |
| 内嵌源码 | `output-code/sglang-das` | official `baa9befa381511cd4d56f83980951e5a45fffc6d` + 上述插件；538 项清单、532 个插件文件逐项核验 |
| wheel | `output-code/dist/sglang_dcu_plugin-0.3.2.dev5-py3-none-any.whl` | SHA256 `c2260d30015e932faf33c09082758dee675113785353717a2750f75775544fac`；独立 CPU 安装/加载已验证 |
| v0.3.1 回退 | 插件分支 `validated/v0.3.1-before-v3-1026`；`output-code/.sglang-das-v0.3.1-validated-backup` | 旧插件 `39038d5` 和原 448 项内嵌文件保留 |

#### 2026-10-26 8.1 最终验证矩阵

下列日志相对 `artifacts/v3-20261026/backend/`。服务原始响应分别位于上级 `backend-source-service-attempt3/` 与 `backend-embedded-service-attempt2/`。

| 验证 | 源码插件 | 内嵌源码 | wheel |
|---|---|---|---|
| 完整 CPU 合约 | 435 passed / 1 skipped，147 subtests | 435 passed / 1 skipped，147 subtests | 435 passed / 1 skipped，147 subtests |
| 严格加载 | 363/363，三个具名 seam | 363/363，屏蔽外部入口 | 363/363，实际模块来自独立安装目录 |
| GPU / Graph | 37 passed / 1 skipped，13 subtests | 37 passed / 1 skipped，13 subtests | 未执行 |
| TP4 native all-reduce Graph | 四 rank × 两套捕获分配 × 三轮重放，通过 | 同范围通过 | 未执行 |
| MiniMax TP4 服务 | 自有 health 200；最终答案 `2`、`2`、`Paris`；正常 stop；清理通过 | 同范围通过 | 未执行 |

CPU 日志为 `final-source-cpu.log`、`final-embedded-cpu.log`、`final-wheel-cpu.log`；GPU 为 `final-source-gpu.log`、`final-embedded-gpu.log`；通信为 `final-*-collective-graph/rank-*.json`。部分 AST 抽取用例读取外部测试仓源码，因此独立加载、启动隔离和实际模块路径另行核对，不由总测试数推断产物来源。

服务仍使用原 `minimax-debug/d.sh` 的用户授权派生配方，仅将 TP8 改 TP4、NUMA 改为 `2 2 3 3`，物理卡为 4–7。保留 FA3、FP8 权重/KV、Graph、context 131072、chunk/prefill 等参数。每轮在待测目录下验证新进程来源，识别自有监听者后调用真实生成并按完整最终答案验收；前导 think 只按本轮明确允许的闭合前缀处理。重复请求不作为 cache hit 证据，也不作性能结论。

服务验证实际执行于候选目录；正式输出具有相同提交与 532 个相同插件文件，并在正式目录另做严格加载和模块来源验证。最终来源报告为 `delivery-source-runtime.json`、`delivery-embedded-runtime.json`；wheel 为 `final-wheel-runtime.json`、`final-wheel-startup.json`。

#### 2026-10-26 8.2 HCU 逻辑对照结论

`delivery-fork-logic-comparison.json` 比较正式内嵌核心与固定 HCU 当前树，列出 1208 个内容不同文件、387 个 HCU 独有文件及 20 个内嵌核心独有文件；这些计数是审阅清单，不直接代表行为相同或不同。生成器以官方源码为核心，HCU 行为经插件注册接入。

532 个插件文件与生成树相同，284 项当前 JIT 来源资源核验通过；完整后端的 42 个模块、482 个顶层节点、9 个桥接节点及十个前向字段有声明映射对照，BF16 SWA 维度修正单独记录。GLM 缺失模块导入、Python 序列切片、候选评分行对齐、FP8 原生载体和 BF16 元素维度均是明确的适配/修正。

**结论：明确迁移的节点及已执行契约有来源和行为证据；没有宣称整个 HCU fork 全逻辑等价。** 完整 DeepSeek 模型引擎、全部 decoder/DSpark streaming source-event/ACK、W4A8/DeepEP shared-expert、AOT FP4 构建导出及所有原生分支仍未形成全范围验收。CP4 的原生完整批次数值差异与显式匹配批次对照分别保留，详见 §7.6，不将对照选项解释为生产精度修复。

#### 2026-10-26 8.3 需求与证据对应

| 原需求 | 实际交付与证据 |
|---|---|
| 1. 使用给定社区/HCU 完善插件 | 正式源码提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`；各模块及修复见 §2、§7 |
| 2. official/plugin 两输入回溯 | 正式内嵌目录、来源 manifest 与 538 项文件核验 |
| 3. 空闲 GPU 测两种形态 | 指定容器内两份 GPU/Graph 与 TP4 native collective 通过 |
| 4. 原 d.sh 服务 | 两份本轮服务报告、配方哈希、三次生成和自有进程清理 |
| 5. 记录已完成功能模块 | 本文各阶段实现表、当前提交与支持边界 |
| 6. 比较 HCU 逻辑 | 正式核心差异清单、迁移节点映射及明确差异/未验证范围 |
| 6. 迭代 Skill 与文档 | [经验索引](development-methodology.md)、开发 Skill 的状态/原生/数值/来源测试契约及操作文档 |

最终逐项检查保存在 `backend/requirements-completion-audit.json`，包含来源保护、正式输出、测试、服务、文档与未覆盖边界。上述结论限定于本轮固定输入和执行配置；公共 PyPI 发布、TP8、全模型精度和性能不在通过声明内。

## 2026-10-27 V3 方法库导览与统一任务入口实施记录

> **本文用途**：记录 2026-10-27 方法库导航、统一 Skill 和 README 的建设范围、验证方法与限制。此次修改仅涉及方法库，不修改外部插件或内嵌运行时代码。

### 2026-10-27 1. 需求与交付对应

| 需求 | 实现 | 使用入口 |
|---|---|---|
| 修复网页导航与界面使用问题 | 离线文档阅读器、内部文档及章节路由、全文检索与类型筛选、搜索空态、请求复制反馈、窄屏导航状态；区分方法库和外部运行时 | [architecture.html](../architecture.html) |
| 新增总 Skill 检索并管理现有方法 | 新增统一入口，先识别意图并检索当前目录，再选择专项 Skill；利用已有信息，必要时补问，并在输入齐备后继续已授权任务 | [hcu-plugin-assistant](../skills/hcu-plugin-assistant/SKILL.md) |
| 从使用者角度完善 README | 补全文件树、功能及场景、简化调用方式，加入开发、安装和排障案例，说明人工使用、路径配置、维护检查及运行时边界 | [根 README](../README.md) |

统一入口是 Agent 的任务执行指引，不是单独运行的调度服务。它通过读取专项 `SKILL.md` 衔接执行；无需先安装全部 Skill 或修改客户端全局配置。总数为 1 个统一入口和 7 个专项 Skill。

### 2026-10-27 2. 网页构建与维护

网页的布局和交互维护在 [源模板](../templates/architecture.html)，由 [build_guide.py](../tools/build_guide.py) 读取当前共享文本构建。输出只写入 `architecture.html`；运行 `--check` 时不写入，快照过期返回非零。模板与内容不手工双份维护。

页面内嵌文档、Skill、参考和方法工具的阅读内容及检索信息，不依赖外部字体、脚本、CDN 或浏览时的目录请求。操作指南之间的链接转换为阅读器路由，源文件另有明确入口。外部网页链接保留为用户主动访问的链接。相对路径不得越出方法库，代码与原文按文本转义，不能关闭内嵌数据脚本或执行原文中的 HTML。

阅读器支持当前资料使用的标题、链接、代码块、列表和表格；Mermaid 图表以可展开源码展示，图形查看可使用支持 Mermaid 的 Markdown 编辑器。它不声称实现完整的 GitHub Markdown 渲染器。

新增构建工具后，本库共有 10 个方法工具。[runner 配方](../.workflow/l0-gate.yml)、[CI 说明](ci.md) 和 [仓库约定](../AGENTS.md) 已同步快照维护规则。实际平台自动 CI 接入状态保持原有范围。

### 2026-10-27 3. 统一入口的任务选择与补问

统一入口先枚举实际存在的 `skills/*/SKILL.md`，读取文档索引，再按目标、症状、组件和版本词搜索。候选表是初筛提示；实际文件的描述与适用范围是判断依据。根据阶段选择必要输入，避免将所有任务字段一次交给使用者填写。

| 场景 | 要求的处理方式 |
|---|---|
| 只问“怎样开发新插件” | 提供初始化指南和最小案例，不因缺少机器路径而阻断方法咨询 |
| 要求实际创建工程，尚未提供写入位置 | 选择初始化 Skill，询问外部新目录；等待时仍可解释结构、检查已有官方输入 |
| 已提供路径、日志或版本 | 优先读取和核实，不重复索取可得到的信息 |
| pip 或入口发现异常 | 选择安装 Skill，先核对解释器、metadata 和模块来源 |
| 已加载后的 Graph、TP 或数值回归 | 选择诊断 Skill，检索相似经验并定位本次首次分叉 |
| 要求生成 embedded 并对照 HCU | 选择回溯 Skill；HCU 作为独立比较来源，不伪装为生成器的第三输入 |
| 一个请求包含多项交付 | 按真实依赖衔接，不强制顺序执行全部专项 Skill |
| 找不到匹配入口 | 说明候选与差异，询问期望产物，不编造 Skill 或擅自扩展范围 |

上述行为完成了指引和调用资料的一致性审阅。此次没有把这些案例当作已执行的外部工程创建、安装或模型服务任务；Skill 元信息校验也不证明任意 Agent 都能正确理解所有自然语言请求。

### 2026-10-27 4. 验证与证据边界

| 检查 | 本轮结果及范围 |
|---|---|
| L0 | 通过：18 个 Python 文件、8 个 Skill；结构、资源链接和方法库阶段边界 |
| 工具契约测试 | 53 项通过；新增网页构建用例覆盖离线内链、代码与数据转义、表格代码中的管道符、目录边界和只读快照检查 |
| Skill 元信息 | 8 个入口通过 `quick_validate.py`；新入口与对应 UI 提示一致 |
| 文档与网页内链 | 文件、标题锚点、生成阅读器路由及内嵌原文一致性检查通过；计数以证据 JSON 为准 |
| 网页交互 | 无网络 JSDOM 测试通过：任务与章节跳转、跨文档链接、历史返回、错误路由恢复、检索及筛选、复制成功和拒绝反馈、窄屏菜单与 Escape |
| 响应式检查 | 检查样式规则及 390 / 1440 宽度下的 DOM 菜单状态；未测量真实浏览器布局几何 |
| 实际浏览器与截图 | 未执行：内置浏览器安全策略拒绝 `file://` 页面打开；未换用其他浏览器或方式绕过拒绝 |
| 外部插件、GPU 与模型服务 | 本轮未执行，不引用历史运行结果充当本次验证 |

证据位于方法库 ignored 目录 `artifacts/v3-20261027/`：`local-validation.json`、`guide-qa.json`、文档检查结果、远端检查日志及最终需求验收报告。ignored 证据需另行保管，不随 clone 自动分发。

新导航、Skill、文档和工具已落入方法库；原有未提交改动按原状保留。本轮不会自动提交或推送现有工作区。

## 2026-10-28 运行时代码迁入与入口验证

## 2. 迁移与提交台账

| 日期 / 步骤 | 完成内容 | 提交记录 | 验证与边界 |
|---|---|---|---|
| 2026-10-28 / M01 运行时代码迁入 | 525 个来源文件；180 个 Python 文件仅改命名空间；239 个源 Python 文件 AST 对照；286 个原生资源/许可证按原字节保留 | [`4d82a16`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/4d82a1663a54c5d4075c5215acc2d91dcfc66750) | 本次静态迁入对照通过；不把旧 GPU/模型结果记为新包验证 |
| 2026-10-28 / M02 安装与启动入口 | 本仓打包定义、`hcu` 入口、显式 launcher、命名空间可选严格加载工具、阶段门禁及当前使用文档 | 同一代码提交 `4d82a16` | 65 项仓库测试、源码/wheel 各 363/363 项 Hook 和 3 个具名接口通过；完整服务未执行 |
| 2026-10-28 / M03 HCU 发现入口 | entry point 改为 `hcu = hcu_sglang:register`；版本 `0.3.2.dev7`；同步当前安装与开发说明 | [`443eb83`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/443eb83eb56d91ec5a86c1c53989419fb0077667) | 源码/wheel 各 363/363 项 Hook、3 个具名接口；旧 `hcu` 名称隔离通过 |

代码提交：`4d82a1663a54c5d4075c5215acc2d91dcfc66750`，标题 `feat: import HCU runtime as hcu_sglang with isolated packaging`，共 557 个变更文件。范围包括 526 个运行时包文件、根打包定义、迁入清单、当前使用文档与 Skill、结构门禁和运行验证工具/测试；未迁入输出仓的 docs、scripts 或 tests。其后提交 `c8ba45a` 仅回填该代码 SHA 并重建导览。

当前入口在 M03 改为 `hcu`；M01/M02 中的 `hcu` 与 dev6 验证结果保留历史含义。

当前代码来源是已存在实现的迁入，不是宣称整个 DAS fork 已完成转换。模型能力、Native、CP/EP/PD 的历史范围仍以对应来源记录为线索，本命名空间必须建立自己的产物与运行证据。

## 4. 实际验证记录

### M01/M02：首次迁入（dev6 / hcu）

验证日期：2026-10-28。实际环境为 Python 3.10.12 与既有 HCU/DTK 依赖容器；官方源码固定在上述 `baa9befa`。安装、构建和仓库测试使用源码所属用户；真实 Hook 加载使用容器原有的可访问设备的执行身份。没有修改设备权限或共享 site-packages。

| 检查 | 实际结果 | ignored 原始证据（相对仓库根） |
|---|---|---|
| 在线来源核验 | 525 个来源/目标 SHA256 一致；official、fork、来源 plugin 的 HEAD 与干净状态保持 | `artifacts/v3-20261028/provenance.json` |
| 结构及仓库测试 | L0：261 个 Python、8 个 Skill；65 项 unittest 通过 | `artifacts/v3-20261028/l0.log`、`tests.log` |
| 源码隔离安装与冷导入 | distribution `hcu-sglang-plugin==0.3.2.dev6`，唯一 `hcu = hcu_sglang:register`；冷导入未载入 Torch/AITER/旧包 | `artifacts/v3-20261028/source-install.log`、`source-cold-import.log` |
| 源码真实加载 | 363 项目标集合完全匹配且全部应用；3 个具名 resolution 接口；软错误与未应用列表为空；未导入旧包 | `artifacts/v3-20261028/source-hooks.json`、`source-hooks.log` |
| wheel 构建与完整性 | wheel 内 526 个包文件路径/字节与源码一致；未携带外部 docs/tools、旧包或全局 Python 启动文件 | `artifacts/v3-20261028/runtime-validation.json`、`build-wheel.log` |
| 隔离 wheel 安装与冷导入 | 从独立 target、仓库外 cwd 导入；唯一新入口；冷导入无 Torch/AITER/旧包 | `artifacts/v3-20261028/wheel-install.log`、`wheel-cold-import.log` |
| wheel 真实加载 | 未将仓库根加入 PYTHONPATH；实际导入来自 wheel target；363/363 项和 3 个具名接口通过，无加载软错误或旧包导入 | `artifacts/v3-20261028/wheel-hooks.json`、`wheel-hooks.log` |
| 显式 CLI 参数入口 | 源码和 wheel 的 `python3 -m hcu_sglang --help` 均 exit 0；只证明参数入口可用 | `artifacts/v3-20261028/source-cli.log`、`wheel-cli.log` |
| 文档与 Skill | 8 个 Skill 元信息、导览快照及链接/锚点校验通过；迁移计划保持原三方快照并链接当前记录 | `artifacts/v3-20261028/local-validation.json`、`guide-qa.json` |

本次 wheel 为 `hcu_sglang_plugin-0.3.2.dev6-py3-none-any.whl`，SHA256：`dc992c7a420148204cee1e0a9274ca590ea555189fb9bc58d74b606e1f720e8a`。它包含 Python 与原生源码资源，实际运行仍依赖兼容 vendor 库及工具链；没有发布公共 PyPI 版本。

复现构建和隔离 wheel 安装（`ARTIFACTS` 为本次独立可写目录）：

```bash
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir "$ARTIFACTS/wheels" .
python3 -m pip install --no-deps --no-build-isolation --target "$ARTIFACTS/wheel-site" \
  "$ARTIFACTS/wheels/hcu_sglang_plugin-0.3.2.dev6-py3-none-any.whl"
```

wheel 加载进程在仓库外执行，只将固定官方 `python/` 与本次 `wheel-site/` 加入 PYTHONPATH，并将严格验证工具的 `--plugin-root` 指向 `wheel-site/`。目标合同仍使用本仓迁入清单。

**环境排查记录**：首次用无容器 passwd 记录的普通 UID 验证时，Torch 默认缓存落到不可写目录，AITER 随后尝试写不可用的用户 JIT 目录。配置本次任务的 `XDG_CACHE_HOME`、`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`、`AITER_JIT_DIR` 后缓存问题消除；首次 AITER 原生依赖构建完成。随后普通 UID 报设备不可用，而容器原有执行身份可见 8 张设备，真实加载因此使用后者。失败尝试日志保存在 `artifacts/v3-20261028/failed-attempts/`，不能将编译耗时或 loader 捕获异常后的退出视为加载通过。

**未执行范围**：新命名空间的模型生成、GPU 数值/Graph/多卡服务及 embedded 产物未在本轮复验。真实 Hook 加载使用了设备环境，但不能代替这些验收；旧命名空间的历史结果仅作为后续回归线索。

### M03：HCU 插件发现入口（dev7 / hcu）

代码提交：`443eb83eb56d91ec5a86c1c53989419fb0077667`（`fix: discover HCU plugin through the hcu entry point`，13 个文件）。

将根打包元数据入口改为 `hcu = hcu_sglang:register`，开发版本更新为 `0.3.2.dev7`。运行时实现及 363 项目标合同不变；同步当前安装说明、开发指南和 Skill，保留旧产物的历史入口。实际执行环境沿用固定官方 `baa9befa` 与 Python 3.10.12 / HCU 容器。源码与 wheel 均从新的隔离 target 安装；构建/安装使用源码所属用户，设备加载使用容器原有执行身份。

| 检查 | 本次结果 | ignored 证据（目录 `artifacts/hcu-entrypoint-20261028/`） |
|---|---|---|
| 安装 metadata 与冷导入 | dev7 仅提供 `hcu → hcu_sglang:register`；源码及 wheel 冷导入均无 Torch/AITER/旧包 | `source-cold-import.log`、`wheel-cold-import.log` |
| `SGLANG_PLUGINS=hcu` 源码加载 | 363/363 目标全部应用、3 个具名接口，无加载软错误或旧包导入 | `source-hooks.json`、`source-hooks.log` |
| `SGLANG_PLUGINS=hcu` wheel 加载 | 从仓库外 cwd、隔离 wheel target 加载，363/363 和 3 个具名接口通过 | `wheel-hooks.json`、`wheel-hooks.log` |
| 旧名称隔离 | 新进程设置 `SGLANG_PLUGINS=hcu`，0 项 Hook，新包未导入 | `old-name-isolation.log` |
| wheel 完整性 | 526 个包文件路径及字节与源码一致，无旧包或全局启动文件 | `runtime-validation.json` |
| 仓库门禁 | L0 与 65 项 unittest 通过；两个更新的 Skill、链接/锚点和导览快照校验通过 | `l0.log`、`tests.log`、`guide-qa.json` |

本次 wheel：`hcu_sglang_plugin-0.3.2.dev7-py3-none-any.whl`，SHA256：`5d61be21697b03c9c5e21cf26c7ec74a161357f7af15e93b0bf2c2e911740ecd`。新版本更新安装元数据后采用 `export SGLANG_PLUGINS=hcu`；dev6 的 `hcu` 结果不作为 dev7 的实际加载证据。本轮未执行模型服务、数值或 Graph 复验。

构建首次使用设备执行身份时被 Git 所有权校验拒绝，已改用源码所属用户完成构建；保留失败日志，未修改全局 Git 信任配置。

## 6. 2026-10-29 分阶段继续迁移

输入保持 official `baa9befa`、DAS `15c7204b`；本轮起点为本仓 `405b8dc`。输入只读，不推送远程。原固定盘点含 3,010 个非 same 路径，其中运行时范围差异 1,614 个；本轮执行台账将全部非 same 路径纳入，未审核项保持 pending，不能把 450 条历史映射线索当作当前行为覆盖。

| 阶段 | 本轮实施 | 退出条件与当前边界 |
|---|---|---|
| M0 基线与迁移工具 | 当前/旧包识别；清单和处置目标发现；抽取/签名 CLI；v4 embedded；全差异执行台账 | 离线契约与新包隔离加载夹具；当前类 profile 仍可能拒绝漂移 |
| M1 基础与量化 | 延续既有实现，逐项处置尚未完成 | shape/dtype/依赖/回退行为验证后更新人工决定 |
| M2 Attention/KV/模型 | 延续既有实现，逐项处置尚未完成 | 构造、状态与消费者、数值、Graph 和指定模型分别验收 |
| M3 通信/PD/扩展 | 新增 HCU StandardDispatcher 哨兵映射修复 | 公共 dispatcher CPU 回归；真实 EP 多进程、GPU Graph 与模型仍待验收 |
| M4 产物与交付 | 当前包 embedded 生成与核验能力 | 新包实际源码/wheel/embedded 严格加载分别记录；不代表全 DAS 等价 |

### StandardDispatcher 来源与行为

来源 `python/sglang/srt/layers/moe/token_dispatcher/standard.py` 的 helper 与 dispatch 方法迁入 `hcu_sglang/layers/moe/standard_dispatcher.py`。只把官方模块全局名显式限定为 `S.*`，保留来源注释；新增 HCU AROUND 门控与顶层注册。非 HCU 交还传入 original，保留前序包装。没有从 DAS 动态导入。

原始反例：映射 `[0, 1, -1, -1, 2]` 下，输入 `-1` 被 Python 负索引映射为共享专家 `2`；修复后负数与超出全局表的 ID 均保持 `-1`。正常 routed/shared IDs、DP pad 的后置屏蔽、空 batch 和跳过映射分别验证。当前完整目标集合为 [当前合同](hcu-sglang-development.md#历史-json-快照)（`hcu-sglang-targets.json`）；原迁入 manifest 的 363 项保持历史快照。

运行测试要求固定 official python 与本仓在 PYTHONPATH，使用已有兼容 HCU/Torch 容器，从仓库根执行：

```bash
python3 tests/runtime/standard_dispatcher.py
python3 tools/build_migration_execution.py --check
python3 tools/build_guide.py --check
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

本次原始证据在 `artifacts/v3-20261029/`。逐路径台账的 implemented 只表示对应文件范围的代码接入，不是全领域完成。GLM5-next、Muse Spark、各 KV/Graph/native 分支等仍需逐项迁移与完整验收；本轮不宣称全量 DAS 转换完成。

### 本轮本地提交与实际检查

本地工具提交 `af5c635`：当前包识别、清单/处置/抽取/签名和 v4 内嵌工具，含旧包兼容与新包隔离夹具。运行时提交 `42fe693`：StandardDispatcher 修复、当前 364 项 Hook 合同、公共 dispatcher 回归和全差异执行台账。提交均未推送。

| 检查 | 实际结果 | 本轮证据 |
|---|---|---|
| L0 / 离线契约 | 267 个 Python、8 个 Skill；70 项 unittest 通过 | `l0.log`、`tests.log` |
| 真实公共 dispatcher | 4 项 CPU 张量测试通过，含原行为反例、EP 映射与共享专家、DP pad、空 batch 和跳过映射 | `dispatcher.log` |
| 源码严格加载 | 364/364 项应用、3 个具名 seam、无加载软错误与旧包导入；集合与当前合同一致 | `source-hooks.json`、`source-hooks.log` |
| 隔离 wheel 加载 | 仓库外 cwd、独立 wheel-site；364/364 与 3 个 seam 通过 | `wheel-hooks.json`、`wheel-hooks.log` |
| wheel 构建 | 本地 dev7 产物 SHA256 `d4ed63db35ee5ad9ef42a058e1bde39205bed60dc254c3b314d002c5abec3fea`，与历史同版本产物按 SHA 区分 | `wheel-build.log`、`artifact-checks.json` |
| Skill 与网页 | 3 个修改 Skill 的 validator 通过；重建导览、实际桌面和 390px 窄屏阅读器与导航开关检查 | 本次浏览器观察；快照门禁日志 |

测试环境为已有 HCU/DTK 容器、Python 3.10.12、Torch 2.11.0；公共 dispatcher 测试使用 CPU 张量并只控制设备/并行环境及昂贵通信条件，保留真实官方构造和 dispatch。节点宿主 Python 3.6、本机 Python 3.9 均不作为本项目验证解释器。首次未安装 AITER bootstrap 的测试导入失败，加入与 launcher 相同的兼容启动后通过；失败日志不能当作通过证据。


### 本轮实际 embedded 复验

从干净本地提交 `08b027e3228ace34537a5cf4a3e334f00ef9b747` 与固定官方 `baa9befa` 生成独立 `embedded-hcu` 源码树，迁入 527 个包文件；532 项 manifest 哈希全部通过。禁用所有外部 entry point、仓库外 cwd，仅加入产物 `python/` 后，364/364 项 Hook 和 3 个具名 seam 均通过，官方与插件模块均来自该产物，无旧包导入。证据为 `embedded-build.log`、`embedded-hashes.json`、`embedded-hooks.json` 和 `embedded-hooks.log`。产物位于本任务新建的 sibling `tmp/hcu-embedded-v3-20261029/`，不是原始 DAS fork 的逐文件复刻，也未执行模型/Graph/多进程 EP。

当前来源的方法同步预检明确失败在 `DeepseekV4DecoderLayer`：DAS 当前提供 Base/Cuda/Platform 拆分类。工具退出 2、不写任何输出；本次通过包名适配不等于该旧 profile 可直接执行。下一步须审查现有 `current_v4` 接入与平台类继承、状态和消费者，再决定删除旧抽取目标或补充精确 profile，不能将类名机械替换后覆盖既有适配。原始诊断见 `method-preflight.log`。

本轮结束时 official/DAS 两输入 HEAD 与起点相同、工作区干净。全量迁移总任务保持 in-progress；台账剩余 3,009 个未审核路径不等于 3,009 项缺失实现。

## 7. 2026-10-29 模型服务测试中的 spawn 启动修复

本轮以 cookbook 的 SGLang 0.5.18 单机八卡配方为参考，在已有 HCU 容器中运行本仓插件与固定官方 0.5.22 源码；它不是在容器预装 DAS 0.5.18 上加载插件的兼容声明。用户选定代表模型 Qwen3、MiniMax、GLM/DeepSeek，原始请求、响应、配方和运行记录保存在用户指定测试工作目录。

首次 Qwen3-235B-A22B-Instruct-2507 TP8 服务在加载权重前退出：父进程虽有 AITER shim，`runpy` 将主模块身份换成官方 launcher，spawn worker 在反序列化 scheduler 时先导入 quantization，触发旧 AITER 缺少可选 `gemm.fused` API。此前仅父进程的 Hook apply 不能覆盖此路径。

新增 `hcu_sglang.launch_server` 作为 spawn 可重入入口，在模块导入时仅安装惰性 shim；包 CLI/console 入口将主模块设为该入口，再以不替换主模块身份的方式执行官方 launcher。worker 在解析 scheduler 目标前先恢复相同 bootstrap，不安装全局 sitecustomize/usercustomize，不伪造真实 Quark kernel；真调用缺失 kernel 仍抛出原有 NotImplementedError。

版本更新为 `0.3.2.dev8`。真实 spawn 隔离夹具在旧 dev7 上失败、在修复后通过；71 项 unittest 和 L0（268 个 Python、8 个 Skill）通过。dev8 wheel 含 528 个包文件，逐路径/字节与源码相同，SHA256 `457e064d8d7a57413038ca453330604ef0ede781f79be33f6221eadb8d32f5b5`。同一 Qwen3 TP8 配方复跑已越过原失败点并进入八 rank 权重加载；最终生成结果在后续记录中补充。

测试工作目录内证据：`logs/spawn-regression-original.log`、`logs/spawn-regression-fixed.log`、`logs/repository-tests.log`、`logs/l0.log`、`artifact-dev8.json` 及 `runs/qwen3-235b-instruct-tp8-attempt2/`、`runs/qwen3-235b-instruct-tp8-attempt4/`。前者保留原始启动失败；后者为修复后的独立运行。

## 8. 2026-10-29 量化 CLI 与 forward-context KV 归属修复

GLM-5 INT4 W4A8 的 cookbook 配方首先在参数解析阶段被拒绝：运行时量化 registry 已有 `slimquant_w4a8` 和 `slimquant_w4a8_marlin`，CLI choices 却只加入 `w4a8_int8`。`server_args.py` 现在从同一 registry 常量导出全部三个名称；保持原有选择、幂等性和量化实现的惰性加载。真实 CLI help 已包含这些名称，离线回归检查实际 argparse 接受与不提前导入 Torch/量化实现。

随后用空闲四卡的小型 Qwen3 诊断完整 forward，确认 Torch/HIP 当前设备与张量设备一致，并在 Graph warmup 复现 `ForwardBatch` 不再拥有 `token_to_kv_pool` 的异常。固定官方在 `model_executor.forward_context.get_token_to_kv_pool()` 从当前 backend 解析 KV 池；共享 fused rotary、Qwen2 融合路径和模型公共融合判定均改用此接口，不补写旧字段、不缓存单一全局池。

真实 forward-context 回归在旧 dev9 wheel 上失败、在修复源码上通过：无旧 pool 字段的 batch，在外层/内层/恢复后的 context 中分别写入对应 KV 张量并保留融合 store 标记。昂贵的融合数学 kernel 被替换成可观察的 CPU 写入，官方 context 与实际池选择未 mock。证据为测试工作目录 `logs/kv-context-original.log`、`logs/kv-context-fixed.log` 与 `tests/runtime/fused_kv_context.py`。

当前开发版本为 `0.3.2.dev10`，wheel SHA256 `239e573e98ed5b159d84534a1bccf4ce18d06ef0e550c67bd05034281da71146`。CLI 修复后的 72 项离线 unittest 通过；模型结果另行记录，不将解析或 CPU 写入测试当作整模型精度通过。

## 9. 2026-10-29 四卡代表模型复验与融合门控修复

用户在 GPU 0–3 被其他服务占用后授权先验证四卡服务；使用物理 GPU 4–7、独立 ROCR 可见映射，保留其他服务。Qwen3-30B-A3B-Instruct-2507 BF16 TP4 在 dev10 中完成权重加载，越过 init_cublas 后于 Graph warmup 的模型公共 helper 导入处失败。原始证据为测试工作目录 runs/qwen3-30b-instruct-tp4-attempt1/；不是精度通过。

按失败路径对照固定 DAS models/utils.py，现有迁入 helper 使用已删除的 is_prefill_context_parallel_enabled，且通配导入不会导入私有 _is_cuda/_is_hip 标志。当前改为显式导入和官方 layers.cp.utils.is_cp_active(forward_batch)，保留 DAS 对 HCU 的 CUDA/BF16/非 SWA 门控、当前 CP 与 dcp_kv_mask 限制，其他 HIP 平台沿当前官方门控；池仍由真实 forward context 提供。不运行 DAS 对照服务，不动态导入 DAS 实现。

开发版本 dev11；新增 tests/runtime/model_utils_fusion.py 验证真实 helper 导入、forward-context 池、当前 CP 分支、DCP mask、SWA/dtype 和非 HCU 路径。修复前反例、修复后检查、wheel 与模型复验结果见下文；CPU 和打包检查不代表模型通过。

实际检查：dev10 wheel 在 logs/model-utils-original-attempt2.log 复现已删除 API 的 ImportError；修复后 logs/model-utils-fixed.log 的 3 项真实 helper/forward-context CPU 回归通过。72 项离线 unittest 与 L0 通过（logs/repository-tests-dev11.log、logs/l0-dev11.log）。dev11 wheel SHA256 e7badb052105ea5f06be25f92249427bb2e6fdcfd8931615730352ce4b5e07be，528 个包文件与源码和隔离安装目录逐字节一致，见 artifact-dev11.json。

相同四卡 Qwen3 配方的 dev11 attempt2 在权重和 16K KV 缓存加载后，仍于未修改的 init_cublas 16x16 FP16 矩阵乘法触发 rocBLAS Tensile kernel 缺失；15:53:57 终止，已记录自有进程清理为空、端口无监听。此轮没有请求或模型精度通过；Graph 原失败点的完整模型复验仍 pending。固定 DAS 与官方 attention_backend_setup 文件相同、init_cublas 调用及辅助函数相同，未据此认定环境或插件为最终根因。

## 10. 2026-10-29 按触发路径对齐 DAS 的 AITER 通信逻辑

用户明确要求插件与 DAS 的实现路径对应，不以关闭优化路径代替修复。本轮新增 distributed_aiter.py，迁入固定 DAS custom_all_reduce.py 的三个函数：AITER 注册选项、工作空间大小与完整 dispatcher；现有 HCU dispatch Hook 懒加载该实现，非 HCU 保留传入 original。HCU auto 不再按旧 SGLANG_USE_AITER_AR 默认关闭 AITER，而沿 DAS 的确定性判断与 AITER 导入选择。显式后端参数在 helper 层保留；官方 CLI/GroupCoordinator 的新增后端字段及 strict-Fabric 组合尚未在此轮移植，不宣称该组合已支持。

迁入保留 GraphSafeAiterCustomAllreduce：Graph warmup 运行真实 all-reduce/fused AR RMS、传递 live residual，capture 与 piecewise 路径交还 AITER 父类；IPC 注册受 memory-saver 及 DeepSeek-V3/GLM 特定拓扑约束，Fabric/auto 使用 copy-in。保留 DAS 的环境值校验、显式 aiter 与确定性模式拒绝、auto 下仅 ImportError 回退；不吞 kernel 错误。

来源文件 SHA256 为 2fe4be4c81564948b96d283e66f326f6a37b46ccc13b29fde051c581eb7834f9。只做模块全局名限定、HCU 检测入口及相对导入适配；还原这些适配后三个函数的 AST 与 DAS 完全一致。生产运行不动态导入 DAS。静态核对、修复前后 CPU 回归、wheel 和服务结果分别记录；当前准备的开发版本 dev12 不代表模型或 GPU collective 已通过。

实际执行：dev11 在 logs/aiter-align-original.log 的默认 HCU 后端反例失败；修复后 logs/aiter-align-fixed.log 的 6 项针对性 CPU 回归、72 项仓库 unittest 和 L0（272 个 Python、8 个 Skill）通过。dev12 wheel SHA256 30fbb912c349aed615c8febc8631f4e765b873cec7be40df9e6046162cf8c758，529 个包文件与源码、隔离安装逐字节一致。Qwen3 四卡 attempt4 的四个 rank 均记录 AITER IPC、copy-in，越过 BLAS 初始化后在 Graph 暴露 fused_topk 旧签名不接受 num_fused_shared_experts；没有生成请求或模型精度通过。此过程同时经历节点重启及主机内存恢复，不能据此将先前 rocBLAS 错误单独归因为通信选择。

## 11. 2026-10-29 MoE top-k 当前调用契约与 DAS 分支对齐

Qwen3 TP4 dev12 在 Graph warmup 经真实 select_experts 调用，因旧 REPLACE 的 fused_topk 缺少 num_fused_shared_experts 参数退出。固定 DAS 与官方 fused_topk 函数去除唯一 HCU LightOp 分支后 AST 完全一致；两输入文件哈希已与本次来源核对。插件改为 AROUND，完整保留官方的 11 项参数，非 HCU 及其他分支委托传入 original，保留已有包装。

仅在 HCU/softmax/原有 SGLANG_USE_FUSED_TOPK_SOFTMAX 开关启用、且不优先命中 AITER 分支时执行 DAS 的 LightOp topk_softmax。AITER 优先级、sigmoid、路由缩放、共享专家数、packed 输出与非 padding token 数交还固定官方的相同实现，不吞 kernel 错误，不用忽略新参数规避启动异常。版本 dev13；公共 HookRegistry 分发和 CPU 消费者回归、wheel、模型复验分别记录。

实际回归：dev12 wheel 在 logs/topk-original.log 通过真实 HookRegistry 包装的公共入口复现同一个 num_fused_shared_experts TypeError；修复后 logs/topk-fixed.log 的 6 项公共 Hook/CPU 消费者测试通过。72 项仓库 unittest、L0（273 个 Python、8 个 Skill）通过。dev13 wheel SHA256 8f7f9a0d69e6e96c2d180905f11932a4d9f799022c64d59d3d8491fda2b98b6e，529 个包文件与源码、独立安装一致。来源逻辑核对见测试目录 source-diff/topk-source-alignment.json。

模型复验：runs/qwen3-30b-instruct-tp4-aiter-attempt5/ 使用物理 GPU 4–7、官方 0.5.22、dev13 wheel、DAS 对应 AITER IPC 路径及正常 Graph，完成 batch 1/2/4/8 捕获。六个确定性请求严格 6/6 通过：323、北京、Paris、4、10、重复 323，全部 finish_reason=stop。17:31:25 完成，cleanup_remaining_pids 为空、端口无监听。仅代表该四卡短请求 profile，不外推为 235B 八卡、数据集精度、吞吐或全部 DAS 等价。

## 12. 2026-10-29 GLM/DeepSeek Decoder 退役旧替换并恢复当前 stage

GLM-5 INT4 TP4 在加载权重前被旧 Decoder 构造器拒绝 skip_rope。固定官方与 DAS 的 DeepseekV2DecoderLayer 全部方法已相同；当前实现使用 attn_boundary/ffn_boundary，旧插件仍使用 layer_communicator 且 forward 多出 residual 参数。按实际调用/消费边界成对退役该类的 __init__/forward Hook 和两份不再使用的旧实现，使用当前官方与 DAS 相同的真实类，不复制旧函数后仅添加参数。

当前 Hook 合同去掉这两个目标，变为 362 项；迁入 manifest 及历史 364 项验证保留原义。新增 tests/runtime/deepseek_decoder.py 经真实注册入口验证当前构造、skip_rope/build_stages、NextN NoPE、归约 owner 及 native forward 对 stage 字段的消费。版本 dev14；实际 CPU、wheel 和模型结果在执行后补充。

## 13. 2026-10-29 当前 DeepSeek V2 调用契约与 DAS 数据流同步

GLM-5 INT4 TP4 dev14 仍在加载权重前失败：旧 Attention 构造器给当前 ColumnParallelLinear 传入已删除的 tp_rank。逐项审查后，本轮从固定 DAS 同步现有 24 个注册方法，覆盖 MLP、MoE 与 MLA/MHA/ROCm/CPU attention；保留当前 parallel_group、skip_rope、输入量化载体、非 padding token、共享专家与归约 owner 的完整参数和消费路径。只限定实际 provider 的全局名、显式 super owner 和插件依赖导入；移除文档字符串缩进影响后，24 个方法的规范化 AST 均与 DAS 相同。旧全量同步 profile 仍因 V4 类拆分预检失败，本次没有绕过预检运行它。

DAS 新增但固定官方没有的 fused_rms_quant、GLM5-next 模型类型谓词与 MoE graph 输出桥接放入插件命名空间。运行不导入 DAS 源码。W4A8 线性层同步 supports_prequantized_input 和显式 INT8/FP32 输入检查，显式 q/scale 不再依赖无关 RMS 开关；保持原 GEMM 后端。该谓词不代表完整 GLM5-next attention 后端迁移已完成。

同时核实并修正共享 Qwen/Bailing fused rotary 门控：沿 DAS 的 SGLANG_USE_FUSED_RMS_ROTARY 默认关闭语义，Qwen3-MoE 保留原生融合分支优先级。已有 dev13 Qwen 6/6 属于旧版本结果；当前版本必须另行复验，不能将旧请求记作新版本通过。

开发版本 dev15。tests/runtime/current_v2_dataflow.py 检查量化载体经真实 prepare 到线性层的传递、真实 CPU 数学消费、非法载体拒绝、graph 共享输出所有权与 Qwen 分支门控；仅替换昂贵外部 kernel。实际 CPU/仓库门禁、隔离 wheel 与 GLM/DeepSeek 服务结果记录于测试目录，不以静态同步证明整模型精度。

实际检查：24 项规范化方法 AST 核对通过；5 项当前 V2/W4A8/graph/Qwen CPU 回归、72 项仓库 unittest 与 L0（276 个 Python、8 个 Skill）通过，证据为 current-v2-ast.json、logs/current-v2-dataflow-dev15.log、logs/repository-tests-dev15.log、logs/l0-dev15.log。dev14 GLM attempt2 保留原 ColumnParallelLinear tp_rank 异常；新模型启动与请求结果另记。MiniMax-M2.5 BF16 TP4 dev13 已完成 6/6 严格请求检查，属于该版本四卡短请求结果。

## 14. 2026-10-29 GLM EAGLE NextN 当前 stage 同步

GLM-5 INT4 TP4 dev15 四个 rank 已加载目标模型，在 EAGLE 草稿初始化触发旧 NextN 的 is_dsa_enable_prefill_cp NameError；原实现同时保留已删除的 Decoder CP 参数、residual 参数和三值返回。按固定 DAS 同步现有两项 NextN Hook，使用 layer_stack、skip_rope、当前 residual_batch 边界与两值 Decoder 返回，保持 IndexTopKShareState 更新/发布顺序和 SBO stream 选择，不关闭 EAGLE。

两项规范化方法 AST 与固定 DAS 相同；缺失的 legacy CP helper 复用插件现有 HCU backend provider，CP-v2 可用门控按 DAS HCU 默认关闭语义读取环境值，避免读取官方已废弃的 EnvBool registry 名称。通用 CP/GLM5-next 全路径尚未验证，不能从本轮 attn_cp_size=1 的配方外推。生产运行不导入 DAS 源码。版本 dev16，tests/runtime/nextn_current.py 通过真实公共 Hook 调用核对当前构造器契约、SBO stream 和 forward 的 residual/返回消费顺序；实际门禁、wheel 和模型结果另记。

## 15. 2026-10-29 DSA KV 池与 DSpark 解析路径同步

GLM dev16 目标/草稿加载越过 NextN 后，在 KV 池构造因 index_kpool 参数失败；DeepSeek-V4 dev16 首轮在解析阶段因原始 device=None 被旧包装器读取而失败。本轮分别沿触发边界对照固定 DAS，不运行 DAS 对照服务。

DSA 池迁入当前构造与缓存方法命名空间，34 项方法的规范化 AST 与 DAS 相同。现有 AROUND 转发所有参数，在 HCU 消费完整 index_kpool/compress、tail、request、DCP 和 indexer_layer_ids；非 HCU 委托原函数。缓存 facade 同步紧凑全局层映射，保留当前 FP8/BF16/INT8 模式选择、真实缓存读写/搬移和共享 workspace 所有权。INT8 helpers 保留 gfx936 opt-in 限制，读取环境值以适配官方未提供的新 EnvBool；本次 gfx938 FP8 配方不构成 gfx936 INT8 的整模型验证。DAS 的 _should_allocate_index_layer 在当前来源误带 property，但实际消费者调用该方法；保留官方同一函数体的 callable 绑定。facade 补齐通过现有 GetKAndS 消费的连续 K/scale 接口，避免当前 DAS 方法调用不存在的 facade 名称。新增 host_pool_decls Hook，当前合同变为 363 项，历史 362/364 项结果不改写。

DSpark 同步 DAS 的 _handle_dspark 与 PD/CP 资格判断，读取真实 resolving_view，沿 declare_resolution 写入决策，不再临时改写原始 device。保留 HCU DeepEP target/draft runner、静态 verify 与 CP/PD 限制。其他平台沿传入 original；运行不导入 DAS 源码。完整 PD/CP、INT8 GPU kernel 与长请求仍未验证。

开发版本 dev17。5 项 tests/runtime/dsa_dspark_current.py 通过真实公共 Hook 检查构造参数、page/压缩 tail、紧凑层读写、key/scale 搬移、非 HCU 参数保留和真实解析 overlay；仅替换外部 MLA 分配与昂贵 kernel/配置文件读取。证据为 current-dsa-audit.json、current-dspark-audit.json、logs/dsa-dspark-current-dev17.log。仓库门禁、独立 wheel 与模型复验另记。

已有 dev16 四卡复验：Qwen3-30B-A3B-Instruct-2507 attempt6、MiniMax-M2.5 BF16 attempt2 各 6/6 严格通过，Graph 保持开启，均清理自有进程且端口无监听；当前 GLM/DeepSeek 尚未通过，不将启动进展记作精度结果。

## 16. 2026-10-29 DSA HCU FlashMLA metadata 与 Graph 路径同步

GLM dev17 已加载目标/草稿并分配 KV，在 Graph 初始化误调用 CUDA sgl_kernel.flashmla_ops。固定 DAS 使用 flash_mla.flash_mla_interface 和对象 scheduler metadata。本轮迁入当前完整 DSA backend/multistep 类及 metadata producer/manager/indexer 依赖，HCU 注册两个类替换；其他平台沿原类。六个选择出的定义还原命名空间导入后 AST 与 DAS 相同，实际 package/probe 记录在测试目录。相同的 kpool mixin 与其他依赖复用固定官方，不动态导入 DAS。

保留 HCU metadata wrapper 的创建、刷新、Graph 消费、MLP-sync 实际行裁剪和输出补齐；不将 backend 对象塞入旧 CUDA tensor metadata 结构。补齐两个 EnvBoolWithAlias 字段，类型、默认 false 和旧名称沿 DAS；保留原始用户输入与环境兼容。CP helper 复用现有 HCU provider 和原生 kernel；本轮 TP4 非 CP 不外推 CP/PD 验证。

版本 dev18，当前 Hook 合同变为 365 项。tests/runtime/dsa_flashmla_current.py 的两项 CPU 回归通过：真实 producer 到 attention 消费边界传递新 scheduler 对象/num_splits，真实补齐行还原，HCU 保留 fresh source 对象而 CUDA 保留 tensor 地址；仅替换外部 FlashMLA kernel。证据 logs/dsa-flashmla-current-dev18.log、current-dsa-backend-audit.json；实际 wheel 与服务复验另记。DeepSeek-V4 dev17 尚在旧模型 get_pp_group 构造路径失败，Base/Cuda/Platform 拆分的迁移继续处理中。

## 17. 2026-10-29 DSA backend 实际 entry point 注册修复

dev18 wheel 的 542 个文件通过字节验证，但完整 Hook 合同检查发现两个 DSA 类未注册：新增调用仅放入旧 layers.attention.register_hooks，实际入口 hcu_sglang.register_optimizations 使用显式模块列表。修正在实际入口导入与调用 current_dsa；版本 dev19。保留 dev18 的失败合同证据，不将其 363 项已应用结果称为 365 项通过，也不将 dev18 GLM 复跑当作新后端验证。下一轮先确认完整 wheel 合同通过，再启动服务。

## 18. 2026-10-29 DSA TopK 当前依赖与 paged v2 门控同步

dev19 完整 wheel 的 365/365 Hook 与 3 个 seam 均通过；GLM 实际 Graph 已越过 CUDA FlashMLA 扩展错误，在当前 metadata producer 调用旧 DSATopKBackend.should_use_topk_v2_paged 时失败。同步固定 DAS TopK dependency 模块并让 backend/indexer 使用同一枚举与方法，避免新旧 consumer 混合；补齐 SGLANG_HCU_TOPK_V2_PAGED EnvBool 默认 false，沿 DAS 对 HCU paged v2 的明确 opt-in，保留 ragged v2 的 layout/dtype 条件。版本 dev20，完整服务复验继续；没有请求或模型精度通过。

## 19. 2026-10-29 DSA indexer HCU Hadamard 路径同步

GLM dev20 已越过 FlashMLA metadata 与 TopK 接口，实际 Graph warmup 的 Indexer forward 在 native HIP 旋转导入 fast_hadamard_transform 时失败。固定 DAS 对 HCU 默认使用缓存 dense Hadamard 矩阵及 F.linear，显式 SGLANG_USE_FAST_HADAMARD_TRANSFORM 才选择扩展。本轮迁入该缓存、数学消费和完整 rotate_activation(apply_scale=True) 函数，保留 power-of-two 校验与归一化开关；不以其他旋转或随机矩阵替代。显式 fast 扩展开关缺包仍正常报错。版本 dev21，当前合同 366 项；真实 CPU/GPU 数学与完整服务复验另记。

## 20. 2026-10-30 DSA indexer 当前 HCU producer/consumer 同步

在已触发的 Indexer 路径继续核对，native HIP logits 分支仍断言 page_size=1，而 DAS/HCU 当前池使用 page_size=64。本轮不启动这个已知不匹配的 dev21 全模型组合；dev21 三项 CPU/GPU Hadamard 回归通过，真实 BF16 GPU 最大绝对误差 0.0171714，属于单算子证据。

迁入固定 DAS 的完整 Indexer 类及当前本地 helper，HCU 通过原 module 和 DeepseekV2 的实际构造别名注册同一实现，避免只替换类定义后原 caller 仍持有旧类。保留 HCU FP8/BF16/INT8 cache 读视图、LightOp dense/sparse logits 分发、与 mask TopK 配对的资格检查、head-gate scale 和 logits scratch 预算；既有相同依赖复用官方，Hadamard 复用已验证的 DAS 实现。sparse producer 只有匹配的 consumer/API/布局/门控全部满足才可运行，未命中沿 DAS dense producer。运行不动态导入 DAS。

版本 dev22，合同 368 项；隔离暂存导入已通过，CPU/GPU 消费者、来源 AST、wheel 与完整服务结果继续记录。gfx936 INT8 与 CP/PD 仍未执行整模型验证，不能从 gfx938 TP4 profile 外推。


本轮实际检查：Indexer 与两个 HCU logits helper 规范化 AST 均与固定 DAS 相同；4 项消费者回归通过，涵盖 owner/broadcast 读存储、page64 packed ABI、dense clean-logits 参数、sparse producer/consumer 配对门控以及真实 HCU head-gate Triton 数学。72 项仓库 unittest 通过。依赖字段审计补齐 DAS 保留的 SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB EnvFloat(None)，无全局预算覆盖；证据 current-dsa-indexer-audit.json、logs/dsa-indexer-current-dev22.log、logs/repository-tests-dev22.log。

## 21. 2026-10-30 HCU paged-MQA 后端选择依赖同步

GLM dev22 在资源检查停止两轮后，使用通过准入的物理 GPU 0/1/6/7 越过前述路径，进入新 Indexer 的 paged logits 分发；其残留原生枚举缺少 is_lightop。迁入 DAS 当前 DSAPagedMQALogitsBackend，并让 Indexer 使用该同一枚举。HCU auto/显式 lightop 选择 page64 LightOp；普通 HIP auto 仍选择 AITER；不支持的显式后端正常拒绝，不把 HCU 当普通 HIP。当前配方使用 auto，不宣称已验证所有显式 CLI choices。版本 dev23，合同仍 368 项，模型复验继续。

## 22. 2026-10-30 四卡 W4A8 Triton producer/consumer 路径同步

GLM dev23 已越过 attention/indexer，在真实 MoE warmup 触发已安装 LightOp Marlin 对 E=256,N=512,K=6144 的 unsupported size。当前 DAS auto 同样优先选择可导入的 LightOp，不能把它记作 plugin 独有精度错误；DAS 另提供显式 SGLANG_W4A8_TPMOE_BACKEND=triton，而旧插件仅 auto/lightop/aiter。本轮同步固定 DAS 的完整 SlimQuant Marlin 配置/后端实现和独立后端规范化/context helper，保留 auto 的原选择，四卡新配方明确选择 Triton。

权重生产端必须在载入前选择后端：Triton 保留原 packed INT4 权重，消费同一 channel scale，沿 DAS 的 chunk、block 适配、align、两次 GEMM、激活、路由/共享输出与缩放；不能把已 Marlin repack 的权重直接送入 Triton。版本 dev24。当前仅选择标准 TP4/EP1 路径；完整 DeepEP W4A8 与 DSpark override 尚未执行模型验证。CPU/GPU 数学、完整 wheel 和实际请求结果另记。


Triton GPU 数学夹具首先复现原生 align 的 HCU 320-thread launch 超过 256-thread bounds，随后非法访存。固定 DAS 的同一 align 明确记录并修复该问题。本轮补齐其 HCU Triton count/prefix/sort、padding sentinel 和无效专家映射，公共入口 AROUND 在其他平台委托原实现；当前合同 369 项。原故障证据 logs/w4a8-triton-current-dev24.log 保留，修复后数学测试另记，不把崩溃夹具称为通过。


实际数学检查：修复公共 HCU align 后，随机 INT4 两层 MoE/SiLU/路由/缩放的 GPU 输出与显式参考完全相同，最大绝对误差 0。另以 E=256、N=512、K=6144、top-k=8 的 GLM 四卡真实权重形状执行 warmup、Graph capture 与 replay，零输入/零权重结果正确；共 3 项检查通过。旧夹具的高/低 nibble 顺序错误已修正，失败日志保留；不是通过修改数学容差或算子绕过。证据 logs/w4a8-triton-current-dev24-graph.log 与 w4a8-source-alignment.json。完整模型请求仍待复验。

## 23. 2026-10-30 DeepSeek V4 当前 Base/Platform 执行接入

DeepSeek dev17 的旧模型替换仍调用已删除的 get_pp_group。按当前 DAS 拆分迁入 DecoderLayerBase/Platform 与 NonCudaDecoderMixin、模型/输出包装和所需 mHC/transport provider。模型、causal class 继续保留官方类身份，经原公共入口调用当前方法；显式 super owner 适配保证原类与派生类实例均只执行一次 nn.Module 初始化。当前 Model/causal 共 13 个实际不同方法沿 DAS 同步，旧 Decoder 三个方法替换退役为完整 Platform 类替换；注册同步更新所有已导入的 NextN 构造别名。

14 个选择定义还原 provider 导入、super owner/base 和未启用 mxscale 的惰性导入后 AST 与 DAS 相同。8 个 mHC 数据/状态类与官方 AST 完全相同，复用其类型身份，使 target、attention、DSpark 生产/消费一致；仅把 _mix_stats_impl/post 两项 HCU 差异函数接入同一当前实现。当前合同 376 项。固定 DAS 的 AITER in-tree helper 与版本化 CSV/JSON 原样打包，保留其 upstream-first 能力检查；当前容器缺少三项 gfx950 tuning API，沿 DAS 原逻辑记录 skip，不将其称为实际 tuning 验证。

HCU 不执行的 CUDA row-bucket replay/tail boundary 依赖改为首次实际使用时导入；当前 HCU maybe_create 按 DAS 返回 None。显式 dense-mxscale 仍需基线实际支持的接口，本轮保留默认 false 并仅在原 opt-in 分支导入该接口，不宣称 gfx950/CUDA tail/CP/PD 已迁移验证。没有为缺失算子制造成功结果或替代数学。

版本 dev25；真实原模型类入口、PP group/五 stream、Platform 状态、NextN 已导入别名和 mHC 状态身份的 CPU 回归、完整 wheel 与 GPU 模型复验分别记录。静态/构造通过不代表 DeepSeek 精度通过。证据 v4-execution-source-audit.json、v4-shared-mhc-types.json、v4-execution-env-specs.json 与 logs/v4-execution-current-dev25.log。


GLM dev24 已越过 Triton MoE，在 EAGLE draft forward 的 CP 资格 helper 读取旧 enable_prefill_context_parallel 字段时失败。固定 DAS parallel_hook 为 legacy CP 从 enable_prefill_cp 与当前 attention backend 派生该 alias；官方 bag 没有此字段。共享 HCU helper 现优先使用真实 legacy 字段，缺失时按同一派生条件读取当前 getter，排除 dsa/dsv4 的 MLA legacy alias；不伪造全局 bag 属性。NextN 公共 forward 回归改为真实经过 CP-disabled helper 分支，避免仅在 enable_cp_v2=True 时跳过原失败点。完整 CP/PD 仍未验证。

## 24. 2026-10-30 EAGLE 当前 KV plan 返回与 hidden-state 规则同步

GLM dev25 完成 target/draft Graph 后，在服务 warmup 的 EAGLE draft 调用触发旧包装器按二值解包当前三值返回。prepare_for_draft 已返回 (ForwardBatch, graph_flag, kv_loc_plan)，runner 位于第五个 positional 参数。包装器现只读取第一项补齐 DAS 的 mHC pre-norm 条件，原返回容器、KV plan 和其他项按身份完整返回；普通 GLM 不改 hidden-state 标志。

同时核对固定 DAS eagle_utils.eagle_prepare_for_verify 与官方均明确 return_hidden_states_before_norm=False；旧 HCU wrapper 强行改 True 已不对应当前路径，退役此一项 Hook 与函数。当前合同 375 项；不通过丢弃 plan、关闭 EAGLE 或伪造返回规避异常。版本 dev26，公共入口/调用者三值消费与完整服务复验另记。


DeepSeek dev25 已加载目标权重并进入 mHC Graph，旧 hcu_sglang.kernels.mhc 仍从官方 DSA utils 导入被移除的 round-robin CP 名称。改为同仓已接入的当前 HCU compat provider，保留当前拓扑与 round-robin/interleave 判定；实际模块导入和服务路径继续复验，不设置虚构 CP 状态。

## 25. 2026-10-30 GLM 精度失败与当前 DAS mHC 同步

GLM-5 INT4 dev26 attempt17 已完成 target/draft Graph、warmup 和 HTTP 请求，但算术、中文、英文三个响应均乱码且耗尽 1024 token，不能作为精度通过。主动终止余下请求，控制器清理完成，残余 PID 为空且端口关闭。实际 checkpoint 的三个 dense INT8 投影与 FP32 INT8 matmul/scale 参考误差均为 0；这只排查局部量化投影，未证明模型正确。

DeepSeek V4 dev26 attempt4 在 Graph 捕获进入旧 mHC 的 TileLang Allocate monkey patch 时失败。对照固定 DAS，当前 mHC 已删除该旧补丁并增加 HCU 小 batch split-K、AITER 物理维度压紧等路径。本次整体同步其 mHC 模块，只将 is_hcu 导入改为 plugin 检测；模块内所有顶层函数/类 AST 与固定 DAS 一致。没有通过关闭 Graph、mHC 或 AITER 避开问题。

真实 HCU 运行 8 token、4×7168 hidden 的 mHC pre，对照同源 _mhc_pre_torch：post 最大绝对误差 1.85e-6，comb 8.8e-7，BF16 activation 0.0078125；Graph 捕获及 replay 通过。初次探针误用了另一历史 mhc_pre_torch 辅助函数（其 post 额外乘 2），失败原日志保留，改用当前实际 dispatch 的 _mhc_pre_torch；生产公式未为测试修改。dev27 整体服务复验另记，GLM 精度仍待修复。Qwen3 dev26 attempt7 六请求再次通过；MiniMax dev26 attempt3 正在复验。

## 26. 2026-10-30 MLA INT8 反量化的 None block size 分支同步

继续排查 GLM 精度时发现旧 post_load_weights 通过 hasattr(weight_block_size) 判断 block quantization。当前 QuantizationConfig 即使 channel INT8 也显式带 weight_block_size=None，因此旧分支跳过 channel scale，w_kc/w_vc 留为原始 INT8。固定 DAS 已改为检查实际 block size 是否非 None；plugin 同步其整个 INT8 分支，分支 AST 与 DAS 一致。该修改保留 MLA 路径和 checkpoint scale，不改变注意力/量化后端配置。

新增以显式 None block size、非均匀 channel scale 的实际 post_load_weights 数值回归，检查拆分出的 w_kc/w_vc 是 BF16 且精确对应反量化值；旧 wheel 失败证据与修复后结果均单独保存。该局部问题修复不等同 GLM 整体精度通过，后续完整六请求复验另记。另补充 256 experts、高位 expert IDs 的真实 HCU W4A8 单算子数值参考通过，最大误差 0。

DeepSeek dev27 attempt5 已越过 mHC 并完成 decode Graph，在 verify Graph 遇到 HIP metadata factory 返回 None，而 HCU FlashMLA consumer 要求 FlashMLASchedMeta 对象。沿用当前 DAS HCU backend 的对象 metadata 路径，新增 HIP factory 的 HCU 定向 Hook，通过同源 get_flashmla_op 访问当前 flash_mla_interface provider；非 HCU 仍调用原 factory。合同增至 376 项。

## 27. 2026-10-30 GLM 六请求通过与 RowParallelLinear 融合能力接入

GLM dev28 attempt18-admission-retry2 使用原 EAGLE、FP8 KV、正常 Graph 和显式 DAS Triton W4A8，在四卡上严格 6/6 通过：323、北京、Paris、4、10、重复 323，均 finish_reason=stop。目标 verify batch 1–8、draft decode/extend Graph 已捕获；04:28:33 清理完成，无残余 PID、端口关闭。一次准入资源竞争只造成未启动的 pending_resources，随后控制器自动重试，没有同时启动两个模型。

DeepSeek dev28 attempt6 越过 HCU FlashMLA metadata 后，在共享 MLP 查询 RowParallelLinear.supports_fused_silu_mul_fp8_quant_input 时失败。同步 DAS 两个能力谓词与完整 Row forward 的融合输入消费实现，仅对 HCU 新量化参数的调用进入该实现；普通调用和非 HCU 交还原 forward。保留环境门控、kernel 可用性和真实 consumer trait 的组合判断，不用恒 False 或关闭配方开关代替缺失 API。新增 FP8 EnvBool 的默认 false 与 DAS 相同；DPSKV4 原有独立融合开关保持原值。

三个方法只限定真实官方全局 provider；能力方法挂到当前真实 RowParallelLinear，新增已有 forward 入口的 AROUND Hook，合同 377 项。回归覆盖能力/consumer 配对、FP8 q/scale/输出 dtype 经当前 Row forward 到 quant method、默认及非 HCU 委托。外部 LightOp 数学在该 CPU 数据流夹具中替换为可观察 quant producer，不能作为其 GPU 精度证据；完整 DeepSeek 服务继续复验。

## 28. 2026-10-30 共享 MLP 构造门控同步

进一步核对已失败的共享 MLP 路径，官方 DeepseekV2MLP.__init__ 将所有 HIP 都设为 use_fused_clamp_act_mul=True，而固定 DAS 明确是 HIP 且非 HCU 且 SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL。此前只同步 forward 留下初始化分支差异。本次迁入完整当前构造器；除该赋值外两个输入构造器 AST 完全一致，其他初始化、parallel_group 和权重处理没有省略。按 DAS 添加默认 True 的环境字段，HCU 选择 DAS 的后续 clamp 路径；这是源码中的平台资格判断，不是为跑通测试关闭配方开关。

新增构造器 Hook 后合同为 378 项。真实公共类构造回归保留 nn.Module 初始化，替换昂贵线性分配，验证 HCU 与普通 HIP/环境门控；完整 DeepSeek 服务仍需新版本复验。dev29 的第七轮在资源等待中出现多次准入竞争，实际 run_id 以控制器记录为准；未启动的 pending_resources 不计为模型失败或请求结果。

## 29. 2026-10-30 BW1100 DeepEP ABI、buffer 与量化路径同步

DeepSeek dev29 第七轮实际失败在普通 HIP AITER fused-clamp 包不存在，dev30 已按 DAS HCU 构造门控越过该路径。dev30 attempt8 随后在 Graph 的 low-latency dispatch 被 vendor Buffer 拒绝 use_fp8 参数。固定 DAS 已通过签名检测区分 upstream use_fp8 与 BW1100 quant_type/topk_weight ABI；本次迁入完整 DeepEP dispatcher 模块，保留 normal/low-latency、量化生产/消费、combine、event/overlap，以及 target/draft 共用最大布局 buffer 的管理。

仅替换现有 DeepEPBuffer 和 DeepEPDispatcher 两个公开类，其他 AST 相同的数据/配置类型复用官方身份；合同 380 项。HCU DeepGEMM 能力探测复用 DAS 的真实 deepgemm 导入检查，量化布局另要求实际选择 deep_gemm runner，不能因包已安装就切换 AITER/SlimQuant 的 dispatch。没有安装/替换系统 DeepEP，也不通过关闭 DeepEP、speculative 或 Graph 绕过错误。

CPU 回归覆盖 quant_type=0/2/3 与权重/scale contract、不支持 NVFP4 的明确拒绝、backend/能力/SlimQuant 选择、真实公共 dispatcher 模式选择及 target/draft 最大 buffer 布局保留。通信 kernel 由可观察 vendor stub 替代，不能当作真实多卡通信或模型验收；新 wheel 的四卡完整服务继续验证。


同一失败边界继续核对量化消费，当前官方 DeepEPMoE 没有 DAS 的 HCU channel-FP8 grouped GEMM 分支。因此同步其已验证的 legacy channel-FP8 producer/consumer：完整权重 pack/非持久 buffer、普通及 masked FP8 两层 GEMM、LightOp scatter/gather 资格与 fallback、SiLU clamp/FP8 quant。只在 DAS 同一 channel-FP8/SILU/非 OAI/无 local-shared 条件进入；其他模式委托原实现。旧配置类没有新增 local_shared_experts_without_dispatch 字段时读取其 DAS 默认 False，不声明该 opt-in 功能已支持。构造器同步 legacy flags 和 dispatcher quant config；checkpoint 后按 Source 打包并删除 canonical 权重，保留逻辑 shape。两个现有公开方法新增 AROUND 后合同 382 项；量化后处理复用已有合同目标。运行没有动态导入 DAS 源码。

## 30. 2026-10-30 V4 输出收尾 helper 接到真实模型类

DeepSeek dev31 attempt9 的 target verify Graph 已通过完整 body，包括 DeepEP vendor dispatch、HCU channel-FP8 grouped GEMM 和全部模型层，随后在输出收尾缺少 logits_from_body_output。该 helper 已位于当前 DAS 的 owned causal class，但实际公共 causal 类保留官方身份，原注册只覆盖已有方法交集，遗漏了这个新增方法。

HCU 注册现在把同一已迁入的 helper 接到真实官方 causal 类；不更改数学，不伪造隐藏状态。继续保留既有 forward Hook 及 Native 类身份，合同仍为 382 项；只补齐当前基线缺少的能力方法。公共 forward 回归直接走真实模型类到 helper/logits_processor，检查 pre-hc-head、aux/PD capture 和输出 flatten 的实际消费。Source body/helper 已成对同步，Graph 和完整请求须在新 wheel 复验。

## 31. 2026-10-30 normal DeepEP FP8 全局依赖补齐

DeepSeek dev32 attempt10 进入 normal channel-FP8 分发消费，在已迁入的 contiguous GEMM helper 读 get_offloader 时因遗漏导入失败。该 helper 的源模块额外导入当前官方已有的 get_offloader 与 copy_list_to_gpu_no_ce；本次显式接入这两个真实 provider，保留禁用 copy-engine 时的无 copy-engine GPU 计数搬移分支。

新增 normal 两层 GEMM/quant/scatter/gather 的 CPU 数据流回归，使用真实公共 DTO，检查 count、scale、clamp 与输出传递；外部 GPU kernels 由可观察 stub 代替。同步检查该 owned 模块顶层函数的实际引用全局名均存在，防止只跑 masked 路径而遗漏 normal 依赖。没有改精度容差或关闭 normal dispatch；合同仍 382 项，完整服务由下一轮验证。


全局依赖检查又发现 Native ep_moe module 不再 re-export scatter/gather/m_indices；相同检查在缺失时明确失败。对照 Source，scatter/gather 的现行 HCU-aware 实现亦不同于官方，不能只导入名字相似的旧实现。本次把两个入口及其实际函数依赖闭包和 m_indices fallback 放入 owned kernel provider，保留 Source LightOp 资格与原 Triton fallback；只适配 is_hcu 检测导入，所有选中函数 AST 与 Source 同一实现。未改变其他官方调用方。

## 32. 2026-10-30 V4 target/draft backend 分流同步

继续审查启动失败链，发现固定 DAS attention_registry.create_dsv4_backend 明确用 HIP 且非 HCU 才进入 DeepseekV4HipRadixBackend；当前插件遗漏该分流，前述 DeepSeek 轮次实际仍选普通 HIP 类。现在对现有 registry 函数及 draft decode/prefill 工厂增加 HCU AROUND，按 DAS 同一路径进入 V4 factory；其他平台委托原调用。registry dict 保存 callable，不会仅因模块函数被 Hook 自动更新，因此一并接好实际 ATTENTION_BACKENDS['dsv4'] 字典入口，保留原 callable 供非 HCU 委托。

当前配方 auto/FlashMLA 选择已迁入的 owned HCU Base/Multistep backend；不通过临时篡改 is_hip 或平台全局值选路。显式 TRTLLM/Mega、CP/PD 等模式不因本轮而新增验收。合同增至 385 项。前述普通 HIP 分支没有模型通过记录；正确 HCU backend 的真实服务验证另记。

dev33 已完成 6 项 normal/masked/全局依赖/权重别名/DTO/HCU kernel 回归；11 个 owned EP kernel 定义 AST 与固定 DAS 相同。真实 HCU byte scatter、per-token scale 与路由加权 gather 最大绝对误差 0，m_indices 有效行及 -1 padding 精确匹配。这些属于单算子证据，不能代替完整模型请求。

## 33. 2026-10-30 四卡代表模型验证结果

当前运行时代码提交 `86f3f70ac72e673627c816f546eaf647cb045cf9`，wheel `0.3.2.dev34`，SHA256 `45b88367f9fc3f9462eed414d24279fb7176dacb5f9c588733f0e7e0479d415a`；本地提交，未推送。当前源码/安装文件逐字节核验和 385 项完整 Hook 合同均通过，各次实际版本见下表。

| 模型 | 实际 wheel 目录 | 运行 ID | 物理 GPU | 严格请求 |
|---|---|---|---|---|
| qwen3-30b-instruct-tp4 | `site-dev26` | `qwen3-30b-instruct-tp4-aiter-attempt7` | 0,1,6,7 | 6/6 |
| minimax-m25-bf16-tp4 | `site-dev26` | `minimax-m25-bf16-tp4-attempt3` | 0,1,6,7 | 6/6 |
| glm5-int4-tp4 | `site-dev28` | `glm5-int4-tp4-attempt18-admission-retry2` | 0,1,6,7 | 6/6 |
| deepseekv4-fp8-tp4 | `site-dev34` | `deepseekv4-fp8-tp4-attempt12` | 0,1,6,7 | 6/6 |

六个确定性检查为算术、中文、英文、短阅读、代码结果与重复请求；最终答案及 stop 条件均严格通过。各服务结束后自有进程清理为空，端口关闭。Graph、EAGLE/DSPARK 与 DeepEP 按对应配方记录，具体捕获模式和参数偏移见原始运行配置与日志。

环境仍为固定官方 SGLang 0.5.22 加隔离插件，既有容器提供 DAS 0.5.18 依赖，cookbook 0.5.18 仅为配方参考。用户授权四卡 profile；本表不外推完整八卡配方、数据集精度分数、长上下文或性能基准。全量迁移台账保持独立状态。

仅在启动/精度失败处对照固定 DAS 代码，未启动 DAS 对照服务。原始证据在指定 test-model 目录的 README.md、results.json、runs/、artifact-dev 各版本 JSON 和 wheel-hooks-dev 各版本 JSON；历史失败完整保留。

DeepSeek dev34 attempt12 的日志明确记录 HCU owned DeepseekV4AttnBackend，用于 target/draft 的 backend 选择已与 DAS HCU 分流对应。attempt11 在静态比例 0.8 下因 target/draft 权重合计超过 KV 预算停止，框架测得最低可用比例 0.8175；成功轮使用 0.9，并保留 16K KV、8K context、Graph max batch 8、每卡至少 132000 MiB 空闲的准入检查。该变化为测试 profile 的显式容量调整，未更改运行时代码或隐藏原内存检查。

## 34. 2026-10-30 HCU 安装命名与文档一致性核验

当前 `pyproject.toml` 的 distribution、Python 包和 general plugin 入口分别为 `hcu-sglang-plugin`、`hcu_sglang`、`hcu`。本轮从干净代码提交构建 `hcu_sglang_plugin-0.3.2.dev34-py3-none-any.whl`，检查 wheel metadata/entry_points 和包目录，再隔离安装并在源码目录外检查 import 来源。唯一 general plugin 入口为 `hcu = hcu_sglang:register`，console script 为 `hcu-sglang-serve`；无旧包目录或 `hcu` 入口。该 wheel 的 SHA256 为 `79150616eb8c55e984317554addeed9faad82af42fe811d2f76516e8fc012675`，与前文远程历史构建分别记录，不因版本号相同替换历史文件哈希。

修正安装指南的 wheel/editable、metadata、严格加载合同、服务及卸载示例，当前合同显式指定 `--plugin-package hcu_sglang` 并先初始化 AITER 兼容。同步配置、回溯、排障、README 和相关 Skill 中仍将当前实现称作旧包的说明；旧版本实施记录、来源清单、兼容工具接口和历史文件名保持其来源语义。硬件接口 `is_hcu()` 不属于插件入口改名范围。

本轮仓库 72 项离线测试、L0 的 333 个 Python 文件/8 个 Skill、全部 Skill 元信息及 Markdown 链接/标题锚点检查通过。网页从原始文档重新构建，不直接修改生成 HTML。检查证据保存在本次 ignored artifacts 中。本轮未修改运行时行为、未在本机执行 HCU Hook/模型服务；wheel 名称与入口检查不作为模型精度或设备可用性验收。

## 35. 2026-11-01 RMSNorm/Gemma HCU 分支继续迁移

继续执行 plugin-v3-1028，输入仍为固定 official `baa9befa` 与 DAS `15c7204b`。从当前干净 `b97b4d9` 开始，本轮只做本地提交，不推送。dev34 的八卡 Qwen3-235B 验证与本轮 dev35 代码验收分别记录，不把隔离旧 wheel 的结果归给新代码。

原插件 RMSNorm 在 residual 非空时直接调用 native；当前 DAS 明确进入 LightOp `fused_add_rms_norm_opt`，保留实际 residual 原地更新和 post-residual 的独立张量语义。本轮迁入该完整 HIP 方法，并将当前 Gemma HIP 方法、显式 `forward_with_lightop_fp8_quant` 与其 vLLM ABI 辅助函数放入 owned `norm_current`；只限定官方 provider 全局名、函数命名和惰性 vendor 导入。AITER Gemma 优先级、`1 + weight` 缓冲、batch-invariant、FP8 门控与 `(quantized, residual)` 返回结构沿用来源。非 HCU 保留传入 original，不改官方文件，也不动态导入 DAS。

[源码对照](hcu-sglang-development.md#历史-json-快照)（`hcu-norm-source-alignment.json`） 固定源文件哈希和五个选择定义；去除导入限定并规范化抽取 docstring 缩进后 AST 相同。这只证明选择的方法，不能证明整个 layernorm 文件已迁移。全文件处置保持 in_progress；AR/RMSNorm 的通信组与 per-group quant 融合仍需独立处理。

`tests/runtime/norm_current.py` 在指定容器内通过六项 CPU 回归，真实公共 RMSNorm/Gemma 类、注册与 forward 调用保留，昂贵 LightOp kernel 用可观察数学夹具替代。覆盖非连续输入、融合分支命中、residual/post-residual 存储语义、batch-invariant、Gemma 权重加载/AITER 优先级、FP8 carrier/gate 与非 HCU original 委托；不据此宣称 vendor GPU 数学、Graph 或模型精度通过。当前准备版本 dev35、Hook 合同 386 项，产物和完整运行验证结果待本次执行后补充。原始日志位于本次 ignored `artifacts/plugin-v3-1028-20261101/`。


本阶段实际结果：dev35 wheel SHA256 `4dabfc1963c943b625de840a17c70a83eb2aab29e5ceb840ac11cd043839e35a`；575 个包文件与源码及隔离安装相同。隔离 wheel 严格加载 386/386 项 Hook、3 个具名 seam，软错误和旧包导入为空；72 项仓库 unittest、6 项公共入口 CPU 回归通过。

`tests/runtime/norm_current_gpu.py` 使用真实 gfx938/HCU LightOp 数学，3 项通过：BF16/FP16 RMSNorm 的 plain/residual/post-residual/非连续输入与 FP32 参考，BF16 Gemma plain/fused 与 `1 + weight` 参考，以及 RMSNorm residual 的 Graph 捕获和两次不同输入 replay。未使用 kernel stub、未修改生产容差；这些仅证明该测试的形状和分支，不代表 Gemma FP8 opt-in 分支或整模型已通过。完整 MiniMax 八卡服务已用该隔离 dev35 wheel 启动，结果在后续验证记录补充。

## 36. 2026-11-01 原八卡配方 Qwen3-235B 验收

在八张卡均空闲时，按 cookbook 固定 `e91f05fe` 的 Qwen3-235B-A22B-Instruct-2507 / BW1100 / SGLang0.5.18 单机配方运行固定官方0.5.22 + 隔离 plugin dev34。保留 TP8、fa3、page64、内存比例0.85、默认上下文/KV 和默认 Graph，没有沿用四卡 bounded smoke 的容量限制；只使用本地只读权重、任务 loopback 端口及既有离线 Rust/缓存配置。

运行 `qwen3-235b-instruct-tp8-dev34-20261101-attempt1` 于 00:02:07–00:13:40 完成。六个算术/中文/英文/阅读/代码/重复请求的 HTTP200、模型身份、完整最终答案和正常 stop 均通过；默认 Graph 捕获从 batch512 开始。结束时自有进程为空、端口关闭。对提交 `b97b4d9`、实际 dev34 wheel 与隔离安装的574个包文件重新逐字节核验一致。

[可提交验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-qwen3-validation-20261101.json`） 保存 cookbook 来源、版本、wheel与响应哈希、每个实际最终答案和清理结果；原始日志/请求/响应继续留在用户指定 test-model/runs 目录。该结果证明 dev34 的八卡短请求配方，不是数据集精度、吞吐、全 DAS 功能等价，也不归因于本阶段 dev35。

## 37. 2026-11-01 归一化通信组与量化 handoff

dev36 将固定 DAS 的两个 communication_op fused norm 入口和两个 layernorm AR/norm/quant helper 接入本仓，按 `tp` / `attn_tp` / `moe_tp` 选择真实 coordinator，不能把 MoE EP 大小当作 MoE TP 大小，也不能在选定 attention/MoE 组后退回全 TP 组。计算 AITER 开关与 communicator 融合开关分开，融合拒绝时仅在来源规定的组上归约一次；post-residual 只加一次。

量化路径保留 FP8 tensor/scale 和可选 BF16 side output、bpreshuffle 的 transpose/materialize 区别及 `None` 拒绝约定。[四个 helper 对照](hcu-sglang-development.md#历史-json-快照)（`hcu-norm-collectives-source-alignment.json`）还原 provider 限定后 AST 与 DAS 相同；两个真实 GroupCoordinator 方法 AST 在官方与 DAS 中完全相同，直接复用。注册通过官方 Hook 传播机制更新已导入别名，未修改两个输入仓。

| 检查 | 本阶段实际结果 |
|---|---|
| CPU 公共入口 | `tests/runtime/norm_collectives.py` 9 项通过：三种组、默认 TP、拒绝未知组、预导入 alias、独立通信策略、decline fallback、Gemma 权重、FP8/BF16 carrier 和 scale 布局 |
| 真实八进程 HCU | `tests/runtime/norm_collectives_gpu.py` 使用真实 AITER IPC fused kernel，TP8 / attention TP4 / MoE TP2 各 rank 的 BF16 输出及 residual 与 RCCL+FP32 参考通过 |
| gfx938 平台拒绝 | 未把 gfx95-only per-group fused quant 伪装成功，三个组均保留 native guard 的 None 返回 |
| 隔离 wheel | SHA256 `7df1214bbc0f347c8b3cacf83ecaf240de8fbf7d360cc11a6e02b28d1383b6b9`；576 个包文件与源码、隔离安装一致 |
| 完整加载合同 | 390/390 Hook、3 个具名 seam，通过；软错误、未应用与旧包导入为空 |

八进程探针显式创建 Gloo 元数据组、NCCL 参考组和真实 AITER communicator，并将其提供给原生 GroupCoordinator 方法；只注入组 getter，未替换通信或归一化数学。该探针不验收完整 coordinator 构造、CLI 后端选择、Fabric 或 collective Graph replay；细节与原始日志哈希见[可提交摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-norm-collectives-gpu-validation-20261101.json`）。模型验收按各自实际 wheel 版本记录。

执行台账新增两个已实现文件处置，当前为3007 pending、3 implemented；这是逐文件审阅状态，不是全量完成率。parallel_state 的新后端选择/strict-Fabric、CP/PD，以及其他领域仍需分阶段继续，不能由本阶段或八卡代表模型结果关闭这些范围。

## 38. 2026-11-01 MiniMax-M2.5 BF16 八卡验收

隔离 dev35（代码 `fefb5fc`）按同一 cookbook 固定版本的完整 MiniMax BF16 单机八卡配方运行，保留 TP8、fa3、FP8 KV、内存比例0.9、131072上下文、prefill16384、max-running-requests512 和默认 Graph。运行 `minimax-m25-bf16-tp8-dev35-20261101-attempt1` 于00:16:58–00:27:48完成，六请求完整最终答案及 stop 全通过，自有进程为空、端口关闭。

对实际 dev35 wheel、隔离安装与已提交源码的575个包文件再次逐字节核验一致。[验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-minimax-validation-20261101.json`）保留来源配方、版本、wheel/响应哈希、实际答案和清理结果。该结果属于dev35；不冒充dev36通信组路径、数据集精度、长上下文压力或性能基准。

## 39. 2026-11-01 原配方后端参数与组构造链路

原GLM八卡配方的 `--custom-all-reduce-backend aiter` 在dev36真实parser上被拒绝（exit2）。本阶段新增完整四选项CLI、worker继承与distributed bootstrap桥接，保留来源的disable优先级和legacy auto提升规则。官方0.5.22原始/comm记录无该字段，采用本仓进程配置，不写入冻结字段；程序化调用方式及单进程边界见[配置说明](configuration.md#当前-custom-all-reduce-参数)。同时迁入来源的两个deprecated NSA别名，保持canonical字段、选择及默认语义。

新增 `distributed_backend` owned实现，提取DAS完整GroupCoordinator构造器、world/model group工厂和backend setter/getter；限定现有官方provider后，[五个选择定义](hcu-sglang-development.md#历史-json-快照)（`hcu-custom-allreduce-source-alignment.json`）规范化AST相同。保留原类身份、原有comm属性和进程组创建路径；只在HCU调用owned实现，其他平台保留传入original。strict-Fabric构造失败保留原异常，disabled明确拒绝，QuickAllReduce沿独立门控及transport排除规则。

`tests/runtime/distributed_backend.py` 在指定容器内13项通过，包含真实CLI/原始record、冷子进程继承、公共bootstrap与GroupCoordinator构造，昂贵进程组/communicator分配使用可观察夹具；不将CPU构造测试记为Fabric硬件通过。首次导入发现官方parallel_state没有re-export来源所用get_bool_env_var，已显式引用真实utils provider，原始失败保留。完整GLM八卡参数解析已通过，未删除 `aiter`、EAGLE或旧NSA选项。

当前dev37 wheel SHA256 `c991a6918ea1fff44bada82348477bab133ef0b0366837043cee2504e7e05498`；577个包文件与源码及隔离安装一致，394/394 Hook与3个具名seam严格加载通过，无软错误、未应用或旧包导入。原八卡GLM INT4/EAGLE/FP8 KV配方正在指定容器复验，结果单独补充；没有运行共享节点的全局NUMA sysctl。parallel_state其余CP/拓扑、新namespace字段等仍按台账保持未完，不由选择定义对照推定完整文件或全量DAS完成。

## 40. 2026-11-01 GLM-5 INT4 原八卡配方通过

代码提交 `89ac151`（dev37）完成后，运行 `glm5-int4-tp8-dev37-20261101-attempt1` 于00:59:54–01:08:40通过六个严格请求：HTTP200、正确模型标识、完整最终答案及stop均通过。TP8/DP1/EP1、旧NSA参数、INT4 quantization、EAGLE、FP8 KV、内存比例0.85和默认Graph保留，无四卡smoke的context/KV-token/graph-size覆盖。只使用本地模型路径、loopback端口及任务Rust/cache；未修改主机NUMA sysctl。

八个rank日志均确认 `custom_all_reduce_backend=aiter requested_transport=ipc selected_transport=ipc disabled=False`；W4A8按原配方`auto`解析为LightOp，未设置TPMOE backend覆盖。权重后处理与target/draft Graph完成，服务结束时自有PID为空、端口关闭。实际wheel/隔离安装与已提交577个包文件逐字节一致，[验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-glm5-validation-20261101.json`）保存版本、wheel/响应哈希、实际答案与后端日志计数。该结果不外推Fabric、数据集精度或性能。

## 41. 2026-11-01 已有SiLU迁移实现复核

固定DAS的activation模块只有 `SiluAndMul.forward_cuda` Python定义与官方不同，另在module级 `_use_aiter` 选择上排除HCU。已有 `norm_activation` 的CUDA AROUND与构造AFTER分别覆盖同一LightOp调用和HCU构造选择；确定性/native及其他平台的effective original保留。其他Python定义AST与官方相同。具体审阅及环境边界见[记录](hcu-sglang-development.md#历史-json-快照)（`hcu-activation-source-review-20261101.json`）。

`tests/runtime/activation_hcu.py` 四项通过：HCU/AITER/确定性构造选择，真实公开LightOp BF16/FP16输出在三组形状下与FP32 SiLU×up参考比较，变更gate/up输入的Graph replay，以及非HCUoriginal委托。未改运行时数学或容差；本阶段补充验证与处置，不重建相同版本wheel。

最初强制全局 `SGLANG_USE_AITER=1` 的探针在进入activation之前，因容器缺少 `aiter.gemm_a8w8_blockscale` 从fp8_utils导入失败；固定DAS与官方该开启表达式/导入要求相同。改用原配方默认的global-AITER关闭状态验证HCU LightOp，原失败日志保留；不增加假函数或篡改来源门控，全局AITER部署仍未验收。现有代表模型配方未要求该全局开关。

DeepSeek-V4原八卡配方正在复验，保留TP8/DP8/EP8、DSPARK、DeepEP、context32768/graph32和64线程加载。两个网卡xxx替换为实际ens14f0，显式DeepEP配置文件由已安装vendor对8-rank的推荐dispatch/combine参数导出，并构造真实Config对象验证字段；不是未执行的调优或性能结论。该任务仍有全量文件审阅与其他功能待完成。

## 42. 2026-11-01 DeepSeek-V4 原八卡 DSPARK/DeepEP 验收

隔离dev37（代码 `89ac151`）运行 `deepseekv4-fp8-tp8-dev37-20261101-attempt1`，01:26:47–01:34:20完成六请求，HTTP200、模型身份、完整最终答案及stop全部通过。保留TP8/DP8/EP8、DSPARK、多stream/fast sampling、DeepEP、max-running32、context32768、graph32、内存0.90与64线程加载，无四卡KV/context/graph缩减。八个DP rank日志明确命中owned HCU `DeepseekV4AttnBackend`；结束时自有进程为空、端口关闭。

[验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-validation-20261101.json`）保存实际版本、577个包文件的源码/wheel/隔离安装核验、响应哈希、backend路径/ranks与DeepEP配置哈希。网卡/模型/draft/profiler/config占位符按节点实际值填写；EP8推荐配置由实际vendor工厂读取数值、重建真实Config验证，并保留独立来源哈希。首个日志提取器误用了历史文案，修正为实际logger的 `HCU dsv4 backend:` 模式后确认八rank；没有重启或改变通过的模型运行来修补证据。

## 43. 2026-11-01 完整 Python AST 文件复核

新增 `audit_python_equivalence.py`，在固定两输入的Git tree/blob上完整比对，保留docstring/type comments/常量/imports/guards/decorators；权限、对象、shebang、输出重叠、dirty输入和并发决定修改均保护。5项实际Git fixture检查通过，包含布尔/数值差异、docstring、无效语法、入口/权限、人工决定保留与陈旧stat cache下index字节保留。

本次覆盖1534个双边changed Python路径：26个AST一致、1494个实际不同、14个权限/类型漂移。复核26个匹配路径后，只添加尚无人工记录的代码级upstream，未覆盖已有决定；两个来源HEAD/index不变。当前台账为2975 pending、26代码级upstream、5 in_progress、4 implemented。报告保留[逐文件哈希](../plans/sglang-das-python-equivalence.json)，代码复用与行为/依赖验收分层，不据此计算全功能完成率。

## 44. 2026-11-01 线性层预量化消费者与 fused RMS 契约

从干净 `5082647` 继续M1。旧Column包装器没有 `input_quant_args`，公开接口在dev37上实际抛出unexpected keyword；同时它对rms_weight执行单独归一化，没有来源的fused INT8消费契约。本阶段迁入来源的Replicated/Column/Merged forward、Column构造/partition、Merged构造和attn_cp group解析，保留原生类身份和准确super owner。完整helper原已在本仓与来源同字节，本次将其接入实际投影消费者，不另写数学fallback。

显式q/scale不依赖legacy开关，unsupported消费者按来源拒绝；保留epsilon、input/residual更新、Merged fused分支五值返回和普通分支二值返回。Row的来源位置参数及swiglu_limit也接到已有完整forward；普通无扩展调用与旧位置output buffer保持effective original。[对照记录](hcu-sglang-development.md#历史-json-快照)（`hcu-linear-source-alignment.json`）包含全部十个改变/新增Python定义，去除provider限定和显式owner适配后AST相同，helper原始字节相同。

| 验证 | 实际结果与范围 |
|---|---|
| dev37公开反例 | 显式prequantized Column调用因unexpected keyword失败，原始日志保留 |
| CPU公共入口 | 10项通过，含真实SlimQuant消费者与post-load布局、三个投影类、QKV继承、fused输入/residual/epsilon、Merged五值、组选择、unsupported拒绝及Row扩展/旧调用 |
| 真实HCU | dev38隔离wheel的3项通过：BF16/FP16 fused RMS量化/残差；真实INT8 GEMM的显式/融合路径及Graph replay；packed MLA q/kv归一化、q量化与rope保持 |
| dev38产物 | 578包文件源码/wheel/site一致，SHA256 `a8c00a5ff80060ecbafa8484a2f3c27e1b1bb20fd0967a2af0aa71da56ecab5c`；400/400 Hook与3 seam通过 |
| dev39产物 | 只补Row调用路由；578包文件一致，SHA256 `b6c970994bb9b8c490174ea28bf5eae8fbc25520bf80b5860341dca62d2be2f3`；400/400 Hook与3 seam通过；模型结果按实际版本另记 |

GPU首轮的投影夹具错放为连续[K,N]，不符合真实INT8 GEMM post-load[N,K]布局。修正为调用真实process_weights_after_loading后通过；生产GEMM、来源数学和测试容差未改。3项GPU结果只归于dev38；dev39路由由新增CPU公共调用覆盖，不将其冒充新GPU执行。原始证据在ignored `artifacts/plugin-v3-1028-linear-20261101/`。

[原八卡Qwen3-235B的dev38运行摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-qwen3-dev38-validation-20261101.json`）记录六请求通过、进程/端口已清理；该中间artifact来自独立build-dev38源码快照，不伪装为含后续Row修复的dev39提交。后续按最终代码版本记录完整服务。

本阶段77项仓库unittest、L0、执行台账快照、Skill校验及实际宽/窄屏检索/阅读/导航检查通过。GPU证据与源码快照按版本分开，未推送远程。

## 45. 2026-11-01 channel-FP8 fallback 来源路径同步

从本地 `9976f34` 继续M1。旧FP8包装器在HCU上固定调用LightOp channelwise GEMM，未覆盖当前DAS的torch._scaled_mm探测、逐形状BLASLt失败缓存与选择性Triton分支。本阶段迁入完整 `_apply_fallback_scaled_mm`：仅RuntimeError且文本包含blaslt才记录失败形状；其他异常保留原对象抛出。key包含device type/index、M/N/K、输入/输出dtype，成功probe不污染cache。

缩放、bias、padding裁剪、旧tuple输出和dtype转换按来源执行；非HCU保留effective original。[源码证据](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-fallback-source-alignment.json`）去除两个provider限定后AST相同，复用的Triton wrapper、kernel、scale适配与weak-contiguous检查四个定义在固定DAS/官方中AST完全相同。没有用另一个数学实现伪装fallback成功。

`tests/runtime/fp8_fallback.py` 6项CPU公共入口回归通过：成功路径scale/bias一次、BLASLt失败只probe一次、dtype cache隔离、padding/tuple输出、非BLASLt与非Runtime异常原样传播，以及非HCU委托。昂贵backend在CPU夹具中可观察替代，GPU数学/Graph另测。

当前dev40 wheel SHA256 `577e634e1b5bd16dc761e4c3681124bf9e0308615f6c90e797a038e2f7322b6f`；579个包文件源码/wheel/site逐字节一致；400/400 Hook和3个具名seam严格加载，无软错误或未应用。dev39 MiniMax原八卡服务仍在本次运行中，不因等待另启同一模型；dev40的真实fallback GPU验证按实际执行另补。fp8_utils其他block/mxscale/channelwise opt-in差异仍未关闭，全量目标保持进行中。

## 46. 2026-11-01 FP8 多维scale与Triton混合bias控制适配

dev40真实GPU探针暴露两个固定source/native共有的问题：先把[M,N]恢复为多维输出后再乘per-token[M,1] scale，广播失败；原Triton kernel的 `c += bias` 在FP16输出与FP32 bias下导致分支SSA类型不一致、无法编译。dev41保持二维行域完成scale/bias后最终reshape；在同源优化kernel的bias分支显式执行原store本应完成的输出类型转换，保持先转换accumulator再加bias的顺序，不替换成普通矩阵乘法或外部逐元素fallback。

迁入wrapper时provider别名N又被局部输出维度N遮蔽，已改成native_fp8_kernel，并在方法抽取工具加入replacement/owner目标命名与局部绑定的预检及两个实际执行/拒绝回归。原始失败保留。更新后的[对照](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-fallback-source-alignment.json`）明确这两处有意适配，反向归一后恢复source fallback、wrapper、kernel AST，不再声称逐字源相同。

源树真实GPU三项已通过：输出[2,14,128]的per-token scale/bias/padding与BF16参考；M17/K6144/N768数值与变更scale的Graph replay；显式预置已知cache条目的M19真实Triton FP16+FP32 bias。当前设备对M17的真实hipBLASLt调用成功，不能冒充自然触发的失败/cache；选择/异常策略由公共CPU夹具独立验证。没有改变测试容差。

当前dev41 wheel SHA256 `774751e5a3c49382e8b3eadc5cbbf9e8801e18260911ad3fc9b4a97b6c195dd2`，580包文件与源码/site一致。隔离wheel的完整CPU/GPU/Hook与门禁结果按本次实际执行补记；尚不据此关闭fp8_utils其余定义、所有profile或全量DAS目标。

## 47. 2026-11-01 dev39 MiniMax 八卡复验

本地代码 `9976f34` 的隔离dev39按原MiniMax BF16八卡配方，02:52:18–03:02:41完成六个严格请求，正常stop、自有进程为空、端口关闭。TP8、FP8 KV、context131072、max-running512与默认Graph保留；578包文件与该已提交代码/实际wheel/site逐字节相同。[运行摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-minimax-dev39-validation-20261101.json`）保存实际artifact与响应哈希，不归到后续dev41。

隔离dev41实际结果：7项CPU、3项GPU数值/Graph与400/400 Hook、3seam通过；79项仓库unittest、L0、Skill和实际宽/窄屏检索/阅读/导航检查通过。真实M17与M32均未进入失败cache，已知cache夹具用真实同源优化Triton kernel。仅当前选择范围通过，其余fp8_utils定义及完整模型dev41验收不由此关闭。

## 48. 2026-11-01 通用HCU FP8 apply与条件量化生产端

本阶段从本地 `1951b9e` 延续FP8迁移。来源 `apply_fp8_linear` 在HCU上使用真实DeepGEMM接口并支持预量化输入，原插件没有该路径接入。现提取完整来源函数并沿公开Hook更新已导入alias；原生 `Fp8LinearMethod.apply` AST与DAS相同，直接保留其consumer，验证从原生类入口到新apply，而非旁路私有quant_method。

[源码/边界证据](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-linear-source-alignment.json`）明确两处受控buffer适配：直接carrier统一flatten token维及scale、还原输出leading维；output按实际q行数（包括padding）分配，截取真实token行后还原shape。源函数此前用未补齐行数分配GEMM输出，CPU严格backend夹具在q17/out2时失败，未在GPU上冒险执行越界配置。

真实GPU进一步发现Native ROCm scaled_fp8_quant在input8行/scale17行copy失败。DAS对此已有实际条件HCU分支：LightOp dynamic per-token返回未补齐且匹配的q/scale，tensor/static helper只写真实行。已将该分支及三个配套helper原样提取到owned fp8_quant，沿公开函数Hook传播到所有早期alias；保留Source capability/dtype/shape和错误边界，不只修consumer掩盖producer缺口。

12项CPU公共入口/原生consumer回归通过；源树4项真实DeepGEMM GPU检查通过，覆盖BF16/FP16输出、直接多维carrier、dynamic/compressed生产端及变更scale的Graph replay。fixture起初错误要求view后的Tensor对象identity，已改为storage指针与数值/参数检查；并未放宽数学参考或容差。dev42中间artifact保留，其compressed GPU失败指向原生产端；后续隔离dev43结果按实际执行补记。

fp8_utils、quantization/fp8_kernel的其余dispatch、post-load、block/mxscale及其他定义仍分别待审。此处完整选择函数、4个producer定义及未改原生consumer的来源对照不能代表整个文件或全DAS完成。

## 49. 2026-11-01 dev41 DeepSeek-V4 八卡复验

代码 `1951b9e` 的隔离dev41运行 `deepseekv4-fp8-tp8-dev41-20261101-attempt1`，六请求严格通过，自有进程与端口清理。保留原TP8/DP8/EP8、DSPARK/DeepEP、context32768、graph32及64线程加载；580包文件与源码commit/wheel/site逐字节核验。[独立验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev41-validation-20261101.json`）保存实际版本、响应/配置哈希和HCU backend日志，不归因于后续dev43。

本阶段隔离dev43实际结果：12项公开CPU producer/consumer、4项真实DeepGEMM GPU数值/Graph、401/401 Hook与3seam通过；582包文件与源码/site一致，wheel SHA256 `d04f014e94f3dbd32784902d9ac53805a0570587baeeea72674185affcd3a2e3`。79项仓库unittest、L0、执行台账、Skill及实际宽/窄屏检索/阅读/导航检查通过。测试只证明选择路径，block/mxscale/post-load及全量目标仍未完。

完整加载复核确认scaled_fp8_quant早已有kernel_compat动态分支hook，因此去重后的总target为401而非推算的402。旧局部wrapper与registration已退役，当前完整Source量化函数统一拥有入口；两层同时注册的中间dev43保留各自artifact，不以重复wrapper作为新能力数。最终dev44重新构建/隔离核验。

最终dev44隔离验证完成：12项CPU、4项真实GPU数值/Graph、401/401 Hook（errors与unapplied均为空）、79项仓库unittest通过；582包文件与源码及实际安装内容一致。wheel SHA256 `d9987b88b04eeb87c2321dc34fcf19414af8cf22fb1cfb5b7a0e633741aea6f7`。完整模型记录仍分别归属于其实际版本，不将dev41的八卡结果归到dev44；其余DAS迁移范围保持未完成。

## 50. 2026-11-01 HCU block-FP8 hipBLASLt 与加载布局

从已提交 `469976c` 继续迁移。DAS在DeepGEMM、FlashInfer、AITER优先项均不命中时，HCU auto选择LightOp hipBLASLt；对应加载方法把checkpoint [N,K]及[N/128,K/128]转成连续[K,N]与[K/128,N/128]。缺任一端都会形成错误consumer契约。

完整提取auto dispatcher、两个量化/GEMM helper和block post-load方法，[来源对照](hcu-sglang-development.md#历史-json-快照)（`hcu-block-fp8-source-alignment.json`）对4个函数归一native命名空间及本仓owner导入后AST相等。post-load Hook仅在实际选择本仓hipBLASLt callable时使用该来源方法；其他backend沿用有效原生方法，不提前替换其权重布局。计算仍使用来源LightOp per-token-group quant与hipblaslt blockwise GEMM，不替换优化backend。

源树7项CPU覆盖优先级、早期alias、原生加载/消费、参数identity、权重与非方形scale转置、backend参数和异常；3项真实GPU覆盖M1/17/32、BF16/FP16与bias、多维token输入，以及改变输入的Graph replay，全部通过。数学参考将真实量化activation与逐128块scale解码，再独立FP32乘积；容差rtol/atol均0.025。保留原来源忽略传入input_scale而重新量化的接口语义；不扩大为tuple carrier支持。

当前源码dev45；完整隔离wheel、全部Hook及仓库门禁验证按本阶段实际执行补记。AITER mxscale生产/消费、其他post-load分支及全量DAS仍未完成，原配方八卡dev44/dev45尚无新模型运行证据。

隔离dev45实际结果：7项CPU、3项真实GPU数值/Graph与403/403 Hook、3seam通过，errors/unapplied为空；583包文件与源码/site逐字节一致，wheel SHA256 `7041059d9e6f164fb6db763c7e91b7a9b93f8feceaa3d8a07c3e5e695b68af8c`。79项仓库unittest、L0、执行台账、Skill元信息及实际宽/窄屏浏览器检索/阅读/导航检查通过。八卡dev45服务另行执行并记录，不用局部测试代替。

## 51. 2026-11-01 dev45 DeepSeek-V4 八卡复验

本地 `0995491` 的隔离dev45从04:02:42至04:08:13完成原TP8/DP8/EP8、DSPARK/DeepEP、Graph32、context32768及64线程加载配方；六项严格请求均以stop结束，自有进程为空、端口关闭。583包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev45-validation-20261101.json`）保存版本、配置和响应哈希。仅六项确定性生成检查，不是数据集精度或吞吐指标，亦不归到后续dev46。

## 52. 2026-11-01 dense mxscale 准备、生产与消费

从已提交 `0995491` 继续：模型已有dense_mxscale_ready/set_dense_mxscale调用，却缺少公共符号和融合RMSNorm/e8m0来源模块。现迁入8个公共/配套helper、AITER完整consumer及加载后准备方法，绑定原生模块/类的缺失符号并检测冲突；block post-load仅对实际hipBLASLt或AITER consumer采用完整来源方法。保持capability检查、gfx95 bpreshuffle条件、shape白名单和正normal二次幂scale的无损判定；先shuffle权重，再附加e8m0 scale副本，共享生产端要求全部consumer ready。

模型融合生产端改用本仓byte-identical RMSNorm/e8m0 kernel；不假装官方存在缺失模块。[源码对照](hcu-sglang-development.md#历史-json-快照)（`hcu-dense-mxscale-source-alignment.json`）归一native owner后10个函数AST相等，模型helper仅重绑定import，融合kernel整文件逐字节相同。源树11项CPU路由/数据契约、7项block回归、3项真实gfx938融合GPU/Graph通过，覆盖权重/scale准备、早期alias、混合scale保留fp32 backend、缺失scale拒绝、空rank不launch、M1/17/8、K256/384/7168、零输入与改变input/weight的Graph replay。数值参考独立计算RMSNorm及ceil(log2(amax/448))量化，实际scale字节按column-major storage解释。

实际SGLANG_USE_AITER=1导入发现安装的vendor缺少gemm_a8w8_blockscale，来源/原生fp8_utils也无法导入，原始失败保留。CPU测试在实际可导入环境验证数据流并显式模拟gfx95条件，不能证明优化GEMM数学或性能；gfx938上的融合Triton生产端是真实执行，不能替代gfx95 mxscale bpreshuffle GEMM。未升级或替换vendor，未禁用优化分支掩盖失败。

测试夹具早期误把view要求为同一Python对象、缺少条件native import的fixture及解包错误已修正，原始日志保留。运行时guard使用getattr保留AITER关闭时原生模块没有该alias的正常情况；未修改来源数学或放宽容差。当前dev46隔离wheel/全Hook/门禁按实际结果补记；其余DAS与完整dev46模型未完成。

隔离dev46实际结果：11项mxscale CPU、7项block回归、3项真实融合GPU/Graph及404/404 Hook、3seam通过；585包文件与源码/site一致，wheel SHA256 `b2760cbc92f83cc260f6d50d186922303bba616ef9e85890d7b01c36ebf5ecdc`。本地79项仓库unittest、L0、执行台账、Skill元信息和实际宽/窄屏检索/阅读/导航检查通过。此结论不包含gfx95 mxscale GEMM、性能或完整dev46模型；本阶段dev45八卡验证单独归属 `0995491`。

## 53. 2026-11-01 dev46 DeepSeek-V4 八卡复验

提交 `5ba8e96` 的隔离dev46，04:17:30至04:23:01按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32、context32768、64线程加载完成六项严格请求，全部正常stop、自有进程为空、端口关闭。585包文件与提交/wheel/site逐字节相同。[运行摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev46-validation-20261101.json`）保存实际版本、配置与响应哈希，不归到dev47，不代表数据集精度或性能。

## 54. 2026-11-01 compressed FP8 的完整调用与加载契约

从本地 `5ba8e96` 继续。进程启动前显式设置 `SGLANG_USE_LIGHTOP_CHANNEL_FP8=1` 启用该路径，缺省关闭；它只选择compressed CHANNEL路径，不能代替block/tensor配置或全模型兼容验收。LightOp channelwise opt-in要求OCP E4M3权重[N,K]，不能继续使用普通HCU转置/FNUZ布局。完整提取来源helper、scheme capability/post-load/apply与上层CompressedTensorsLinearMethod.apply；复用来源与官方AST相等的6个native构造/建参/入口方法。公开capability限定HCU、非AITER、dynamic channel、非block；channel opt-in保留连续OCP权重，动态调用真实LightOp quantizer或直接复用tuple，并保留来源RuntimeError→KME重试及其他错误传播。

[来源对照](hcu-sglang-development.md#历史-json-快照)（`hcu-compressed-fp8-source-alignment.json`）归一namespace、property绑定和抽取方法名后恢复5个函数AST，并显式列出受控适配。AROUND保留legacy函数反射签名，为owned scheme提供来源参数元信息，其他scheme仍按来源inspect/cache过滤参数；不mock掉上层native wrapper入口。

真实CPU反例发现两个来源逻辑缺口：compressed caller已传pre_quant_output_dtype，但generic HCU tuple分支固定BF16；compressed block后端选择来源hipBLASLt时，加载后权重/scale未做其要求的成对转置。现分别令allocation遵守forwarded dtype（缺省仍BF16），仅对该block consumer转换weight及scale，保留Parameter对象与其他backend布局。[generic对照](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-linear-source-alignment.json`）追加dtype适配，不再把这处修复称作逐字源相同。

源树14项CPU从实际native构造/建参/加载/包装apply到backend验证通过：compat kwargs、其他scheme旧签名、capability矩阵、动态/预量化/FNUZ、opt-out DeepGEMM dtype、KME复用同一operand、错误传播和非方形block布局。初始fixture缺tp_group已补TP1上下文；原始dtype与布局失败保留。GPU初次bias使用FP32违反LightOp的bias.dtype==out_dtype契约，已改为实际模型对应dtype，未改运行时规避检查或放宽容差。真实GPU/Graph、隔离wheel和全Hook按实际结果补记；不据此关闭所有compressed schemes或全部DAS。

源树6项真实GPU测试全部通过：opt-in BF16/FP16预量化+bias、动态多维量化、opt-out tuple输出dtype、KME实际计算、更新activation/weight scale的Graph，以及native compressed block加载到实际hipBLASLt的非方形scale数学对照。自然预量化两种dtype均为primary1/KME0；KME数学由明确的primary miss夹具触发（KME1），不称为自然硬件失败。统一rtol/atol=0.025，未放宽容差；block反例由[256,512]原错误layout转为[512,256]后才通过。

隔离dev47实际结果：14项CPU、6项真实GPU/Graph、407/407 Hook与3seam通过，errors/unapplied为空；586包文件与源码/site一致，wheel SHA256 `3f9ba712b33e60c70e063b6e7b136175eadf56989988f7a60aac7111572331b2`。79项仓库unittest、L0、执行台账、Skill与实际宽/窄屏检索/阅读/导航检查通过，固定official/fork输入工作区复核干净。该artifact的原配方完整八卡模型验证尚待独立运行；dev46六请求证据仍只归属 `5ba8e96`。

## 55. 2026-11-01 dev47 两组原配方八卡复验

提交 `d2dc00d` 的隔离dev47先后完成[DeepSeek-V4](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev47-validation-20261101.json`）与[GLM-5 INT4](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-glm5-dev47-validation-20261101.json`）八卡配方，各六项严格请求均正常stop，已清理自有进程与监听端口。两次586包文件与提交/wheel/site逐字节相同，分别04:41:59–04:47:30、04:48:50–04:57:01，均只归到dev47。

DeepSeek保留TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768和64线程加载。GLM保留原TP8、INT4、EAGLE、FP8KV、默认Graph、memory0.85和显式AITER custom-AR；八rank IPC选择记录与multimem all-gather运行时capability降级日志独立计数，没有手动取消优化参数。GLM SlimQuant使用独立LinearMethod，属于组合回归，不能用它替代所有compressed schemes的验证。

## 56. 2026-11-01 FP8 utility 剩余定义与递归盘点

从本地 `d2dc00d` 继续，迁入完整显式backend选择函数与公开torch_fp8_block_gemm参考函数。HCU请求不可用AITER时，错误现在按来源指出HCU设备及SGLANG_USE_AITER=1，其他backend返回有效native/已hooked consumer；非HCU保留有效原分派。公开参考函数按来源使用per-row/per-K-block及per-K/N-block scale，不把参考Torch矩阵乘法接入优化计算fallback。

[定义/global对照](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-utils-source-alignment.json`）覆盖17个变化的顶层定义与62个AST未变的native定义，并核对7个顶层global及conditional import归属。Native CK表保留来源2行并额外保留(4096,2048):2048与(4608,4096):512两条上游ROCm7.0 NaN防护，不能为了消掉旧DAS差异删除这些已存在保护。其gfx95性能/数学依然未经当前gfx938验证。

递归FunctionDef检查发现顶层盘点外的CUDA-only _fp8_blockwise_scaled_mm_abstract，另有CUDA raw FlashInfer BMM从eager变LazyImport的来源差异，均尚未迁入。它们不会在当前HCU条件命中，但仍在本文件处置范围内，因此fp8_utils继续in_progress，不把顶层17定义核对称为整个模块完成。

源树6项CPU通过实际公开/早期alias检查显式backend身份及诊断，并用独立scalar double乘积验证ragged K/N block、三种输出dtype、输入不变和空token。2项真实HCU GPU/Graph将reference函数与CPU double按block求和对照，覆盖K257/N129尾块和更新scale的Graph；它是新公开debug/reference helper的设备验证，不代表优化GEMM新增或性能结论。

另外2项CPU使用真实CompressedTensorsLinearMethod及native W8A8 INT8静态/动态scheme，实际构造、建参、post-load、signature cache/kwargs过滤到apply均验证；仅量化和scaled-MM计算mock，证明旧签名路由不回归，不能声称INT8 HCU kernel验证，也不替代后续INT8 scheme来源迁移。隔离wheel、全部Hook及门禁按实际执行补记。

隔离dev48实际结果：6项utility CPU、2项真实native INT8 scheme旧签名路由CPU、2项reference GPU/Graph及408/408 Hook、3seam通过，errors/unapplied为空；586包文件与源码/site逐字节一致，wheel SHA256 `3269c3b34ed33d5e3358417b9af53f05e454526b22a2b83ce6d17c4675c08be8`。79项仓库unittest、L0、执行台账、Skill与实际宽/窄屏检索/阅读/导航通过。Generic来源反向比较新增精确受控分支AST检查，先验证允许的buffer/dtype适配再恢复Source，避免笼统替换整个分支掩盖额外修改。dev47两次八卡结果不能自动记为dev48；其余CUDA条件分支和DAS范围仍未完成。

## 57. 2026-11-01 dev48 DeepSeek-V4 原配方八卡复验

本地 `b0eafcf` 的隔离dev48从05:13:31至05:19:13按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768及64线程加载完成六个严格请求，正常stop，自有进程为空、端口关闭。586包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev48-validation-20261101.json`）保存实际版本、响应与配置哈希，不归到后续dev49，也不是数据集精度或吞吐指标。

## 58. 2026-11-01 FP8 MoE 建参、HCU分派与加载布局

从本地 `b0eafcf` 继续。来源HCU在FP4建参保留int8 packed bytes与FP32 scale，不使用generic ROCm的e8m0 scale初始化；generic runner-core AiterMoeQuantInfo在HCU返回None；standalone/grouped MegaMoE有各自格式边界；HCU AITER/ASM的block-FP8加载消费b8布局，不能套用generic16x16 shuffle或MXFP4路径。上述4个变化方法需配套迁移，不能只改单个dtype。

完整来源方法迁入owned fp8_moe，四个HCU-only AROUND沿原生Fp8MoEMethod绑定；非HCU保留有效original。Native HookRegistry保留static factory的staticmethod descriptor。既有MegaMoE注册先发布HCU runtime/builder，再注册quantization，避免导入未发布helper。ASM使用来源aiter.ops.shuffle.asm_shuffle_weight_b8 provider，在实际分支调用时加载，未伪造vendor接口。初值SGLANG_USE_AITER_FP8_ASM_MOE=False及gfx95 AND NOT HCU的MXFP4 shuffle条件保留来源语义。

[来源对照](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-moe-source-alignment.json`）覆盖Fp8MoEMethod全部4个变化定义，归一native P后AST相等；16个AST未变方法继续复用native，包括真实constructor、create、apply和FP4转换helper。来源的声明、注释和许可保留，说明放本记录；方法抽取阶段误带第二个future import，已在执行测试前修正header，未改来源计算语义。

10项CPU通过实际Native Fp8Config/Fp8MoEMethod构造、create_weights到static factory，验证FP4在AITer/e8m0 override请求下仍生成FP32 scale；验证HCU quant-info在no_combine assert之前返回、standalone非法格式先拒绝、grouped block转交、ASM顺序1/2且不走generic shuffle、packed FP4 int8保留、dequant早退与non-HCU original。MegaMoE builder及CPU ASM计算采用明确spy/mock，不作数学精度结论。

2项真实GPU均从native完整process_weights_after_loading入口执行：FP8 E2/H256/I256在真实已安装ASM b8 provider后，用独立逆排列逐字节恢复w13/w2，保留Parameter及storage；packed FP4用全nibble值与正二次幂scale模式，经真实native FP4->FP8转换后按128-block scale解码，与独立E2M1解码逐值相等，保留Parameter对象。它们证明布局/转换，不证明完整MoE expert routing、GEMM或性能；未把权重初始化反复Graph replay当作forward Graph验收。

当前容器全局SGLANG_USE_AITER=1的FP8 import仍缺vendor API；方法分支内的实际b8 provider可用，不等于整个AITer FP8配置已可部署。CPU patch分派标志与真实GPU provider证据分开记录。dev49隔离wheel、全Hook及完整模型按实际执行补记；全DAS、其余global/conditional/provider组合仍未完成。

追加非serialized BF16 expert的完整加载检查：从native create到owned HCU quant producer量化w13/w2，得到FP8权重与FP32 per-expert scale，解码后按FP8舍入界限rtol0.075/atol0.005比较原BF16值；不是放宽既有lossless FP4或byte布局断言。初次fixture缺实际FusedMoE的num_local_experts上下文，补齐2个expert后3项GPU通过。最终隔离用例另断言两组权重、2expert共4次调用确实进入owned producer，避免只凭数值推断alias已切换。

进一步补查真实routed FP8 consumer：以完整插件入口注册、真实ServerArgs发布exec上下文，TP1用公开parallel override显式声明无通信group；模型配置只读位置由HCU_TEST_MODEL_PATH提供。E2/H256/I256、M16/topk1实际native create_moe_runner→apply选择Triton，HCU alignment计数1，两次真实GEMM各自记录中间结果供独立Torch检查，未替换任何计算kernel。

最初仅登记FP8相关方法的测试漏掉了生产入口已有的HCU alignment，原生小batch CUDA alignment触发320线程超过256 launch bound并使后续fused kernel VM fault。GPU进程退出后设备内存恢复，未重置设备或修改来源；改用完整register复验不再触发故障。该失败属于不完整测试注册，不能描述成修改kernel后修好了生产故障。

直接BF16 dense参考漏掉两次activation FP8量化、中间BF16舍入，产生31/4096差异；补齐后单点量化阈值仍放大首层rounding差异。现分别核对gate/up GEMM与activation（rtol/atol0.025）、用实际中间值作down阶段输入再独立按128-group量化/分块Torch计算down及route，最终out维持原rtol0.08/atol0.03，不放宽门限。两次GEMM和SiLU均独立检查，实际forward结果未被参考值替换。源码环境单用例通过；最终隔离四用例复验按实际结果补记。

最终隔离dev49四项GPU用例全部通过（完整register、HCU_TEST_MODEL_PATH真实只读配置上下文、Native routed forward）；owned producer4次及HCU alignment1次断言通过。10项CPU、412/412 Hook、3seam和79项仓库unittest通过；587包文件与源码/site一致，wheel SHA256 `ad42689f38b850b6bdc6c53d8860e512ea9a2369c9a486386068bb94b7442f9f`。L0、执行台账、Skill及实际宽/窄屏检索/阅读/导航检查通过。完整dev49八卡服务仍待独立运行，不把dev48结果归到它；其他MoE后端/拓扑、global AITER配置与全DAS仍未关闭。

## 59. 2026-11-01 dev49 DeepSeek-V4 原配方八卡复验

提交 `dc59cd0` 的隔离dev49在06:11:43–06:17:19按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768及64线程加载完成六项严格请求，均正常stop，自有进程为空、端口关闭。587包文件与提交/wheel/site逐字节相同。[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev49-validation-20261101.json`）保存实际版本、配置及响应哈希；不归到dev50，不代表所有MoE profile或数据集精度/性能。

## 60. 2026-11-01 compressed INT8 strategy 与预量化入口

从本地 `dc59cd0` 继续。来源使用W8A8_SUPPORT_METHODS选择1 Triton、2 cutlass入口（HCU实际委托rocBLAS NN）、3 hipBLASLt、其他值rocBLAS；不同strategy的权重布局必须与其provider一起迁入。默认值1；变量应在进程启动前设置。完整迁入constructor、static scale normalization、post-load与apply，使用既有owned quant_ops/cache_config，来源gfx928 KME predicate逐字保留。普通gfx936/gfx938 strategy3保留连续[N,K]；gfx928按来源[K,N]、stride(1,K)打包，当前设备无该硬件验证。

BF16 checkpoint scale在加载前转FP32；strategy1按已有cache reader读取指定TRITON_JSON_DIR，只有提供的config才预热，LMSLIM_TUNING_JSON显式存在时才导出model.json。未向输入仓库或模型目录写文件。SGLANG_USE_LEGACY_FUSED_RMS_QUANT与SGLANG_USE_FUSED_SILU_MUL_QUANT仍缺省关闭，启用后消费对应input_quant_args/silu_quant_args，RMS优先。Native linear wrapper读取owned scheme参数元信息，避免AROUND旧反射签名吞掉新增参数。

[源码对照](hcu-sglang-development.md#历史-json-快照)（`hcu-compressed-int8-source-alignment.json`）归一native N后4个变化方法及架构predicate AST相等，create_weights/get_min_capability复用AST未变native方法；沿用已存在的owned cache兼容类，不把其实现称为本阶段逐字Source迁入。Non-HCU保留有效original。Legacy旧签名测试只登记wrapper，保留真实未迁入native INT8类作接口兼容单元测试；生产GPU用例登记完整插件，这与forward测试漏登记alignment的错误情形不同。

严格public CPU反例确认共享legacy hipBLASLt helper接收bias却直接返回不含bias的GEMM结果，6/6元素全部丢失。现仅在该返回路径用in-place add加入非None bias并保留输出dtype；精确移除该AST增量后恢复完整来源helper。未替换kernel、未禁用hipBLASLt；gfx928 KME原有bias处理保持其provider接口。

13项CPU覆盖真实native constructor/create/post-load/wrapper apply、BF16 scale、gfx928 stride/AZP元信息、tensor scale、cache预热/m bucket、两种预量化参数优先级、四provider分派及异常、bias反例和non-HCU委托。4项真实GPU覆盖四种strategy实际backend、普通动态INT8量化、RMS pair进入真实hipBLASLt、SiLU pair进入真实Triton、以及更改scale/bias后的Graph重放；数学参考使用实际量化结果及独立FP32矩阵乘法，rtol/atol0.025。Static/asymmetric仅核对来源metadata，不能冒称AZP kernel数值通过；gfx928也仅CPU契约通过。隔离wheel、全Hook与模型结果按实际执行补记。

隔离dev50实际结果：13项INT8 CPU、4项真实GPU/Graph、2项指定wrapper旧签名兼容单元测试、415/415 Hook与3seam通过，errors/unapplied为空；588包文件与源码/site逐字节相同，wheel SHA256 `2d897fad205952c99b6c66058937d2918da0bed1c7222bb55359c6b466b94997`。本地79项仓库unittest、L0、执行台账、Skill及实际宽/窄屏检索/阅读/导航检查通过。新增helper bias按独立AST增量记录；完整dev50模型未由这些局部检查替代，dev49模型结果仍仅归属dc59cd0。

## 61. 2026-11-01 dev50 GLM-5 INT4 原配方八卡复验

提交 `814792e` 的隔离dev50于06:31:46–06:37:17完成原TP8、INT4、EAGLE、FP8 KV、memory0.85、默认Graph和显式AITER custom-AR配方，六项严格请求均正常stop，自有进程为空、端口关闭。588包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-glm5-dev50-validation-20261101.json`）记录实际响应、配置和backend日志计数。GLM走独立SlimQuant，本次是共享helper/组合回归，不作为新compressed INT8 scheme的全模型证明，也不归到dev51。

## 62. 2026-11-01 small-M ASM 与量化 target 父类匹配

从本地 `814792e` 审查相邻pending文件发现：既有quant_ops虽已迁入大部分helper，却缺来源small-M ASM fast path及其致命binary缺失架构保护；Native matching只看当前类名，不能识别不含Linear命名的自定义子类。现迁入完整来源7个helper和匹配函数，保留默认Triton/显式config/bias/layout/shape gates，以及来源异常后Triton fallback。m/n的slide1/9选择、完整K256 tile要求、gfx928禁用与未知arch probe失败均按来源保留，未为了通过改掉kernel选择。

当前官方utils没有is_hcu导出，owned module将该导入绑定到既有detection.is_hcu；未给官方模块伪造符号。Source quant_ops全部函数与module其余AST均核对，除这处真实provider重绑定及dev50精确bias AST增量外相同。[全文件对照](hcu-sglang-development.md#历史-json-快照)（`hcu-quant-ops-targets-source-alignment.json`）还证明target utility唯一变化函数归一native N后相同，其余完整module AST与native相同，因此这两项代码处置改为implemented；不据此把全部DAS或所有profile改为完成。

Target按exact/regex layer path、Python MRO中的leaf/parent、fused全组件匹配的顺序选择，跳过object，支持空字符串显式匹配。公开Hook传播到早期alias，并保留non-HCU有效original；不因自定义layer名称改变而漏配quantization。

6项CPU验证ASM架构缓存、source slide与provider完整参数、bias/tuned config/partial K/large M回到原Triton、gfx928不调用会exit的provider、exception和non-HCU边界；6项CPU验证target inherited-only、leaf/regex/path/MRO优先级、fused-all/object/空值。3项真实gfx938 GPU验证自然small-M命中ASM1/Triton0、带bias命中ASM0/Triton1及实际ASM Graph读取scale更新，独立FP32数学参考rtol/atol0.025。gfx928只测保护契约，不作硬件执行结论；没有性能A/B结论。隔离artifact、全Hook和消费者回归按实际结果补记。

追加真实native CompressedTensorsConfig.get_scheme_dict消费者检查：自定义HiddenProjection仅通过父类LinearStem命中，返回实际QuantizationArgs payload；用于验证native config模块内的早期import alias也更新。该用例将target CPU检查增至7项，隔离执行结果单独记录。

隔离dev51实际结果：6项分派CPU、7项target CPU（含真实native config消费者）、3项ASM/Triton GPU/Graph和4项实际compressed INT8 backend回归均通过；416/416 Hook、3seam与79项仓库unittest通过。589包文件与源码/site逐字节相同，wheel SHA256 `8d2c37a703cafbcea4f148a34a6a79492c59c2337715ff42273d059020535635`。L0、执行台账、Skill及实际宽/窄屏检索/阅读/导航检查通过。Source helper7函数完整覆盖与config utility单一变化覆盖只支持对应两项implemented处置，未关闭其他文件或全DAS。另有106个pending来源路径与owned文件字节相同，仅作为后续编译/注册/caller审查线索，未凭此更改状态；完整dev51模型尚未运行。

## 63. 2026-11-01 dev51 GLM 原配方八卡复验

本地 `701fce6` 的隔离dev51于06:49:40–06:55:01按原GLM INT4 TP8/EAGLE/FP8 KV/默认Graph/memory0.85和AITER custom-AR配方完成六项严格请求，全部正常stop，已清理自有进程与监听端口。589包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-glm5-dev51-validation-20261101.json`）保存实际配置、响应与backend日志计数；不归到dev52，不是数据集精度或性能分数。

## 64. 2026-11-01 JIT 资源的可重复编译闭包审查

前阶段106个pending同字节路径只是线索。本阶段新增只读工具[资源审计](../tools/audit_jit_resources.py)，从固定fork Git tree核对plan OID、owned文件mode/bytes及wheel成员，并验证历史sgl_deps.json的deps-key、库文件存在、历史安装与当前owned内容摘要。清单与路径被当作数据，不执行来源代码或载入.so；输出不修改处置状态。工具6项单元测试覆盖来源HEAD/OID漂移、缓存清单篡改、库缺失、历史/当前内容漂移、wheel遗漏、外部安装路径与来源删除，并核对Git index/状态未改变。

本机历史review为14个recorded compiler dependencies、74个仅同字节owner、4个来源已删除；[覆盖摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-jit-resource-coverage.json`）保留代表性manifest及哈希。历史编译记录不是本次执行、更不是所有模板数值通过。这14项先转in_progress审查，74项和删除项不自动完成；输出目录由显式变量指定，未改写源仓库或全局Git配置。

4项native load_jit/BuildSpec检查从早期import alias进入真实loader，在compute_build_key前截获不可变spec，核对owned cuda_files、include首位、外部绝对路径与native未覆盖路径、重复注册和调用方list不变。Header-only的实际include保存在translation_units().source，而非ninja文本的cuda.cu文件名中；原测试断言位置已修正，未更改loader来迎合测试。

冷缓存实际调用owned candidate_blocks.amax8_varlen暴露block_amax.cuh的HIP编译错误：__launch_bounds__中带<typename, typename>的表达式逗号被宏解析，后续报BlockAmaxKernel不存在并不代表Python模板实参顺序错误。现只给两个常量表达式加括号，线程512、occupancy4及算法不变，[精确反向字节对照](hcu-sglang-development.md#历史-json-快照)（`hcu-jit-block-amax-source-alignment.json`）证明除这处适配外保持来源。没有换Torch/Triton fallback或修改输入树。

新任务cache目录强制经历实际JIT编译，2项gfx938 GPU检查覆盖FP32/BF16 input/output四组合、空/partial/skip rows、最新块+inf、未写区域保留，以及改变score/length的实际Graph replay，逐值一致。四个新编译清单也被审计；该header被正确标成owner_differs而非伪称同字节。[Source/覆盖记录](hcu-sglang-development.md#历史-json-快照)（`hcu-jit-resource-coverage.json`）将受控适配、当前实际GPU测试与历史编译分开。对应block-amax实现完成；其余资源的模板、caller与profile仍待审，不能把14份历史记录视为整仓完成。

工具复跑：将FORK、PLUGIN、WHEEL、CACHE、INSTALL_BASE、REPORT分别设为只读fork、插件目录、待核wheel、JIT cache、历史隔离安装父目录和新输出文件，然后执行：

```bash
python3 tools/audit_jit_resources.py --plan "$PLUGIN/plans/sglang-das-execution.json" --fork "$FORK" --plugin-root "$PLUGIN" --wheel "$WHEEL" --cache-root "$CACHE" --install-base "$INSTALL_BASE" --output "$REPORT"
```

运行工具需要Python3.10+和当前用户对上述只读输入的权限；cache可省略，此时不产生编译证据。GPU用例使用显式SGLANG_JIT_CACHE_DIR独立空目录做冷编译；后续隔离wheel/门禁与完整模型按实际执行补记。

隔离dev52实际结果：6项audit工具单元、4项native loader/BuildSpec CPU契约、2项实际GPU/Graph（四dtype编译变体）全部通过；416/416 Hook、3seam与85项仓库unittest通过。589包文件与源码/site一致，wheel SHA256 `4cfeeee574672743e8cb9424207abfb6413cd07504edc3856c7c182c088bf47b`。隔离cache的4份新依赖清单与库文件及当前owner摘要核对，owner_differs状态按语法适配保留。L0、执行台账、Skill和实际宽/窄屏检索/阅读/导航检查通过；当前完整dev52八卡服务仍待执行，不能归用dev51模型结果。14个历史资源保持in_progress，74个只同字节与4个删除项不作执行/完成声明。

## 65. 2026-11-01 dev52 DeepSeek 原配方八卡复验

本地 `b2302c3` 的隔离dev52于07:23:57–07:29:23按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768及64线程加载完成六项严格请求，均正常stop，自有进程为空、端口关闭。589包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev52-validation-20261101.json`）记录版本、配置及响应哈希；不归到dev53，不是所有kernel/profile或数据集精度证明。

## 66. 2026-11-01 BF16 short-row radix top-k 的HCU接入

从本地 `b2302c3` 继续选择仅有字节存在证据的topk_bf16_small.cuh，通过既有公开topk_transform_bf16_small冷编译。原Source在gfx938报5类编译错误：CUDA shuffle、signed dot打包与BF16 pair mask intrinsic不可用，原始日志保留。pivot原为固定32lane/每lane8个bins；后续dev54核对确认当前Source使用逻辑warp32，不能把硬件wave64等同于该逻辑宽度。

现使用既有portable shuffle/broadcast；每lane管理256/deviceWarpThreads个bins；当前Source utils.cuh固定逻辑warp32，CUDA与HIP均为8，不能按硬件wave64推断为4，对应vector load、signed-MSB顺序和suffix pivot按实际宽度分配。HIP pair比较直接使用BF16原始位序，显式处理NaN与signed-zero相等；8个半字比较mask用整型shift打包，CUDA保留原intrinsic。线程512、occupancy3、两阶段radix、top-k quota和page-table接口保持，未强制wave32，也未换用torch topk计算返回值。来源已有说明保留，CUDA特定lane注释按原文保留；HIP差异由本记录说明。

[Source对照](hcu-sglang-development.md#历史-json-快照)（`hcu-bf16-topk-source-alignment.json`）列出全部适配；当时保存的差异补丁经临时副本反向 apply 后与固定 Source 逐字节相同，补丁现可从 Git 历史读取。首轮helper类型限定误将全局bf16x2_t写成device成员，已修正后重新冷编译；未伪造vendor API。

5项公开入口真实GPU检查通过：256宽下空/partial行与任意返回顺序下的精确page slot集合；来源最大width16384/k2048与positive/negative BF16 bins；重复值与signed-zero quota128、唯一且有效索引；有限subnormal位序；Graph中同时改变score、length和page table。普通/最大值参考使用独立topk/page公式，subnormal参考使用整型BF16 rank，均为exact比较而非放宽浮点容差。

Subnormal首轮64/64差异不能归因为GPU：默认CPU float top-k给出129开头索引，关闭CPU flush-denormal后与整数位序同为63开头，kernel整数参考结果正确。初次将该失败直接归因HIP浮点比较的判断不充分，原probe/log保留；最终HIP比较按BF16位语义实现并避免任何flush依赖，不把参考环境修正冒称生产GPU bugfix。NaN沿来源接口不支持；CUDA32仅静态保持原intrinsic/8bin specialization，未在NVIDIA实测。

隔离dev53实际结果：独立空JIT cache冷编译并通过5项真实GPU/Graph检查；416/416 Hook、3个resolution seam、0 errors与85项仓库unittest通过。589包文件与源码/site逐字节相同，wheel SHA256 `3de6fde9d43a45cb9a153a742fd1a30b73766577e63c4ea7ead2411f7b559d78`。资源审计记录新cache的4项编译依赖，受控patch仍标记owner_differs，不混同字节一致与执行证据。完整dev53模型服务尚未执行；该单资源实现完成不关闭其余JIT资源和全DAS范围。

## 67. 2026-11-01 dev54 MoE hash routing 与空 pre-dispatch 接入

核对固定DAS的kernels/ops/moe/dsv4.py与官方差异：HCU应选择已有owned hash_topk JIT而非通用HIP Triton；空输入的非标准stride需在mega-MoE pre-dispatch之前恢复标准空布局。本阶段保留两个完整Source函数与既有native JIT factory，新增HCU范围AROUND并保留非HCU有效原调用链。入口、pre-import alias、真实GPU/Graph、数值与隔离产物验证待实际执行，当前不作完成声明。

## 68. 2026-11-01 dev53 DeepSeek 原配方八卡复验

隔离dev53对应提交 `acb7ae6`，按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768与64线程加载完成六项严格请求，均正常stop，自有进程清空、端口关闭。589包文件与提交/wheel/site逐字节相同；[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev53-validation-20261101.json`）记录时间、版本、配方与响应哈希。该结论属于dev53，不归用到新增MoE ops Hook的dev54，也不是数据集精度或全DAS迁移完成证明。


dev54边界复核：owned utils.cuh的device::kWarpThreads固定32，hash接口保持topk_fused<=32；首轮64路测试超出Source契约，拒绝属于预期边界，不能据此扩大接口。dev53的top-k constexpr仍按实际逻辑32选择8bins，真实5项GPU结果有效，但先前“HIP64为4bins”的解释不实，已更正。未改全局warp常量或强制底层执行模式。


dev54隔离GPU的7项边界检查中6项通过，零token hash触发Source的零grid HIP launch错误。dev55在hash与同header的mask host入口完成参数核验后、发射前对num_tokens==0直接返回；非空数学、warp、PDL、路由与page布局不变。当时保存的差异补丁保留 Source 差异，现可从 Git 历史读取；失败dev54 wheel/cache/log不覆盖。dev55冷编译与隔离重验待实际完成。


隔离dev55结果：7项真实GPU/Graph检查全部通过；4项CPU契约及85项仓库测试已在本阶段执行，当前418/418 Hook、3个resolution seam、0 errors。590包文件与源码/site逐字节相同，wheel SHA256 `8e25ca64434c064581b177ee7282fe7b75c401bc8b23320ff66d93fc10c11fac`。hash/mask两份新JIT清单均引用当前owned header，精确patch反向恢复固定Source；资源审计仍将受控差异标为owner_differs。完整dev55原配方服务待执行，不挪用dev53结果；其余DAS待办保持原范围。

## 69. 2026-11-01 HCU BF16→FP32 GEMM 的Source与执行核验

继续使用已冻结dev55验证既有owned实现，不把证据补齐伪装成新运行时版本。HCU helper全部4个函数AST与Source相同；gemm dispatcher全部17个函数在仅重定位函数内owned helper import后AST相同。平台检测与模块路径由owned命名空间承接。[对照记录](hcu-sglang-development.md#历史-json-快照)（`hcu-bf16-fp32-source-alignment.json`）保留两份来源/owner摘要。7项DAS原始dispatch CPU测试迁入本仓执行环境并通过。

Source自动调优profile限定gfx936，当前机器gfx938应保留原策略，不能为让原始GPU测试中的SGL断言通过而伪造profile；另通过显式sgl选择检查gfx938真实内核。待执行普通及split-K、M16/M8边界、最大M64、输出清零、独立CPU double参考、真实公共Hook和Graph重放；不预先声明GPU或性能结果。


dev55八卡于Graph捕获失败：HashTopK收到BF16 router_logits，Source JIT只收FP32，自有进程和端口已清理。沿producer定位Source MoEGate.forward在HCU DSV4通过linear_bf16_fp32生成FP32，而当前native gate走非CUDA的F.linear产生BF16。dev56迁入完整Source gate构造与forward，保留router配置dtype、输入weight dtype转换、CP边界、tiny/AITER/auto优先级及已有条件；不在hash consumer强制cast。所有Source说明与许可保留，super绑定到原MoEGate的父类，非HCU保留原链。CPU/GPU producer→consumer和原TP8配方复验待执行。


Source工作树执行结果：gate的5项CPU与2项真实producer→hash GPU/Graph通过，GEMM的7项原始CPU方法和6项GPU通过（10个直接kernel shape，含普通/split8/split4、M16和M8边界、M64、重复清零及split-K Graph）。GEMM普通/显式路径用CPU double参考，rtol2e-4/atol2e-5；精确整数fixture为0/0。初次auto参考错误地假设FP32累加输出，Source在HCU明确先BF16输出再.float()，现按相同Source输出契约将CPU double结果round到BF16再FP32，exact通过；未改运行时后端或放宽显式kernel阈值。

[dev55失败摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev55-failure-20261101.json`）单独保留原配方、提交、590包文件一致性、实际dtype错误与清理，不将失败运行计为通过。新gate的[Source对照](hcu-sglang-development.md#历史-json-快照)（`hcu-deepseek-gate-source-alignment.json`）明确仅两方法完成，deepseek_v2.py其余差异仍待审计。dev56隔离wheel/完整原配方验证按实际结果补记。


隔离dev56实际结果：gate 2项GPU/Graph与5项CPU、GEMM 6项GPU与7项原始CPU均通过；420/420 Hook、3个resolution seam、0 errors与85项仓库unittest通过。591包文件与源码/site逐字节相同，wheel SHA256 `ace94cd214bf842056eb11fb102b9b9e0c47f60a3feffccbcf9f396dc84859a5`。新cache依赖清单中的linear header与固定Source/owner/wheel同字节，真实执行由GPU用例独立证明。当前只证明gate两方法和已核对的GEMM实现，完整dev56原配方与其他DAS差异继续待验。

## 70. 2026-11-01 dev56 原配方八卡验证与MoE路由全量接入

隔离dev56/提交34ff4e9于08:33:06–08:38:46按原TP8/DP8/EP8、DSPARK/DeepEP与Graph32/context32768完成六项请求，均正常stop、自有进程清空与端口关闭。591包文件与提交/wheel/site相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev56-validation-20261101.json`）固定产物与响应摘要。证明gate producer修复越过dev55原故障，不代表所有DAS功能完成。

继续迁入完整TopK与HashTopK来源模块及合成EP均衡helper；已确认相同AST的9个共享类型复用native身份，方法super显式绑定原类父级，保留来源许可/注释和条件custom-op注册。替换旧单分支softmax注册入口，HCU范围接入全部7个变化helper、TopK的3个变化方法与HashTopK的2个变化方法。模拟EP均衡按Source默认关闭，仅作合成性能基准，不用于精度结论。Source对照、导入/契约/GPU与隔离/原配方验证待实际执行。


首次真实LightOP GPU检查3/5通过；V4.1 int64/padding融合两项在Source遗留函数名_eplb_remap_enabled处NameError，该文件实际定义的是is_expert_remapping_enabled。仅将该调用修正到实际Source谓词，保留所有融合条件和开关，并增加EPLB开启必须拒绝融合的策略回归；不通过关闭融合规避错误。


Source路由的6项CPU契约与EP helper的5项独立Python oracle检查通过，原class与owned factory均能构造，9个共享类型身份一致。隔离dev57的7项GPU/Graph suite与1项额外DeepEP共享槽位检查通过：真实LightOP softmax、sqrtsoftplus、V4.1 int64/padding融合及改变logits/有效行数的Graph；TopK/HashTopK合成模式保留weights/shared列；共享槽位只重映射一次，并在最后屏蔽padded routes。595包文件与源码/site相同，wheel SHA256 `8eddc8e9a4ed2d877fc8765f2a0e6ac4f5060e83b3f9ae18ff20592c60f1edd3`。

[完整Source对照与patch](hcu-sglang-development.md#历史-json-快照)（`hcu-topk-routing-source-alignment.json`）可反向恢复固定Source字节。EPLB负向单元初次未发布server args，按Source契约代表“无实际mapping”；补齐发布状态fixture后，EPLB开启正确拒绝融合，没有修改predicate本体或禁用优化。TopK/HashTopK文件继续in_progress：当前native LPLBSolver.solve只收topk_ids，Source新增load-aware solver接收有效token数，实际solver/collective尚待迁入验证；全局AITER环境仍未完成兼容，不从LightOP结果外推。完整dev57原八卡配方继续待验。


隔离dev57收尾：431/431 Hook、3个resolution seam、0 errors与85项仓库unittest通过；8项实际GPU/Graph和11项CPU证据按上述范围记录。L0/Skill/guide门禁及实际宽窄屏检索、阅读与菜单检查通过。原配方八卡服务仍需以本次产物另跑，不归用dev56模型结果；全部DAS范围保持未完成。

## 71. 2026-11-01 load-aware replica solver 接入

迁入Source Triton count/water-fill/sample、完整solver模块与dispatch函数，复用未改变字段的native dispatch-info类身份；接通初始化工厂、ModelRunner初始化门控和重平衡后的solver重建。CLI只扩展现有action的choices，验证沿原解析链追加Source load-aware的A2A/ROCm/冗余expert约束，不改冻结seed字段或Struct定义。Source工厂错误地将静态LP专用kwargs传给load-aware构造器；现按算法传参，并拒绝以静态LP概率配置load-aware。Source核验、CLI/worker往返、边界、实际GPU与Graph均待验证，不预先声明功能完成。


Source GPU边界6项初测3项失败：drop/shared ID会越过counts切片并修改测试私有前后哨兵；非连续routed列在count与sample中被按dense展平读取。现count对logical ID加0..num_logical范围mask，dispatch对无效ID返回-1并屏蔽table读取，两处公共入口先连续化真实routed布局；保留water-fill算法、采样、有效行数设备指针和无collective设计。原失败日志与哨兵fixture保留，正确性/Graph重验待实际完成。CPU静态LP的原始全宽slice是共享存储view，测试改为核对内部缓存身份及原tensor相同data_ptr，没有改LP运行时。


Producer→dispatch集成检查发现metadata并非仅Literal变化：Source还新增layer_id/source_rank采样种子字段，native dataclass不能接收该kwargs。现用完整Source dataclass继承native，保留isinstance兼容并通过原init_new classmethod产生完整owned实例；legacy手工info缺少种子字段时采用Source默认0。Native类型对象保留，返回类型有意是兼容子类，不再声称两者身份相同；新增factory→solver→TopK/Hash consumer检查待执行。


抽取ModelRunner方法的集成模块补齐继承自Source model_runner.py的原版权/Apache声明；dev58已冻结，不覆盖其wheel，以dev59保存该许可声明并重验当前产物。该项不添加说明性runtime注释或改变函数计算逻辑。


隔离dev58实际结果：8项真实kernel GPU检查（含256x264完整padded replica表）、2项原生factory→metadata→solver→TopK/Hash pipeline GPU、7项CPU与85项仓库测试通过；完整CLI采用原八卡配方加load-aware/冗余expert参数，正常ServerArgs post-init/resolution与pickle往返均保持算法和值，不跳过解析阶段。只表示配置与单设备数学/调用链成立，未启动load-aware模型服务。

Source load-aware本来就不做每层EP collective，不能把它描述成尚待补all-reduce；仍需以实际8卡dispatch/model变体验证组合。原dev57原配方已于09:02:55–09:08:39六项请求全过、自有进程为空、端口关闭，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev57-validation-20261101.json`）固定595包文件与8af8abf/wheel/site同字节。该结果不归用到当前新EPLB产物。


隔离dev59收尾：600包文件与源码/site逐字节一致，wheel SHA256 `7676b5ab0ef16aba4edf831119e7e8298ddfb60644fc944543f401f687ed9a00`。当前artifact的2项public GPU pipeline、7项CPU、437/437 Hook、3个resolution seam、0 errors及85项仓库测试通过；完整CLI/resolution/pickle再次通过。与dev58相比，唯一package差异是integration.py继承许可头，函数AST相同，8项kernel GPU测试对应的全部代码字节相同；保留分产物证据，不把历史运行冒称当前重跑。L0/Skill/guide与实际宽窄屏验收通过。后续以原八卡配方加显式load-aware/冗余8专家参数运行功能变体，原参数保留；尚未声明模型精度、跨rank负载或性能。

## 72. 2026-11-01 完整replica元数据与静态locality-fair策略

原生_compute_logical_to_all_physical_map会对所有A2A算法折叠nearest副本，Source只允许static+nearest折叠；这会使load-aware/LP看不到完整候选。迁入Source全部4个变化/新增helper，复用其余native元数据类与helper；仅在原生尚无ep_static_dispatch_policy字段时通过进程配置读取Source默认nearest与locality_fair，标准CLI参数由插件sidecar管理并在seed解析前传给workers，不写未声明的冻结字段。新增策略仅允许static，保留GPU→node→remote局部性优先与静态source-rank公平，不伪称动态token负载均衡。当前dev59服务保持其冻结产物，新的元数据验证与后续8卡复验待执行。


Source CPU 8项通过：load-aware/LP/dynamic全候选、static nearest与旧函数相同、locality_fair优先GPU/node与静态公平、非法topology、CLI/worker进程环境继承、规则与非HCU回退，以及原生metadata builder→真实solver保留复制专家计数。初次对照把原函数存成模块变量，被正式alias propagation一起替换，修正为容器保存原callable后，旧函数确实只保留1候选，新函数保留2。该fixture修正没有改变生产逻辑。原dev59变体09:43:15–09:53:14六项请求通过并清理，[摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-load-aware-dev59-validation-20261101.json`）明确旧候选折叠限制，不能据此关闭均衡功能验证。


隔离dev60的2项GPU检查通过原生metadata工厂→真实solver→原生公开dispatch，完整候选使num_replicated为4而非退化；GPU static/locality_fair元数据表也保持全部候选及正确device/dtype。8项CPU通过并验证sidecar值在worker子进程中继承；完整正常CLI/ServerArgs post-init/resolution/pickle对static/locality_fair+冗余8参数通过。601包文件与源码/site一致，wheel SHA256 `43e2eb0f125976ee6b29a008bcf068ba08a9be66cb8680d72c19b4ca7d304c99`。模型尚待重跑，单设备rank-metadata fixture不等于实际跨rank负载或性能结论。


隔离dev60收尾：439/439 Hook、3个resolution seam、0 errors及85项仓库unittest通过；L0、Skill元信息、guide快照与实际宽窄屏检索/阅读/菜单检查通过。增加启动日志汇总每层实际复制逻辑专家数，后续完整候选八卡运行需看到非零replica计数，不能仅以六个回答通过判断均衡链已生效。全DAS与实际负载/性能范围继续未完成。

## 73. 2026-11-01 EPLB dispatch统计与固定形状int64归约

dev60实际八卡启动日志已在全部8个rank记录43层load-aware solver、每层复制逻辑专家数[8]，与旧nearest折叠不同；服务仍运行，未提前记录请求通过。继续迁入expert_distribution.py全部变化：capture期间不collect；DeepEP auto从normal/low-latency实际dispatch累加，避免padded top-k id误计expert0；窗口先本地sum成固定形状int64再all_reduce，支持rank步数不同与大计数。复用native recorder/gatherer/accumulator类和其余方法，新增auto类继承native低延迟gatherer；保留Source许可/注释。CPU、真实Gloo归约和GPU检查按实际结果补记。


Source统计器验证：5项CPU与2项真实GPU/Graph通过，DeepEP auto normal+low-latency累加真实buffer，伪top-k padded IDs不增加计数；实际CUDAGraph capture跳过collect/host append。真实2-rank Gloo的窗口分别为1/3步和0/2步，归约shape固定[1,2]、dtype int64，输出[40,48]与[2**32,6]精确。该证据是CPU Gloo，不冒称GPU collective或性能；隔离8-rank NCCL/RCCL正在执行。

[dev60八卡摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-load-aware-dev60-validation-20261101.json`）记录10:02:15–10:07:53六项请求正常stop、全部8rank各43层solver与复制专家[8]、601包文件与13497b5/wheel/site相同以及清理。它验证完整候选功能变体的短生成，不是数据集准确率或实际dispatch负载均衡量化结果；新的统计器后续另验。


隔离dev61：602包文件与源码/site逐字节一致，wheel SHA256 `5267f9717f0e1db1c0827900dca68f95692d234ab5fab28021473f8cba7cc3f0`。5项CPU与2项GPU/Graph重验通过；真实8-rank NCCL/RCCL在各rank本地窗口1..8步时全rank得到[480,216]，七个空窗口与rank7两个大计数窗口全rank得到[2**32,6]，shape[1,2]/int64保持，process group销毁正常。[独立collective摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-expert-recording-collective-validation-20261101.json`）固定各rank结果哈希与范围；这是合成counter correctness，不是模型dispatch负载或通信吞吐。

后续模型验证使用单独的任务controller，在本人loopback服务ready后调用公开start/stop/dump recorder端点并保存响应，原始controller不改；模型参数沿dev60完整候选load-aware变体，仅追加stat recorder与明确dump目录。API/实际dump与模型结果未完成前不声明recording服务集成通过。


隔离dev61最终门禁443/443 Hook、3个resolution seam、0 errors及85项仓库测试通过，L0/Skill/guide与实际宽窄屏交互验收通过。原始Source变更函数/class AST经native globals和staticmethod搬移规范化相同；新服务与EPLB真实模型统计仍保持待验，全DAS范围未关闭。

## 74. 2026-11-01 DeepEP Graph capture/replay layout 所有权

Source runner_utils/deepep_adapter.py完整4方法新增capture_end保存LL layout、replay先恢复layout再设置dispatch mode；Source decode runner.capture_one_shape在backend实际capture完成后调用capture_end。当前owned DeepEPBuffer已有layout API但缺少这对调用链，现迁入完整Source adapter（继承原native class以保留isinstance）及完整Source producer方法，仅将未变native globals显式限定。生产者和消费者同时接通，不只修改replay或假设上一模型布局仍正确。来源注释/版权/docstring保留，不新增说明性runtime注释。CPU顺序/跨模型状态与实际Graph/缓冲区/八卡复验待执行。


实际dev61录制服务六项生成与start/stop成功，8rank均记录after all_reduce；rank0在file dump分支因搬移时遗漏time全局限定而NameError，API连接断开，服务判failed并清理。这是本轮实现缺口，保留原失败证据；修正为native N.time并新增原生_StatAccumulator.dump(file)端到端落盘测试，而不是只测_dump_to_file helper。新Graph producer测试目标类名纠正为当前DecodeCudaGraphRunner；不以Hook注册日志替代实际入口断言。


[dev61真实服务失败摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-recording-dev61-failure-20261101.json`）记录602包文件与184ebf1/wheel/site相同、六项严格生成通过、start/stop200及全部8rank after-all_reduce；rank0最终time NameError导致dump失败/连接断开，进程与端口已清理。此结果保持失败，不因生成或归约成功冒称录制集成通过。修正后的Source完整file branch与free-global检查通过；隔离dev62的5项Graph protocol CPU通过。当前603包文件与源码/site一致，wheel SHA256 `ea5404376812f4e2c93a43222067123ad4ce2075116850f0aa00404774cf86d7`，最后门禁与同profile真实服务重验继续待完成。


隔离dev62最终结果：5项Graph CPU protocol与7项recorder CPU（含public dump(file)、free globals）通过；445/445 Hook、3个resolution seam、0 errors及85项仓库测试通过，603包文件同源码/site。L0/Skill/guide与实际宽窄屏检索/阅读/菜单验收通过。新的layout控制无伪造GPU/RDMA结果，仍须同profile实际八卡Graph与start/stop/dump重跑确认；全DAS、数据集精度、实际dispatch负载/性能范围继续未完成。

## 75. 2026-11-01 dev62 录制服务结果与固定数学子集方法

dev62于10:39:39–10:45:24完成六项严格生成及本人loopback start/stop/dump，三端点均200，自有进程清空、端口关闭。603包文件与56043f0/wheel/site逐字节相同，全部8rank各43层solver计数[8]；[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-load-aware-recording-dev62-validation-20261101.json`）固定版本与dump。rank0文件为int64[43,256]、非负、43层全有计数，总和4548352，SHA256为9aaf3d3b872b1471a2a40ea57dcbf8c6b72a34bcbd12448d037bbd8e35d22cd4。它是dispatch计算工作，可能含expert对齐/padding，不当成唯一语义token或性能改善。旧dev61失败记录保留。

新增一次请求、保存逐题原始数据的runner与离线scorer；模型身份/正常stop在数学提取前检查，错误和截断留在分母。冻结官方GSM8K commit与SHA256，按当前SGLang协议使用前5题prefix及后128题评分且互斥，共享库不复制整份数据。目前仅完成方法/协议测试与数据准备，真实128题和同配置DAS基线待执行，不预先声明精度通过。操作与范围见[验证指南](validation.md#固定-gsm8k-子集与原始响应验收)。

新增工具的6项契约测试与本地/容器完整91项仓库测试通过：分离数学错误与传输无效、拒绝截断/错误模型、prefix隔离与索引检查、缺失样本留在分母，以及真实loopback测试HTTP服务验证逐题原始数据和500错误不重试。L0/Skill/guide与实际宽窄屏验收通过。此次只改方法/工具/证据，hcu_sglang运行时与冻结dev62的603文件不变；数据集采集仍待执行，不将工具测试或旧六题作为128题结果。

## 76. 2026-11-01 固定128题测量与匹配对照结果

冻结dev62的load-aware/redundant8/stat功能变体完成128题：127正确、0无效、准确率0.9921875；独立离线复算一致，128份request/response/transport均齐全。[证据摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-gsm8k128-feature-dev62-validation-20261101.json`）固定603包文件、56043f0/runtime与f773a25/method、数据SHA、ordered raw hashes和清理。唯一oracle不一致sample119原样保留：expected95200、predicted99077，正常stop且有效回答；不重试、改标签或从分母剔除。该子集未建立通用准确率或DAS回归结论。

基础配方匹配对照已完成：DAS 127/128（99.21875%），plugin 126/128（98.4375%），两端均0无效响应；相差1题、0.78125个百分点。126题双方正确，sample12仅DAS正确，sample119双方错误；原始答案、模型回答及一次请求记录保留，不重试或改标签。用户明确“无需完全对齐，128题大致一样即可”，本次结果按此接受，不据此推导未来统一阈值。见[成对证据](hcu-sglang-development.md#历史-json-快照)（`hcu-gsm8k128-paired-controls-20261101.json`）。

二者使用同一基础8卡配方、canonical decode graph flag、优化环境、权重路径、数据、prefix、采样和并发；原始DAS只读且跳过general Hook，两端六项短回答检查通过、128份request/response/transport齐全、进程与端口完成清理。比较范围是声明参数与环境；另存的正常启动配置重建并非服务当时的live server_info，schema差异保留，不宣称全部有效配置相同。功能变体127/128单独报告，不混入基础匹配对照。

新增compare工具独立复算并拒绝配置不匹配，4项比较契约覆盖相同总分但不同错题、功能标志与优化环境差异及允许的输出路径差异。精度子集按用户标准接受；全DAS迁移和性能验证继续按各自范围推进。

## 77. 2026-11-01 dLLM 独立上下文编码的 Graph 批大小限制

固定DAS的base_cuda_graph_runner仅在get_batch_sizes_to_capture新增DllmConfig请求上限；现迁入完整Source函数至graph_batch，沿聚合register注册HCU AROUND，非HCU返回当前有效original。配置由原生DllmConfig.from_server_args解析，只有requires_separate_context_encoding时取pool size与配置max_running_requests的较小值；保留原先ceil alignment、captured_req_width、TBO和compile列表逻辑，不能改成简单截断既有结果。来源许可、函数注释及docstring保留，不新增说明性runtime注释。

新增真实原生配置解析与预导入decode/runner公共别名测试：Gemma4Renoise单独上下文路径、默认1请求、pool上限、alignment/width/TBO、普通decode与非separate dLLM不变、非HCU保留已有wrapper。模型配置读取用fixture隔离权重下载，实际算法和DllmConfig保留；这是CPU控制流验证，不能宣称Gemma模型GPU Graph或生成精度通过。隔离dev63产物与正式Hook/仓库门禁按实际结果补记。

隔离dev63门禁完成：604包文件与源码/site逐字节相同，wheel SHA256 b28a3dd1befcfb80e715defa8bfae0875c0907032164b44c2cb21293e0ba8f4a。6项批大小CPU测试、5项既有DeepEP Graph协议测试、446/446 Hook与3个resolution seam均通过且0 errors；本地与容器95项仓库测试、L0、Skill元信息、guide快照和实际宽窄屏交互通过。当前按原八卡基础配方启动独立服务验证普通模型回归，不重复追求已接受的128题完全对齐；服务结果另记。未验证的实际扩散模型路径保持未验证，全DAS迁移未完成。

[dev63八卡回归摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-graph-batch-dev63-validation-20261101.json`）：11:38:15–11:43:57按原基础配方完成六项严格生成，全部HTTP200、完整最终答案正确、正常stop，自有进程为空且端口关闭；604包文件与84f961e/wheel/site相同。DSPARK/DeepEP/Graph及原优化环境保持。该模型为普通解码，验证新函数在此路径没有阻断实际Graph捕获和服务；不能据此宣称Gemma4Renoise GPU分支、吞吐或全DAS迁移完成。已接受的dev62 GSM8K128对照不归用为dev63重跑结果。

## 78. 2026-11-01 Graph 初始化的 PP 预算、混合层与 H16 缓存释放

迁入固定Source cuda_graph_setup全部6个变化/新增函数：混合attention/MoE局部覆盖检查、保留尾部纯MoE的全局索引、H16捕获后同步释放、PP内存预算、完整capture_cuda_graphs与capture_prefill_graph调用方。复用native记录类型和不变helper；预算initialized状态仍写原生模块，不产生插件副本。现有4个入口为HCU AROUND，非HCU保留当前有效original；新增2个helper由owned调用方使用。Source完整函数AST在native globals限定、模块状态写入及相同HCU检测映射后一致，原注释/docstring保留，不新增说明性runtime注释。

待验：真实调用方/预导入别名、尾部纯MoE、洞/重复层、PP组与native预算状态、capture/finalize/cache释放顺序、非HCU回退。CPU fixture控制流与实际collective/GPU/模型服务分别记录，不把模拟内存查询或runner构造当作模型Graph通过。

Source控制流9项通过，包括完整public prefill/capture入口。真实4-rank Gloo在PP2时rank0/1执行本组sum=3，rank2/3执行末级TP内存MIN并由native setter得到2684354560字节；切换PP1后全rank得到3489660928字节，[证据](hcu-sglang-development.md#历史-json-快照)（`hcu-graph-setup-collective-validation-20261101.json`）固定原始输出哈希。CUDA显存查询是fixture，归约与process group真实，不冒称GPU collective或实际PP模型。隔离dev64的1项真实GPU测试通过：实际CUDAGraph捕获后调用native platform synchronize/empty_cache，两次变输入replay精确；这是allocator/小型Graph安全性，不是H16 kernel或模型验证。

[隔离dev64门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-graph-setup-gates-validation-20261101.json`）完成：605包文件与源码/site相同，wheel SHA256 b3d040c862457b000406b574096bc2cf52ac6980045343f5c0fb7dfc9091cd58；450/450 Hook、3个resolution seam、0 errors，9项新控制流、6项既有批大小与本地/容器95项仓库测试通过。L0、Skill元信息、guide快照及实际宽窄屏检索/阅读/菜单验收通过。实际八卡原配方服务仍按独立run继续，未将其提前标为通过；PP/hybrid/H16模型及全DAS迁移范围保持未完成。

[dev64八卡服务摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-graph-setup-dev64-validation-20261101.json`）：11:52:23–11:58:00按原DSPARK/DeepEP/Graph基础配方完成六项严格生成，全部HTTP200、完整答案正确且正常stop，自有进程为空、端口关闭。605包文件与4873fea/wheel/site逐字节相同；实际预算日志24条，owned HCU backend正常加载。该PP1普通模型证明其预算与捕获流程可完成，不外推PP/hybrid/H16模型；也未重复计为128题或性能测量。全DAS迁移保持进行中。

## 79. 2026-11-01 预填充 runner 与 replay 记录契约

迁入固定Source预填充runner全部12个变化/新增方法，owned类继承native以保留类型关系；完整Source构造器显式从native父类的base初始化，不重复调用旧prefill构造器。runner-owned ContextVar scope完整迁入，capture与replay上下文成对接入；原生LogitsProcessorOutput的dataclass子类追加Source三项draft字段，保留官方已有字段与复制生命周期。ForwardBatch沿已有owned schema使用encoder replay元数据。旧官方AttentionBackend缺少的可选prefill_cuda_graph_default_max_seq_len通过getattr(...,None)读取，对应Source基类声明默认值，不写冻结配置或动态补全seed。

变化包括后端默认max_seq_len与本地/DP共用准入、DSA的BCG前缀例外、预热AITER先于backend构造、capture前model prepare、load_batch保留encoder replay字段、V4.1 replay input IDs、MegaMoE valid-token刷新、trim保留draft top-k，以及Full replay等待KV层加载。保留来源注释/许可/docstring。构造、元数据、公开execute、scope生命周期与实际GPU/模型验证按本次证据补记，不以类替换日志代替行为验证。

配对迁入Source完整调度_local_prefill_cuda_graph_vote，使backend vote与max_seq_len在schedule与forward两端读取一致；MegaMoE仅迁入runtime selector和graph token setter，沿native对称buffer registry更新，未迁入其余standalone backend/kernel。Logits、MegaMoE及DP文件仍标in_progress。13项CPU测试通过，构造夹具首轮漏attn_tp_sequence_sharded修正后完整构造完成；GPU夹具首轮漏prefill_backend_name，补齐后隔离dev65的真实primitive Graph经公开load/execute/body/trim两次变输入通过，encoder metadata、prepared IDs、hidden states与draft top-k均验证。两次夹具失败原始日志保留，不是生产补丁或虚假模型结果。实际八卡显式BCG prefill功能变体继续执行，保留原decode/DSPARK/DeepEP优化，不提前声明服务或模型replay hit通过。

[隔离dev65门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-runner-gates-validation-20261101.json`）完成：610包文件与源码/site一致，wheel SHA256 469b7a6fee61cada806a89ac9d7167e79be67aab31f91a19c80d6a5b22c33079；453/453 Hook、3个resolution seam、0 errors。隔离13项prefill CPU、9项既有Graph setup、1项真实GPU公开execute与本地/容器95项仓库测试通过，L0/Skill/guide及实际宽窄屏交互通过。原始8卡基础配方仅新增显式breakable prefill、64/128 token buckets、固定context/sequence上限256的功能变体正在执行；它不是“原配方未变”的基线，也不能将六题服务结果推为128题重跑或DP模型Graph hit。

## 80. 2026-11-01 真实 BCG 捕获通过后的 MLP 同步投票缺口

[dev65失败证据](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-dev65-failure-20261101.json`）固定b017aeb的610包文件与wheel/site一致：八个rank均完成BCG prefill64/128捕获，但scheduler启动时旧prepare_mlp_sync_batch_raw未传Source vote新必填num_tokens，TypeError导致ready前退出，自有进程和端口已清理。这是迁移调用链缺口，旧13项CPU直调vote未覆盖生产入口；不能以捕获完成或primitive GPU测试称服务成功。

修正迁入完整Source MLP producer、同步记录的12列协议/归约、batch metadata更新与elastic拓扑校验；配套完整CPU-only CP token geometry、mixed-replica归一化和encoder replay计数helper。ScheduleBatch追加声明的global_cp_num_tokens以保留dataclass复制；MLPSyncBatchInfo继承native并迁入全部方法。HCU CP-v2环境字段保留Source默认False，不写冻结seed；缺少可选backend folds属性按Source基类默认False读取。真实公共prepare入口新增修复前反例，Source同步组/CP行宽与消费路径按CPU/真实Gloo/模型分别验收。此阶段不宣称剩余CP模型/worker carrier与整个DP文件完成。

Source的8个完整同步函数/方法与7个完整计数/拓扑helper经native globals、owned import及相同默认属性读取规范化AST一致。新增6项CPU通过，保留旧公共producer缺参反例，新公共producer将3+128=131折叠行传给vote与batch metadata；记录12列、ScheduleBatch复制/pickle通过。[真实四进程Gloo](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-sync-collective-validation-20261101.json`）覆盖CPU gather+broadcast、world all-reduce、DP2/CP2混合布局归一化及子组[1,2,3]的global root1。dev66的612包文件与源码/site相同；同一八卡BCG配置正重跑，失败dev65证据不改写，GPU primitive旧dev65结果保持其artifact归属。

[隔离dev66门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-sync-gates-validation-20261101.json`）：612包文件与源码/site相同，wheel SHA256 0f720444c16cd2ab081c8eee768c29ba950c99dc5d118893fc3d25cbdbc702e2；458/458 Hook、3个resolution seam、0 errors，6项同步CPU、13项prefill CPU及本地/容器95项仓库测试通过。Source Gloo证据字节与本产物相同，旧dev65 primitive GPU结果没有改记为dev66重跑。L0/Skill/guide与实际宽窄屏检索/阅读/菜单通过；相同BCG feature服务结果仍待完成。

[dev66同配置八卡BCG服务摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-prefill-bcg-dev66-validation-20261101.json`）：12:35:06–12:40:58完整通过，全部8rank完成目标prefill breakable64/128捕获，随后decode/verify捕获与六项严格生成通过，HTTP200、完整最终答案和正常stop均满足；自有进程清空、端口关闭。612包文件与c61fb5c/wheel/site相同。与dev65失败相比仅运行时修正及输出标识/端口变化，配方和BCG feature参数一致；没有关闭prefill、DSPARK、DeepEP或原优化来回避问题。该证据证明实际模型capture初始化与服务回归，不证明请求实际命中BCG replay、DP混合请求、CP/elastic模型、128题重跑或性能。旧dev65失败保留，全DAS迁移未完成。

## 81. 2026-11-01 Encoder SWA replay 完整折叠与 worker 生命周期

迁入Source完整encoder_swa_replay模块（FoldedExtend、批量legacy replay、fold/drop、window reset、Engram history与执行计数），并迁入完整TpModelWorker.forward_batch_generation方法，将新return_hidden_states_before_norm参数、折叠后FB生产、late-tail行号回映射及返回原调度KV plan成对接通。复用native未变GenerationBatchResult与其余worker方法；HCU AROUND，非HCU保留effective original。Engram history_row完整Source方法以显式receiver helper复用，避免给native类注入未声明方法；其余Engram变化仍待迁移。原计数模块导入唯一owned replay_len/folded_replay_rows，避免重复owner。保留来源注释/docstring/版权许可；旧backend缺可选folds属性按Source基类默认False。

本阶段验收从公开worker入口检查折叠行生产、隐藏态/drop消费、return-kv plan身份和legacy先重播再extend顺序。ForwardBatch初始化/模型计算的昂贵fixture与真实张量/GPU/模型分开记录，不把源AST或Hook计数解释为cache-hit模型精度。

8项CPU通过：137行fold回5个scheduled rows、batch未变、2个legacy replay合成1次forward、dense/late-tail行回映射、clamped Engram history、公开worker fold与非fold两种分发，以及原调度plan身份。隔离dev67的真实GPU张量与primitive CUDAGraph在变输入后dense/late-tail映射精确。源码14个encoder符号、完整worker方法与Engram history_row经native globals/owned import/receiver/docstring缩进规范化一致。调度PrefillAdder预算和模型feature validator仍有Source差异，不将本阶段声明为整个encoder feature链已完成。

预检发现旧snapshot_control_resolution脚本只创建seed/读resolved_view，未调用resolve_once；已修正历史GSM8K共享分析范围，保留原始种子差异及分数，不把它解释为effective/live配置。新的显式resolve_once对当前0731 Flash配置拒绝encoder replay：requires DeepSeek-V4.1。model-query另找到匹配model_type=deepseek_v41的V4.1-Flash，配置与容器可见性已读验；完整模型feature仍待其合法参数与预算迁移验证。当前八卡BCG回归使用原0731模型且未启用encoder flag，不冒称encoder cache-hit服务验证。

[dev67分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-encoder-swa-gates-validation-20261101.json`）：614包文件与源码/site一致，wheel SHA256 b741efaa6649b133f4961d6147a47c206735bf8a222da56914044eead37b268e；460/460 Hook、3个resolution seam、0 errors，8项encoder CPU、6项同步CPU、1项真实GPU映射Graph及本地/容器95项仓库测试通过，L0/Skill/guide与实际宽窄屏验收通过。实际resolve_once对V4.1-Flash进一步拒绝当前硬件（CUDA/gfx950限制），与Source保留条件相同；没有绕过硬件/DP约束。当前HCU的完整encoder feature服务仍未验证，普通BCG服务回归另记，不将算法GPU primitive通过外推为该模型能力。

[dev67普通八卡BCG回归](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-prefill-bcg-dev67-validation-20261101.json`）：13:01:45–13:07:31通过，8rank完成prefill捕获、六项严格生成全部HTTP200/正确完整答案/正常stop，自有进程为空、端口关闭。614包文件与cb69307/wheel/site相同。此run明确encoder flag为False，只验证新worker普通生成路径与既有BCG回归，不称encoder cache-hit或兼容硬件模型验证；V4.1预检限制和下一阶段PrefillAdder预算/feature validator差异保留。旧128题分数与其artifact不归用到dev67。

## 82. 2026-11-01 Prefill replay 预算与 feature 校验

迁入Source全部11个变化/新增PrefillAdder方法及冻结admission记录，复用native构造与不变helper；compute预算按page-ceiled或exact原始extend加replay行，KV预算只收page-ceiled extend及headroom。首次命中chunk先扣replay，idle容量失败才defer staged，retraction完成storage admission accounting。配套三个memory budget的完整Source check_prefill接口与Req辅助SWA属性/needs-load-back读出；固定native staged plan仅有SWA，无AUX窗口，明确按0 AUX解释，保留原plan构造ABI，窗口plan生产/传输继续pending。

Source完整features校验和最小128+page预算迁入，保留CUDA/gfx950/HIP/DP等限制及decoder auto声明。源码证明handle_model_capability_adjustments除末尾预算校验外与native AST相同，沿现有具名seam先调用previous再追加Source尾部，并保留INSTANCE早退；不复制旧分支覆盖其他合法适配，不修改冻结seed。实际CPU预算、公开admission与resolve_once配置验证按本轮证据记录，未将gfx938模型encoder feature限制绕过。

隔离dev68的8项预算与5项校验CPU测试通过：旧公开adder将150预算扣成-10、提交25行extend的反例保留；新实现只提交16行extend并收128 replay，chunk余6，KV仅32。143预算不提交，144预算提交首个16-token页并耗尽chunk；exact与DLLM compute分别扣原始/ceiled extend加replay。三种native预算新接口、idle deferral、retraction storage完成及legacy/AUX读出通过。实际具名resolution dispatcher确认previous先执行、重复注册不叠加以及128+page拒绝逻辑；未mock掉预算reader。18个完整Source方法/函数及admission记录AST规范化一致，native capability body除Source尾部外一致。

[dev68分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-admission-gates-validation-20261101.json`）：619包文件与源码/site一致，wheel SHA256 458c34789eab6969fd1d6b91969dc50dd5218670e6c7d387523438afc548c4c9；466/466 Hook、4个具名resolution seam、0 errors；8项预算、5项校验和本地/容器95项仓库测试通过，L0/Skill/guide与实际宽窄屏通过。实际V4.1 resolve_once在encoder-off下成功，decoder auto=False、per-rank chunk4096、attention DP8/replica DP1；encoder-on仍按Source硬件约束拒绝。另捕获未迁移的V4.1 defaults覆盖显式sparse-prefill问题，不将该预检声称为优化配置交付。普通8GPU BCG回归结果另记；当前改动为CPU调度/接口/配置，未冒称新的GPU kernel数值验收。

[dev68普通八卡BCG回归](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-prefill-bcg-dev68-validation-20261101.json`）：13:46:10–13:52:01通过，全部8rank完成prefill捕获，六项严格生成HTTP200/完整答案正确/正常stop，自有进程为空且端口关闭；619包文件与dd88288/wheel/site相同。encoder flag保持False；该结果验证本次Req/adder/memory-budget接口安装后的普通服务，不证明encoder cache-hit、AUX窗口传输、请求级BCG命中或128题重跑。下一阶段继续迁移V4 defaults/CP和窗口plan生产/传输，全部DAS目标未完成。

## 83. 2026-11-01 V4 attention 默认值与 HCU CP 策略

迁入默认attention后端选择与apply_deepseek_v4_defaults完整Source实现，统一既有Source CP校验所有权；与已有features/budget校验合并覆盖本文件全部变化。显式开启的HCU sparse prefill保留，未显式开启仍按Source HIP默认关闭；auto后端与max-running经声明发布，mega/TRT约束不放宽。CP使用Source HCU规则与统一拓扑声明，保留非HCU/zigzag/TP范围及A2A约束。原有model-specific兼容层保留其余tilelang/FP8开关职责，未以改种子或关闭优化代替迁移。GPU内核未改，Source策略CPU、实际resolve_once环境值与普通八卡回归按本轮证据分开记录。

收敛注册审查：CP校验此前已由current_v4/hcu_backend/arg_groups迁入并注册REPLACE。本轮dev69原型新增AROUND与其同语义叠加，正式报告467个唯一目标揭示合同列表有重复；不将重复计为新能力。最终dev70保留原REPLACE入口，legacy模块只重导出统一v4_defaults函数，去除快照globals副本与新增AROUND，并去重合同。Source策略算法不变，新增别名/单一注册断言；dev69隔离数学run仍使用其冻结原型，结果与最终dev70产物分开记录。

实际解析对照：dev68的0731输入sparse env=1、resolve_once后为False；dev69对0731及V4.1输入/输出均为1。因此旧服务配置中“原优化env保留”只代表声明参数，不能证明此开关有效开启；旧原始分数/日志保持原样。新稀疏路径在原始plain8卡配方上重新固定128题，旧DAS对照与dataset不变且不选择性重试。dev69原型无对应runtime提交，使用不可变build snapshot/wheel/site证据；最终dev70集中CP所有权，实际CLI/门禁/服务分别记录。

[固定128题新稀疏路径测量](hcu-sglang-development.md#历史-json-快照)（`hcu-gsm8k128-sparse-default-dev69-validation-20261101.json`）：冻结dev69原型127/128、0 invalid，384份逐题request/response/transport齐全；独立scorer复算与保存的plain DAS控制组逐题结果相同（127双方正确、1双方错误），没有选择性重试或改标签。比较范围仍为声明参数与env，不声称完整effective配置或权重哈希一致。原型620文件与build snapshot/wheel/site相同，但runtime无对应Git提交，保留其准确归属；最终dev70策略函数AST相同，差异为CP注册收敛，未冒称dev70重跑128题。

[最终dev70门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-v4-defaults-gates-validation-20261101.json`）：620包文件与源码/site相同，wheel SHA256 be336433b63eee297ee126cfcf111d758fbb2076f4301d22bf3ad74b6bc16d33；467/467唯一Hook、4个seam、0 errors。9项默认/CP/alias测试（纠正REPLACE函数由框架包装，不能断言公开wrapper identity）、5项既有feature校验和95项仓库测试通过；0731与V4.1真实resolve_once前/后sparse均1。L0/Skill/guide与实际宽窄屏交互通过。CP/SM100/mega实际模型仍未验证，最终dev70八卡普通回归继续，窗口plan生产/传输等全量DAS范围未关闭。

[最终dev70普通八卡服务](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-sparse-default-dev70-validation-20261101.json`）：14:15:35–14:21:18通过，六项严格生成全部HTTP200/完整答案正确/正常stop，自有进程为空、端口关闭。620包文件与32fb304/wheel/site相同，实际server日志确认保留显式sparse开关；原plain配方的DSPARK/DeepEP/decodeGraph/并行与权重加载保留，未加CP/BCG变体参数。此结果属于最终dev70普通集成，128题127/128仍准确归属于隔离dev69原型；不是CP/SM100/mega模型或吞吐验收。全DAS窗口plan生产/传输及其他pending模块继续未完成。

## 84. 2026-11-01 双窗口计划与树修复后端

作为窗口pipeline迁移前置，迁入完整WindowLoadBackPlan/StagedPrefetchPlan owned记录，暂不替换固定native staged plan的swa_tokens构造ABI；已有Req AUX reader可消费owned windows。迁入Python TreeCore全部3个窗口查询/attach/segment变化方法，按component_type分别识别SWA/AUX tombstone、拆分真实树节点、设置独立数据与LRU/ledger。Rust两项窗口方法按Source仅允许SWA，AUX明确拒绝；沿现有get_tree_core_factory扩展点保留原factory，在实际选择并创建native Rust实例后绑定新方法，不在插件加载时强制编译可选Rust。自定义工厂/子类保留其原行为，非HCU effective original保留。

此阶段不改变pipeline producer与H2D发布，避免新plan先替换旧构造字段而破坏现有caller。测试保留真实TreeCore构造、native组件分片、节点arena与LRU，不mock _split_node或ledger setter；storage/H2D/完整窗口cache模型范围继续pending。

8项CPU通过，保留旧API不接受component_type的反例；使用真实TreeCore构造/native组件split/LRU与账本，验证AUX[2,6)拆成2/4/2三个节点、FULL/SWA字节不变、AUX缺失[0,2)/[6,8)、protected/evictable归属以及pending backup split action。两轮夹具分别补齐只读allocator内部字段与cache session开关，未修改生产算法。隔离dev71的1项真实GPU确认clone设备/存储分离及独立SWA数据；实际native Rust CPU经工厂选中、insert/局部attach/补齐/SWA ledger8通过，AUX明确拒绝。Rust采用原始受锁定Cargo/native来源，输入sglang Git工作树检查仍干净。stage/producer/H2D端尚未迁入，因此owned记录暂未替换native schema，也不声明完整窗口缓存模型通过。

[dev71分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-window-tree-gates-validation-20261101.json`）：623包文件与源码/site一致，wheel SHA256 28b13d4a7d31e45f6b0ac8418cb81fe2f8439e40751ba3f844cc3654e7779344；471/471 Hook、4个seam、0 errors。8项窗口CPU、8项既有预算、本地/容器95项仓库测试通过；1项真实GPU独立数据/clone测试及实际Rust CPU工厂+binding测试通过。5个窗口方法与2个完整Source记录AST规范化一致，Rust加载延迟仅为适配方式变化，不改其SWA算法或AUX guard。L0/Skill/guide与实际宽窄屏检索/阅读/菜单通过。普通八卡回归继续，不将无hicache参数的服务用于证明storage/H2D或窗口cache-hit。

[dev71普通八卡回归](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-sparse-default-dev71-validation-20261101.json`）：14:53:32–14:59:14通过，六项严格生成全部HTTP200/完整答案正确/正常stop，自有进程为空且端口关闭；623包文件与2cb0f1b/wheel/site相同，显式sparse保留。该run未加入hicache/buffer/storage参数，只是新树/factory接口安装后的普通集成，不证明window plan生产、H2D发布或缓存命中；这些下一阶段范围仍pending。原型dev69的128题保持其独立产物归属，不挪作dev71测量；全部DAS目标仍未完成。

## 85. 2026-11-01 多窗口 staged plan 与 H2D 发布

迁入Source BufferModePipeline全部16个变化/新增方法、完整buffer-only双窗口host fence，以及TreeComponent工厂与UnifiedRadixCache.init_hicache生产caller，使用owned windows记录替换该生产/消费闭环中的旧swa_tokens语义。Source未变allocation/queue/ACK核心保留native；缺失区间按组件切host bounce，仅为缺失行排队H2D，resident FULL范围修复树，tail随普通insert发布；完整原始bounce保留到ack。Tail-only backup沿Source False默认，keys和device rows一起裁剪并推进backend prefix context。

固定native SWA组件通过只读view补齐pool_name与完整Source restore mapping helper；已有custom字段/callback原样保留。新Source控制器prefix helper在旧基类缺方法时以显式receiver调用，有现有callback则保留其行为。Root cache retention/readonly-match、其他SWA组件与L3/linker变化仍按独立file scope待迁移，不将generic pipeline迁入解释为所有cache模型完成。测试按真实树修复/接收者与显式传输fixture分层记录。

配套新增PoolName.AUXILIARY_SWA完整Source枚举；捕获的legacy成员保持str/hash键相等，通过copyreg转换后pickle回owned成员，未修改内部Enum表。9项CPU通过：prepare生成SWA[(2,4),(6,8)]、AUX[(4,6),(6,8)]，各仅4行H2D，原三份8行bounce到ACK才释放；分配失败context回滚并保留staging，发布错误保留pending状态并fail-stop，covered窗口移除其派生sidecar。实际TreeCore repair/match保留，cache publication/allocator facade显式作为fixture。public init_hicache与base-component factory、tail keys/rows/prefix、双窗口共享arena fence通过。修正夹具固定prefix长度与原生日志/配置字段后无生产算法改动。隔离dev72 GPU fixture controller经真实pinned host→CUDA stream/event复制，逐池字节oracle及ACK释放顺序通过；不冒称native HiCacheController线程、L3或完整模型cache-hit。

[dev72分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-window-pipeline-gates-validation-20261101.json`）：627包文件与源码/site相同，wheel SHA256 348a661f96d463f063aa3a4eaa41479fbe8da872f487dde9923b24408fcbfe63；476/476 Hook、4个seam、0 errors，9项pipeline CPU、8项树CPU、本地/容器95项仓库测试通过。1项GPU pinned host/stream/event H2D逐池字节与ACK顺序通过，但physical queue/allocator/cache insert facade明确为fixture，不称native controller线程或L3模型。21个完整Source方法/函数及完整PoolName枚举AST规范化一致，legacy字符串键/pickle兼容通过。L0/Skill/guide及实际宽窄屏通过；普通八卡回归继续，其他cache retention/linker/FileIO与全量DAS目标未完成。

[dev72普通八卡集成](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-sparse-default-dev72-validation-20261101.json`）：15:32:59–15:38:41通过，六项严格请求HTTP200/完整答案正确/正常stop，显式sparse保留，自有进程为空、端口关闭；627包文件与8e01b89/wheel/site相同。该run未加hicache/buffer/storage参数，只验证新枚举和pipeline接口安装后的普通集成，不是L3/cache-hit模型验证；GPU H2D使用明确的fixture controller范围，二者不合并成native controller/storage通过。后续retention、readonly匹配、cache/linker/FileIO等Source差异及完整缓存模型继续pending，全量DAS未完成。

## 86. 2026-11-01 FileIO 的 TP 状态与共享 MLA 文件归属

复现旧版两个反例：未限容量时不同 TP rank 的同一 Mamba key 读回首个 rank 的数据；开启容量限制后 MLA 非零 rank 拒绝自己的状态写入。按固定 Source 迁入完整 LRUFileEvictor 类及 HiCacheFile 的 constructor、suffix、metadata warmup 三个完整方法，Mamba 用 TP suffix，MLA KV 继续共享；每个 rank 管自己的 TP 文件，rank 0 额外管理共享文件。公共 lazy factory 解析 owned HiCacheFile 子类，15个未变化 native 方法逐一 AST 比对后继承，读写/临时文件原子替换/commit/abort/batch-v2 保留原生数据路径。

[Source 对齐记录](hcu-sglang-development.md#历史-json-快照)（`hcu-file-backend-source-alignment.json`） 检查完整类和三个方法，不靠局部语句匹配。`tests/runtime/file_backend_cpu.py` 从原生 StorageBackendFactory 构造，真实临时文件和 CPU tensor 读写，覆盖两个反例、其他 rank 文件不被淘汰、metadata 回调/重启扫描、rank 0 双 suffix 启动淘汰、PP/CP、pending 写入保护、失败回滚，以及公共 batch-v2 状态读写与 prefix 命中。batch 的 host pool 为数据夹具，backend 和磁盘 IO 均为真实实现；不将该项解释为控制器线程或完整模型 L3 命中。

精度接受条件按用户补充：128题整体与基线大致一致即可，允许少量差异；无效输出/运行错误仍需修复。原型dev69的127/128和DAS127/128保留原有产物归属，不挪作dev73的测量；本轮 FileIO 不改变模型数学路径，也不重复跑无存储参数的普通服务来证明缓存行为。其余 retention/readonly-match/controller/linker 和全量 DAS 目标继续执行。

[dev73分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-file-backend-gates-validation-20261101.json`）：wheel SHA256 `33dddedb42fb0751d1e66e4c4890ef09b4a61f3a810bebe85ff6d1260ac47970`，629包文件与本轮源码及隔离site逐字节一致。10项FileIO测试、478/478 Hook、4个seam、0 errors，本地/容器95项仓库测试通过；旧实现两个反例均按预期失败，固定来源SHA/clean和文件哈希复核通过。台账将完整FileIO变化按逐文件实现记录，保留未验证的native controller线程、分布式rank进程协调和完整模型L3命中范围；不能从该记录推断全量DAS迁移完成。

## 87. 2026-11-01 Python KV 事件的 LoRA namespace

继续梳理 retention 参数时发现 CacheInitParams 新默认表依赖 KvEventLoraNames，先闭合其 Python 生产/消费。完整迁入 Source events 模块的 LoRANameTable、KvEventLoraNames、KVCacheEventRecorder 三个类；默认事件对已登记的最终 namespace 附加 hash seed，store/remove/parent 链使用同一命名，queue 合并要求 lora_name 相等。保留 publisher 生命周期内的名称，允许 GPU removal 和请求释放后发布 host store；Dynamo 格式仍走原有 adapter ID fallback 与 namespace seed。

[Source 对齐](hcu-sglang-development.md#历史-json-快照)（`hcu-kv-events-source-alignment.json`） 检查完整模块前缀字节、三类及两份完整 scheduler 方法，六个导入 hash helper 与 fixed native AST 相同。原生 public generate/embedding handler 和 elastic namespace 方法保留 original；ContextVar 在外层保存可选 IPC lora_name，namespace seam 在 original 完成后调用完整 Source 登记 helper，finally 恢复上下文。旧 IPC 无名称字段时用 Source 已登记 adapter ID 回退；不改写消息 dataclass。这里仅完成 scheduler 的 KV 名称片段，其余变化继续 in_progress。

测试保留两个 native 反例：默认不同 namespace 的 LoRA ID 冲突、不同名称事件被合并。直接构造和 take 真实 Python recorder，保留普通/Dynamo 兼容，store/remove/parent、请求消失后 host store、disabled 和 context 异常/嵌套。public handler 使用原生 ExitStack 入口与真实注册 namespace seam，昂贵内部 admission 和 elastic namespace 场景显式用数据夹具；不是完整 LoRA 模型部署。原生 UnifiedTreeCore constructor/insert/leaf eviction 的实际事件出口单独验证。修正夹具的 keyword-only 参数、session 配置及 FULL 应走 leaf eviction 的调用后，未改生产算法来迎合断言。Rust tagged binding、发布 socket/router、CacheInitParams/retention 与其余全量 DAS 目标保持待做。

[dev74分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-kv-events-gates-validation-20261101.json`）：12项CPU测试、484/484 Hook、4个seam、0 errors、本地/容器95项仓库测试通过。隔离wheel SHA256 `317d4ba6d6982be22b7b75da952bef5b7ab74c2307e2821585e9a1d4c965de6f`，631包文件与本轮源码及site相同；原生Python树实际insert/leaf eviction产生的store/remove hashes一致。初轮夹具错误日志保留，最终门禁引用修正后的attempt2。后续普通八卡只用于原生scheduler请求路径集成，不替代LoRA事件、Rust或L3模型验证。128题接受“大致一致”的标准保持，本轮不将旧产物分数挪给新wheel。

[dev74普通八卡回归](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-sparse-default-dev74-validation-20261101.json`）：16:09:20–16:14:58通过，六项严格请求全部HTTP200/完整答案正确/正常stop，显式sparse保留，自有进程为空、端口关闭；631包文件与14abc7e/wheel/site相同。本轮运行实际经过原生scheduler新增包装后的请求入口，未启用LoRA/KV事件/hicache/storage参数；不将普通生成称为事件路由、Rust binding或完整L3/cache-hit验证。全量DAS、retention及其依赖继续pending，128题历史结果保持各自产物归属。

## 88. 2026-11-01 Owned Rust binding 与宽 token hash

延续 Python KV 事件迁移，完整打包固定 Source 的 Rust crate（22文件，含 Cargo.lock 和测试），按完整 Source converter/recorder 消费新增带 extra_key 的 block_removed tuple。沿原生 public Rust loader 指定 owned workspace 和独立 production/inspection 模块名，保留 mode/features/cache 及显式 custom provider；实际选中原生工厂后校验 binding 身份并接入 Source 事件 recorder，旧 binding 不能与新 ABI 混用。Native capability/custom fallback 保留，只有源码/工具链可用性才允许由插件 workspace 满足；部署不依赖输入 DAS 路径或修改官方树。

Source Rust 同时扩大非负 token 到 int64 域，因此成对迁入完整 Source C++ hash 和 Python 输入转换；普通 u32 页面编码保持原样，宽页面使用0xff tag及little-endian u64，前缀hash与bigram按同一规则连接。C++扩展独立命名，native get_native_hash caller/alias 经正式 Hook 接入。Torch兼容header与rpath使用插件资产，额外编译flags保留，加载可选Rust时才构建；Cargo继续使用--locked。详见 [Source 对齐](hcu-sglang-development.md#历史-json-快照)（`hcu-rust-events-source-alignment.json`）。

dev75为冻结原型：659包文件，真实Rust store/remove与Python recorder及宽hash对照已执行，但未完成owned workspace的backend预检；原型测试还需消费eviction返回的CPU索引。dev76补齐预检后独立冻结，旧原型不挪为最终wheel证明。初轮函数对象identity断言不适用于native REPLACE包装，已改为实际caller alias及数值/事件行为；传输AppleDouble元数据单独清除，未用其解析错误改生产逻辑。CPU夹具会断言并消费返回的FULL索引，原有Python树测试也补齐同一ownership要求。

本轮验证范围是Rust CPU binding、事件/hash与源码/包装契约；没有运行新的八卡模型、Rust GPU/L3/LoRA路由/retention测试，不沿用旧模型分数充当本轮产物测量。用户要求本轮任务收尾后停止目标运行；完成当前验证和本地提交后暂停，完整DAS目标的未完成项保留。

[dev76分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-rust-events-gates-validation-20261101.json`）：659包文件与源码/wheel/site相同，wheel SHA256 `234671c432573204f83d94558dd348f5dd0dc48f51b173fbf1d64a9c6d82cfe7`。487/487 Hook、4个seam、0 errors；10项provider、12项KV事件、本地/容器95项仓库测试通过。真实Rust auto及never缓存复用均通过6组default/Dynamo/namespace事件、12组Rust/C++/hashlib hash、4次负数拒绝及8行SWA恢复；两次binding SHA相同，缺失官方manifest时仍选中owned Rust，返回FULL索引已断言并消费，无ignored destructor错误。完整crate执行`cargo test --locked --release --lib`为930 passed/0 failed/1 ignored；ignored为Source原有不可构造的colliding-page panic用例，未新增skip。固定输入SHA/clean和全部Source资产哈希通过。没有新增八卡服务或数据集测量；本轮收尾后按用户要求暂停目标，剩余DAS工作不标记完成。

提交前的diff检查去除owned events.py末尾多余空行，最终另行冻结dev77：wheel SHA256 `b989298e06102ef3410d1f8836a5ad42cc64bebb4c4239e15ddcba5d0b2aee39`，659文件与源码/site一致。与已测dev76逐文件比较仅该EOF换行不同，其他658包文件完全相同，pyproject仅变版本；dev76门禁保留原归属。dev77另行通过487 Hook及真实never模式全部事件/hash/负数/SWA合同，binding SHA与dev76相同；没有将空行修正描述成算法变更，也没有覆盖原冻结产物。最终本地提交完成后暂停目标。

## 仓库命名与文档清理（2026-11-01）

按 plugin-v3-1101 需求统一需提交文本中的 HCU 命名及大小写，重命名迁移 Skill，移除已由 HCU MLA 模块替代的旧兼容文件。Git 忽略的来源树、环境、构建产物和原始日志不参与替换。

删除 docs 下 95 份不被工具直接消费的 JSON 历史报告和清单，文档索引转到历史快照说明；plans 中的历史证据采用 `git:<提交>:<路径>` 定位。当前工具输入与目标合同继续保存在 plans、templates 与 tests/contracts。重建两份网页；计划签名随纯命名替换刷新，仍保留原采集基线与历史验证语义，本次没有重新采集来源或重做 GPU 验证。

### 历史 JSON 快照

删除的 JSON 可从清理前提交读取，不依赖远端是否已推送。例如在仓库根目录执行：

```bash
git show de390225d615381d3fd02b8d3eed639e5ef529b5:docs/hcu-sglang-targets.json
```

其他历史报告以引用处给出的文件名替换命令中的文件名。清理前工作区备份和本次检查日志位于 ignored `artifacts/plugin-v3-1101-cleanup/`。本次验证结果在提交前记录；历史模型、GPU 和性能结果不视为本次复验。

### 本次检查结果

在指定容器、Python 3.10.12 下执行：L0 通过（474 个 Python、8 个 Skill）；95 项 unittest 通过；229 处 Markdown 锚点核对通过；网页导览、迁移计划快照和执行台账一致性检查通过；需提交文件的旧硬件命名残留为零，docs 下 JSON 为零，Git diff 空白检查通过。

容器默认 site-packages 中的 SGLang 缺少 `ReplicatedParallelGroup`，不满足本仓固定接口；显式使用固定官方 `baa9befa38` 的 python 目录后，10 项 current_linear CPU 回归通过，真实入口加载注册并应用 487/487 个 Hook、4 个具名 resolution seam，加载错误为零。未安装、更新或修改官方源码与容器依赖。

本轮为命名及文档清理，没有执行模型、GPU 数值或性能复验。界面控制权限未开启，网页实际浏览器截图、交互和窄屏视觉验收未执行；生成快照与 Markdown 路径/锚点检查不能替代该视觉验收。

## DAS 分支逻辑对齐（2026-11-02）

需求为 plugin-v3-fix，本节汇总固定来源的模块处置、验证范围与未完成条件；原独立对齐记录已整合到本开发记录。提交按模块组成同一变更系列，存在模块间依赖，使用完整系列进行集成验收。来源只读；仅本地提交。

### 公共运行时与调度

修正 HIP 检测及 FP8/provider 分支，补齐环境变量与 ServerArgs 派生选项；迁入 Req/ScheduleBatch/ForwardBatch 生命周期、CP 和 DSA helper，显式接通冻结配置、元数据和 hidden state 消费者。核心单元 9 项、生命周期 7 项、Forward/CP 13 项及既有 CPU 调度组 64 项通过。未执行完整模型服务；具体 GPU/配置边界见本轮记录与测试。

### 计算与量化

迁入 MegaMoE、DeepEP INT8、Marlin 与 DP 数据流并修正 routed scaling、SlimQuant 返回契约。29 个 runtime 脚本 197 项、13 项针对性单元和 GPU W4A8 3 项通过；deepgemm.mega 完整数值/Graph 未验证。

### Attention 与模型

补齐 MLA/FLA/GDN、Qwen/Bailing/GLM 与多模态调用，保留上游接管及正确 KV/通信归属。CPU 15 项、FLA GPU 4 项通过；完整模型与所有 provider 未验证。

### 缓存与通信拓扑

修复容量、LightOP、SWA/DSV4/MLA、cache builder/retention、DP/CP group 和 prefix 协议；接入完整 LayerSplit page plan producer/consumer。allocator 11、topology 10、prefix 6、LayerSplit CPU 2 项以及真实 Gloo 进程组通过；GPU scatter/gather 在 LayerSplit 测试中用 CPU 数学夹具。

### KDA 配置、池与执行

成对迁入配置生产、原生类型身份、Mamba slot/ring 语义、KDA backend、状态 kernel 与 verify/commit/rollback。配置 7 项、池 CPU 4 项、HIPGraph replay 及 KDA GPU 5 项通过；全模型尚未测试。

### PD 与 DSpark

统一 hidden state 所有权，补齐 prefill/decode/scheduler/ACK/drain/清理、描述符传输与 DSpark worker/model/injector。transport 8、frontend 7、HiCache 5、DSpark 6 项通过，覆盖真实 HTTP/ZMQ 与受控 CPU 字节复制；完整多进程 PD 服务和 TP GPU 传输未验证。

### 聚合、打包与最终门禁

根注册链接通配置、CP/DSA/ForwardBatch、KDA 和 DSpark，package-data 包含 HIP 状态索引源文件；当前 target 合同迁到 tests/contracts，修复安装与 Skill 中失效的旧 JSON 路径。117 项仓库单元全部执行通过；L0 556 Python/8 Skill、226 个锚点及三份生成页面/台账快照检查通过。源码加载 597/597 Hook、4 seam、0 errors；721 个 wheel 包文件与最终源文件相同。具体 SHA 与验证边界见本轮记录。未部署、未推送，完整模型与缺失 vendor provider 的 lane 保持未验证；网页实际视觉检查受界面权限限制未执行。

## 2026-11-02 当前插件内嵌源码生成

按 hcu-plugin-backtrack Skill，将当前插件与固定官方 `baa9befa381511cd4d56f83980951e5a45fffc6d` 组合为 `output-backtrack` 独立 Git 源码树，输出分支为 `embedded-hcu`。原插件存在未提交的文档整理，故在 ignored `artifacts/backtrack-20261102/plugin-input` 保存完整工作区快照；生成输入提交为 `00a9be2c935d0b237fbd7f06ed53f57e40ffa6e0`，运行时文件逐字节等于原插件。没有在原仓库自动提交或清理用户变更，原两输入 HEAD/索引哈希不变。

生成器 v5 补齐 KDA HIP、KV C++、Rust crate 和 AITER 配置的 package-data。输出复制 721 个插件文件，726 项清单 SHA256 核验通过；314 个非 Python 包资源均被打包规则覆盖。上游受跟踪修改只有 bootstrap、插件 loader 与 pyproject，来源/许可另由生成器清单记录。

在仓库外、仅输出 python 的 PYTHONPATH 下屏蔽全部外部 entry point，并清空 SGLANG_PLUGINS，验证 `sglang` 和 `hcu_sglang` 均来自内嵌输出；597/597 Hook 全部应用，4 个具名 seam、0 errors。工具仓库 117 项单元测试通过，其中 13 项需显式开启的层运行时测试未执行。本轮没有构建输出 wheel、启动模型服务或进行数值/性能验收；只验证源码生成、资源规则和隔离加载。原始证据保存在 `artifacts/backtrack-20261102/`。

## 附录：布局、操作方式与汇总

以下为布局说明、使用方法、维护规范及跨阶段验证汇总，不作为单独时间事件排序；表内版本和历史结果保留原有含义。

## 1. 当前布局与接口

| 对象 | 当前定义 |
|---|---|
| 代码目录 / Python 包 | `hcu_sglang/` / `hcu_sglang` |
| distribution / 开发版本 | `hcu-sglang-plugin` / `0.3.2.dev62` |
| general plugin entry point | `hcu = hcu_sglang:register` |
| 显式服务入口 | `python3 -m hcu_sglang` 或 `hcu-sglang-serve`，参数传给官方 launcher |
| 来源插件版本 | `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，原包 `sglang_dcu_plugin` |
| 官方接口基线 | `baa9befa381511cd4d56f83980951e5a45fffc6d`，SGLang 0.5.22 |
| 来源清单 | [逐文件迁入快照](hcu-sglang-development.md#历史-json-快照)（`hcu-sglang-import-manifest.json`），用于本次迁入审计，不是未来文件完整性锁 |

复制来源仅为输出仓的运行时包。没有复制外部 docs、tools、tests 或其他生成脚本。包内 7 份不参与运行的来源说明 JSON 未放入运行时目录；源路径、哈希与排除理由保存在迁入清单。保留原生资源、继承的注释/docstring、版权与 vendor LICENSE。

原来的 `usercustomize.py` 位于来源包之外，未作为全局 Python 启动文件迁入。新增包内显式 launcher 先安装同一惰性 AITER shim，再转发官方服务；普通 Python 子进程及冷导入不自动加载 Torch/AITER。

## 3. 当前安装与验证方式

先准备匹配的官方 SGLang、Torch/HCU、DTK 和 vendor 依赖；本次 wheel 不代替该环境。按 [配置管理](configuration.md#1-当前工具的路径配置) 提供真实本机路径，安装时使用隔离环境或 task target，避免同名旧入口重复加载：

```bash
# 从本仓根目录执行；仅安装当前插件，不补齐 vendor/HCU 依赖。
python3 -m pip install -e . --no-deps --no-build-isolation
export SGLANG_PLUGINS=hcu
# 追加本次模型、设备、TP、backend、量化与 Graph 等完整配方。
python3 -m hcu_sglang --help
```

严格加载工具默认保留旧 Python API/命令兼容；验证当前包时显式选择命名空间：

```bash
: "${OFFICIAL:?请设置固定官方源码路径}"
: "${ARTIFACTS:?请设置本次证据目录}"
REPO="$(pwd -P)"
python3 tools/validate_plugin_runtime.py \
  --official-python "$OFFICIAL/python" --plugin-root "$REPO" \
  --plugin-package hcu_sglang \
  --expected-targets tests/contracts/hcu_source_alignment_targets.json --output "$ARTIFACTS/hcu-runtime.json"
```

记录实际解释器、entry point、模块文件来源、目标集合与具名 seam、加载软错误，以及是否误导入旧包。当前 entry point 名为 `hcu`，由固定官方 loader 对 `SGLANG_PLUGINS` 中的名称精确匹配；不是 Python 包名后缀自动扫描，也不保留 `hcu` 别名。升级后须重新安装本仓或新 wheel 以刷新 metadata，再设置 `SGLANG_PLUGINS=hcu`。安装其他插件时同样核对实际白名单和模块来源，不能凭入口存在或目标数量认定通过。源码、隔离 wheel 和完整服务分别验收。

## 5. 后续代码开发与记录规范

每次代码改动追加台账，写明完成模块/行为、源码或前置提交、提交 SHA/标题及文件范围、实际执行的验证和原始证据、未验证模式、已知限制与下一步。提交后的代码 SHA 可在后续记录提交中回填，避免用文档自身未产生的 SHA 伪装已提交。

当前方法抽取、差异清单、处置、签名检查和 embedded 生成已支持当前命名空间。方法 profile 与实际类拆分仍单独预检；不因包名支持而自动执行会覆盖既有适配的方法同步。

## 当前八卡验证汇总

本轮四类代表服务已按各自原八卡配方完成六个严格请求；每行属于对应隔离wheel，不能外推到其他版本或全量DAS。汇总哈希与运行ID见[结构化摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-eight-gpu-validation-20261101.json`），原始请求/响应/日志保留在指定test-model目录。

| 服务 | 实际版本 | 并行与关键模式 | 请求/清理 |
|---|---|---|---|
| Qwen3-235B-A22B-Instruct BF16 | dev34 | TP8、fa3、内存0.85、默认Graph | 6/6 stop，自有进程为空、端口关闭 |
| MiniMax-M2.5 BF16 | dev35 | TP8、FP8 KV、context131072、默认Graph | 6/6 stop，自有进程为空、端口关闭 |
| GLM-5 INT4 | dev37 | TP8、EAGLE、FP8 KV、AITER IPC、auto→LightOp | 6/6 stop，自有进程为空、端口关闭 |
| DeepSeek-V4 FP8 | dev37 | TP8/DP8/EP8、DSPARK、DeepEP、context32768、graph32 | 6/6 stop，自有进程为空、端口关闭 |

全量代码/依赖审阅仍在进行，不把四类代表服务或文件标签数记作整个DAS已完成。

## 2026-11-02 PR #4 审核修复与目标分支整合

恢复历史 wheel 的 `sglang_dcu_plugin` 文件名、原入口及版本归属，保留历史 SHA256，不把当前 `hcu` 命名追溯为旧产物重建。修正 `group_concurrent_contiguous` 的返回注解为两组 Python 整数列表，保留 `.tolist()`、空输入与长度校验。失效的独立对齐记录链接改到本记录已有章节。

合入目标 develop 的既有提交，保留其命名调整和 adapter 姊妹仓文档；冲突处采用本 PR 较新的实现，保留已整合或明确删除的旧文档状态，重新生成导览。目标分支运行时改动已核对仅为命名替换；本轮不回退 PR 的新增运行时功能。

已核实的误报不作修改：盘点 JSON 实际受 Git 跟踪；hidden chunk 异常由 decode 队列捕获并标记单请求失败。本轮修改不声称完整模型或 GPU/PD 服务重新通过。验证结果随本次 PR 修复提交记录。

本次复验：L0 的 556 个 Python 文件和 8 个 Skill 通过；117 项仓库测试中 104 项通过、13 项需专用运行环境而跳过；243 个 Markdown 锚点、导览/迁移规划/执行台账快照及 diff 空白检查通过。已检查实际桌面、390px 窄屏渲染和文档跳转。未执行专用层运行时、GPU 或完整 PD/模型服务复验。

## 2026-11-02 fa3 混合线性模型 paged KV 池判定修复

完成模块/行为：`hcu_sglang/layers/attention/flashattention_backend.py` 的 `around_get_paged_mha_kv_cache` 钩子补齐混合线性注意力模型形态。此类模型（qwen3_5 等）的 `backend.token_to_kv_pool` 是上游 `HybridLinearKVPool` 聚合池，HCU page-major 池挂在其 `full_kv_pool` 属性上；原判定只认裸 `HCUMHATokenToKVPool`，混合形态下判定不成立、放行上游 `get_paged_mha_kv_cache`，上游按 HND 校验拒绝 HCU 转置 V 布局，prefill 图捕获首个全注意力层 warmup 即断言退出。改动后直接形态或 `full_kv_pool` 内池形态均可认出；取数仍走聚合池 `get_kv_buffer`，保留其全局层号到全注意力稠密层号的映射，不能用内池直接取数。非 HCU、纯 MHA 模型、`SGLANG_KV_LAYOUT_HCU_FA=false`（内池为上游布局）三种场景判定与放行行为不变。

问题来源：dev34 期即发现（B4 崩于 `paged_kv_view` HND 断言）；当前 develop 复现同签名，属一直未覆盖的混合形态，与仓库内既有 `full_kv_pool` 解包写法（disagg/utils.py、disagg/prefill.py、mem_cache/cache_builder.py 等）对齐。

实际验证（Qwen3.6-27B，TP2 gfx936，fa3，page_size=64，develop + current_dsa 导入修复，本机日志 dev77-fix1/fix2/fix3 对照）：HND 断言由每次启动 3 次降为 0 次；prefill 图捕获 58 个形状全部完成（修复前死于首个形状 warmup 后约 2 秒），decode 图捕获 bs=1/2/4/8/12 全部完成；修复后服务推进至下一独立死点（disagg 初始化调用上游已移除的 `Scheduler._init_waiting_queue_prefetch_runtime`，AttributeError），与本修复无关，待单独处理。

未验证模式：纯 MHA 模型回归（判定为等价改写）、SWAKVPool 聚合形态、PD 分离模式、`SGLANG_KV_LAYOUT_HCU_FA=false` 显式组合、八卡代表服务复验。

已知限制与下一步：disagg/scheduler.py:93 对上游 Scheduler 方法名的假设需加守卫或对齐新名；混合模型状态池并发上限（mamba 槽位）与 KV 配比未调优。

## 2026-11-02 disagg 调度钩子对官方 sglang 缺失方法的 setdefault 兜底

完成模块/行为：`hcu_sglang/disagg/scheduler.py` 的 `register_hooks` 注册期对 `Scheduler` 做方法级 setdefault。b4ae852 引入的 `init_disaggregation` 整体替换钩子在一切 disaggregation_mode 检查之前无条件调用 `self._init_waiting_queue_prefetch_runtime()`；该方法本仓无定义、官方 sglang 无此 API，仅内部 sglang fork 提供。上游 `Scheduler.__init__` 对 NULL 模式也会调用 `init_disaggregation`，REPLACE 钩子又使上游原版不再执行，导致官方 sglang 上任意模型、任意模式启动即在 scheduler 初始化处 AttributeError。

兜底方式与 `layers/quantization/registry.py` 的 `install_quantization_method` 同范式（探针、填坑、宿主优先，极性反转）：`hasattr(Scheduler, ...)` 为真（内部 fork 或未来官方实现）时不做任何修改，为假时安装 no-op 默认实现并记录一条 warning。调用点不改；提供该方法的 sglang 构建上行为逐字节不变。

实际验证（Qwen3.6-27B，TP2 gfx936，官方 sglang v0.5.22 线，NULL 模式，本机日志 dev77-fix3 与 dev77-fix4-disagg-guard 对照）：修复前双 rank 在 `init_disaggregation` 内 AttributeError、服务不可启动；修复后双 rank 各一条 warning、初始化通过，Uvicorn 监听后输出 ready，冒烟请求 health 200、两道确定性问答回答正确（temperature=0）。门禁：L0、仓库测试与导览快照检查通过（见本记录门禁口径）。

未验证模式：PD 模式在官方 sglang 上（decode/prefill 集成可能仍依赖更多内部 fork API，需同范式逐个探测或声明仅支持内部 fork）；内部 fork 环境回归（hasattr 短路，理论零扰动）；纯 MHA 等其他模型逐一回归（本问题与模型无关，崩点在 scheduler 初始化）。

已知限制与下一步：b4ae852 的 PD 集成面向内部 fork API 面，建议对官方 sglang 的兼容矩阵单独盘查；非 PD 服务路径经本修复在官方 v0.5.22 线端到端恢复。


## 2026-11-03 投机首启崩溃修复（draft_swa_window 垫值）

官方基线投机路径首跑（TP2、NEXTN、fa3、page64、Qwen3.6-27B）在 Scheduler.__init__ → build_kv_cache 的 AROUND 钩子崩溃：cache_retention.py 读取 draft.draft_swa_window，官方 v0.5.22 的 EAGLEWorkerV2 无该属性（官方树全仓 0 命中，系内部 fork 表面属性）。非投机路径 draft 为 None 不触发，故此前的非 spec 服务全绿。

修法沿用 PR #7 的 registry setdefault 范式：register_hooks 注册期对 BaseSpecWorker（EAGLEWorkerV2 / DFlashWorkerV2 公共基类）做属性级 setdefault，官方树缺属性时垫 0（builder 侧断言要求 SWA 模型才有非零语义，Qwen3.6 混合形态恒为 0），fork 或上游自行定义时一个字节不碰，自退役。运行时改动 2 行无注释。

验证：L0 门禁 556 Python 文件 + 8 Skill 零错误；117 项测试 104 过 13 跳（与基线一致）；build_guide 重建与 --check 通过。真机 TP2 NEXTN（日志 dev77-spec-fix1.log）13:55:37 ready，冒烟两题 巴黎/323 全对，accept len 2.48 / accept rate 0.49（Stage C 健康区间 2.4~2.9 / 0.48~0.62）。未验证：DFlash 等非 EAGLE draft worker、SWA 目标模型上的非零窗口语义（官方树无该属性来源，恒为 0）、PD 模式。


## 2026-11-05 V4 平台类与方法同步（模型侧）

本轮仅修改模型侧及对应同步工具、测试和台账。Attention、量化、缓存、通信、disagg 由另一开发者并行负责，模型内部 `current_v4` 仍属于模型侧。来源固定为 DAS `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`，官方 `baa9befa381511cd4d56f83980951e5a45fffc6d`，插件起始 HEAD `1eb4abaa3a3ad5bb7bbc9a8b758488639f4db1c7`；来源只读，未 commit、push、修改环境配置或运行 GPU 用例。

### 来源、实现与支持边界

下表来源路径均相对于 DAS `python/sglang/srt/models/`；静态发现不等于完整实现或运行验证。

| 来源模块 | plugin 模块/接线 | 支持或回退条件 | 本轮验证与剩余边界 |
|---|---|---|---|
| `deepseek_v4.py` Base/CUDA/Platform decoder | [core.py](../hcu_sglang/models/deepseek/current_v4/execution/core.py)、[platform_stage.py](../hcu_sglang/models/deepseek/current_v4/execution/platform_stage.py)、[decoder Hook](../hcu_sglang/models/deepseek/deepseek_v4/hooks/decoder.py) | HCU 将公开 decoder 别名替换为 Platform 类，Mixin 先于 Base；非 HCU 不注册该替换 | CPU 验证 MRO 初始化、NextN 别名、PP group 和 5 个 stream 构造；CP/TP、GPU kernel 和完整 decoder 数值未验证 |
| `deepseek_v4.py` Model/CausalLM | [model Hook](../hcu_sglang/models/deepseek/deepseek_v4/hooks/model.py)、[model facade](../hcu_sglang/models/deepseek/deepseek_v4/methods/model.py)、[core.py](../hcu_sglang/models/deepseek/current_v4/execution/core.py) | HCU 安装缺失的 `_forward_impl` 并注册正式 REPLACE Hook；非 HCU 早退 | 官方公开 forward 已实际执行 owned PP body，验证展平后数值和存储地址；无层 PP 夹具不验证 Attention/MoE、首尾 rank 或 TP 通信 |
| `deepseek_v4.py` MQA 与 DSpark subclass | [attention.py](../hcu_sglang/models/deepseek/current_v4/attention.py)、[MQA Hook](../hcu_sglang/models/deepseek/deepseek_v4/hooks/mqa.py) | HCU 注册完整 MqaAttentionBase/MQALayer 和 DSparkAttention；其他平台保留既有方法 Hook 路径 | 本轮只核对注册及 mHC 类型身份；完整 MQA、DSpark、Graph 计算未验证 |
| `glm5_next.py`、`hcu/glm5_next.py`、`glm5_next_vision.py` | [owned HCU 模型](../hcu_sglang/models/glm5_next/model.py)、[vision](../hcu_sglang/models/glm5_next/vision.py)、[正式注册](../hcu_sglang/models/registry.py) | HCU 按架构严格注册；其他平台沿用原 registry；缺依赖直接报错 | 已迁入完整计算 body，保持 in_progress；专用 CP/MHC communicator、GLM CP 和 legacy layer boundary 未齐，完整导入与模型执行未通过 |
| `glm5_next_nextn.py` | [owned NextN](../hcu_sglang/models/glm5_next/nextn.py)、[draft entries](../hcu_sglang/models/glm5_next/entries/nextn.py) | 保留来源两个 NextN 架构、draft mapper 和 HCU CausalLM loader | 隔离 CPU 检查映射、权重过滤、量化选择和输入顺序；完整导入、真实 loader 与 draft 执行未验证，保持 in_progress |
| `muse_spark.py`、`muse_spark_omni.py` | [owned text](../hcu_sglang/models/muse_spark/model.py)、[omni](../hcu_sglang/models/muse_spark/omni.py)、[omni entry](../hcu_sglang/models/muse_spark/entries/omni.py) | HCU 按需严格注册 MuseSparkForConditionalGeneration；来源 text 无独立 EntryClass，不额外扩展公共架构 | 已迁入完整 body，CPU 检查归一化与模拟 gather 排序；专用 layers/config/processor/kernel 未齐，保持 in_progress |

执行台账共有 122 个 `python/sglang/srt/models/` 非 same 文件，本轮只审上述明确入口和 V4 接线；其余模型、V4 helper 的全部行为仍需逐项核对，不按该表宣称模型领域完成。人工处置将 V4、GLM/Muse 及共享 visual 的已审模型入口记为 in_progress，并分别保留依赖与验证边界；未将源码迁入提升为 implemented 或 validated。

### 修复与方法同步

owned `DeepseekV4Model.forward` 调用 `self._forward_impl`，固定官方类无该属性，旧 Hook 也未安装。以原 HEAD Hook 重放公开 forward，确实得到 `AttributeError: 'DeepseekV4Model' object has no attribute '_forward_impl'`；原失败日志保存在 `artifacts/migration-20261105/v4-before.log`。修复先为缺失属性安装 owned helper，再注册它的正式 Hook，并更新 [target 合同](../tests/contracts/hcu_source_alignment_targets.json)。原生已有属性时不先覆盖其定义，实际 HCU 行为仍由 HookRegistry 接管。

[同步工具](../tools/sync_deepseek_methods.py) 增加显式 `--profile current_v4`：对来源与 owned 平台继承顺序和 Model/CausalLM 全部导出方法预检，只生成两份 `C.Class.method` facade，保留方法的类上下文与 `super()`。默认 legacy 不变；所有预检完成后才写输出。该检查不比较方法体，不同步完整计算实现，完整文件仍为 in_progress。调用与退出码见 [迁移指南](migration.md#4-方法抽取与升级验证)。

[工具回归](../tests/test_method_sync_preflight.py) 新增来源 MRO 漂移、owned helper 缺失时零输出、仅生成 facade 三项；[V4 CPU 回归](../tests/runtime/v4_execution_current.py) 新增公开入口 helper 调用及真实 PP body 测试。首次测试夹具 patch owned 方法会受 lazy resolver 缓存顺序影响，已改 patch 公开类属性；业务修复没有为 mock 改动。

### 本轮验证与证据

验证使用既有容器 `yangyn_sglang_plugin_1022`，Python 3.10.12，源码模式，设置 `PYTHONPATH="$OFFICIAL/python:$PLUGIN"`、`SGLANG_PLUGINS=hcu`，entry point `hcu_sglang:register`。这里明确使用固定官方树，防止 fork 本身补齐缺失 API 掩盖 plugin 缺口；第二阶段 DAS A/B 另按各自 recipe 记录实际源码。宿主工具可用 `/opt/saltstack/salt/bin/python3.10`（3.10.19），默认 3.6.8 未用于验证。

已执行：

- `python3 tools/sync_deepseek_methods.py --fork-python "$FORK/python" --plugin-root "$PLUGIN" --profile current_v4 --check`：两份 facade 均 unchanged，退出 0。
- `python3 -m unittest tests.test_method_sync_preflight -v`：15 项通过。
- `python3 -m unittest tests.runtime.v4_execution_current tests.runtime.v4_logits_current -v`：9 项通过；日志 `artifacts/migration-20261105/v4-after-fixed.log`。
- `python3 tools/validate_plugin_runtime.py --official-python "$OFFICIAL/python" --plugin-root "$PLUGIN" --plugin-package hcu_sglang --expected-targets tests/contracts/hcu_source_alignment_targets.json --output artifacts/migration-20261105/runtime.json`：598 registered / 598 applied、4 个具名 resolution seam、0 errors；保留加载 warning，不称所有 vendor API 已通过。
- `python3 -m unittest discover -s tests -v`：120 项，107 通过、13 skipped；日志 `artifacts/migration-20261105/unittest.log`。

L0、台账与导览快照检查在完成本节后另行记录。未执行 GPU、模型服务、精度/吞吐/显存 A/B、CP/TP/Graph、GLM/Muse 实跑、wheel 冻结、网页截图/交互/窄屏视觉验收；阶段一的模型逐方法迁移与全领域收口尚未完成。ignored 原始日志不进入 Git，长期统一测试台账由测试校对家维护，不将历史模型分数挪为本轮证据。


#### 模型侧补充检查

V4 的四份 helper 已逐函数对照固定来源，原始报告为 `artifacts/migration-20261105/v4-helper-audit.json`，记录来源/owned SHA256、完整方法名和差异。`platform_stage.py` 为 24/24 个方法、23 个方法 AST 相同；`boundary.py` 为 29/29、28 个相同；`replay_tail.py` 为 10/10、9 个相同；`mhc.py` 为 20/20、20 个相同。三个方法差异分别是转向 owned mHC、转向 owned fused_mhc 和延迟导入 RowBucketGraphRunner；已逐项核对。此统计保留 decorator/docstring，仅忽略位置，不检查模块 imports/globals、类基与消费者等价，也不代替 GPU/Graph 验证。

本轮门禁实际执行 `/opt/saltstack/salt/bin/python3.10 tools/build_migration_execution.py` 及 `--check`、`tools/build_guide.py` 及 `--check`、`ci/l0_gate.py`、`git diff --check` 均退出 0。L0：557 Python / 8 Skill / 0 errors；执行台账为 pending 2887、implemented 44、in_progress 53、upstream 26。对应原始日志为 `artifacts/migration-20261105/gates.log`。该计数是当次工作树快照，不能解释为模型等价完成率。仓库单测的 13 项 skipped 均由 `HCU_PLUGIN_RUN_LAYER_FIX_TESTS` 未启用导致，未将它们记为通过；运行时领域由独立开发者负责。


补充执行 `python3 tools/validate_hook_signatures.py --official-python "$OFFICIAL/python" --plugin-root "$PLUGIN" --output artifacts/migration-20261105/hook-signatures.json`，退出 1：130 match / 28 mismatch / 14 missing。14 个 missing 涉及该静态工具不跟随的 facade 重导出，以及官方源码没有但注册期安装的 `_forward_impl`，不能当作真实加载缺失；严格加载与公开 forward 测试另有证据。28 个 mismatch 未整体豁免：静态调用形状样例发现 7 个 helper 存在旧调用缺必选参数或 width_align 不被接受的风险，另有 DeepseekV2MLP.forward 的第 5 个位置参数由 gateup_pre_quant 变为 input_quant_args；具体条目保存在 `artifacts/migration-20261105/signature-triage.json`。这些属于待核对的消费者/分支兼容范围，尚未据此判定全部为运行故障或无害。当前 V4 profile 的 14 个来源/owned 方法签名结构已逐项比较一致，签名一致不证明方法体等价。该诊断工具失败与 L0、仓库单测通过分开记录。


## 2026-11-05 GLM5-next 与 Muse Spark owned 模型迁入

按群内确认的方案 A，模型侧先迁入完整 owned 计算实现，再通过官方公开 `ModelRegistry.register(package, overwrite=True, strict=True)` 接入架构。新增 [模型 resolver Hook](../hcu_sglang/models/registry.py)，由已有 [models.utils](../hcu_sglang/models/utils.py) 注册链安装。仅 HCU 且请求目标架构时导入对应族；检查解析结果必须来自 owned module 前缀，依赖缺失透传异常，禁止静默使用官方 GLM。无关模型不加载这两族；非 HCU 保留原 resolver。这里的支持条件由隔离注册合约验证，尚无完整模型加载通过证据。

### 来源保留与依赖对应

[来源合同](../tests/contracts/owned_model_sources.json) 固定 DAS `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`、官方 `baa9befa381511cd4d56f83980951e5a45fffc6d` 和六份来源 SHA256。测试将 owned 中显式列出的 import 替换逆转后，逐字节校验 SHA256；计算 body、源码注释和许可原样保留，不加来源标记或说明性注释。

| DAS 来源（相对 models） | owned 模块 | 正式入口 / 条件 | 当前边界 |
|---|---|---|---|
| `hcu/glm5_next.py`，1918 行 | [model.py](../hcu_sglang/models/glm5_next/model.py) | [text entry](../hcu_sglang/models/glm5_next/entries/text.py)：Glm5NextForCausalLM / Glm5NextForConditionalGeneration；取代 HCU 别名 | 五个专用依赖模块未齐；in_progress |
| `glm5_next_vision.py`，409 行 | [vision.py](../hcu_sglang/models/glm5_next/vision.py) | HCU GLM 内部视觉模型 | 源码保留检查通过；Attention/MLP、完整视觉模型未执行 |
| `glm5_next_nextn.py`，91 行 | [nextn.py](../hcu_sglang/models/glm5_next/nextn.py) | [NextN entry](../hcu_sglang/models/glm5_next/entries/nextn.py)：两个 NextN 架构 | 隔离父类与 loader 后验证实际 owned 方法；完整链仍阻塞 |
| `glm_visual.py`，95 行 | [visual.py](../hcu_sglang/models/glm5_next/visual.py) | GLM 共享图像/视频入口 | 仅隔离 CPU helper；未执行 ViT 或真实 DP 通信 |
| `muse_spark.py`，1600 行 | [model.py](../hcu_sglang/models/muse_spark/model.py) | omni 内部 text 组件；来源未声明独立 text EntryClass | 专用 kernels/layers/config 未齐；in_progress |
| `muse_spark_omni.py`，476 行 | [omni.py](../hcu_sglang/models/muse_spark/omni.py) | [omni entry](../hcu_sglang/models/muse_spark/entries/omni.py)：MuseSparkForConditionalGeneration | audio/NaViT/config/processor 等依赖未齐；in_progress |

GLM 复用已有 owned fused_rms_quant、GLM runtime accessor、DSA CP predicate 与 V4 HCU CP helper。HCU gated norm 指向已有 [fused_norm_gate.py](../hcu_sglang/layers/attention/linear/kda_kernels/fused_norm_gate.py)，整个 481 行文件与 DAS HCU 来源逐字节相同，双方 SHA256 均为 `ac347fd56dc1165a710552c852d2d3445ca6c8ebf7622d151ebf68906218c16d`；[policy](../hcu_sglang/glm5_next_policy.py) 中 `get_glm5_next_runtime_args` 的 AST 与来源 accessor 一致。这些是来源复核，不等于已执行相应 kernel 或 CP 通信。

仍保留来源 import 的五个 GLM 专用缺口为 `communicator_glm5_next_cp`、`communicator_glm5_next_mhc`、`communicator_glm5_next_mhc_cp`、`layers.utils.glm5_next_cp` 和 `layer_boundary.hcu_legacy`。运行时负责人尚未给出已迁入的 owned 路径，模型侧没有改成不存在的路径。Muse 还需要八个 kernel Python 包、七个专用 layer 模块，以及 `configs.muse_spark`、`multimodal.processors.muse_spark_omni`；已有 JIT 头资产不能替代这些 Python 实现。逐 import 的路径存在性与复用 SHA/AST 证据保存在 `artifacts/migration-20261105/owned-model-dependencies.json`；该报告不校验导入符号、动态 seam 或运行行为，不能只凭路径缺失判断所有 import 均失败。

### 本轮 CPU 验证及失败记录

- `/opt/saltstack/salt/bin/python3.10 -m unittest discover -s tests -v`：133 项，120 通过、13 skipped，退出 0；日志 `artifacts/migration-20261105/unittest-owned-models-final.log`。新增 [模型合约测试](../tests/test_owned_model_registry.py) 共 13 项，覆盖六份来源保留、entry 架构集合、严格按族注册、幂等、缺依赖重试、禁止官方回退、非 HCU 早退，以及真实 NextN 文件在隔离依赖下的 mapper/loader/量化/CP 输入顺序。13 项 skip 仍是未启用 `HCU_PLUGIN_RUN_LAYER_FIX_TESTS`，不记为通过。
- 容器 Python 3.10.12、源码模式 `PYTHONPATH="$OFFICIAL/python:$PLUGIN"`、`SGLANG_PLUGINS=hcu`，同时设置 `HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1`。`python3 -m unittest tests.runtime.owned_models_cpu -v` 等效入口运行 [隔离 CPU 测试](../tests/runtime/owned_models_cpu.py)：11 项通过，`CUDA_INITIALIZED False`；日志 `artifacts/migration-20261105/owned-models-cpu.log`。从实际 owned 文件 AST 选择原样定义并隔离依赖，验证视觉分块顺序/限额/超大单图、dtype、视频帧 grid、缺视觉编码器报错、DP 调用边界，以及 Muse RMS 数值、embedding 权重复用、单 rank 编码、模拟 gather 还原顺序；不冒充整模块导入、ViT、真实 collective 或 GPU 计算。
- `python3 tools/sync_deepseek_methods.py --fork-python "$FORK/python" --plugin-root "$PLUGIN" --profile current_v4 --check`：两份 facade unchanged，退出 0；日志 `artifacts/migration-20261105/current-v4-final.log`。

完整模型导入探针没有通过。禁 GPU 的首轮在官方依赖链 `fp8_kernel` → `is_gfx1250_supported()` → `torch.cuda.get_device_properties(0)` 触发 `No CUDA GPUs are available`；Muse text 首先报缺少 `sglang.kernels.ops.muse_spark`。第二轮仅 mock 设备属性/capability 查询、不 mock 分配或 kernel，官方 `fp8_utils.use_rowwise_torch_scaled_mm()` 仍因无 GPU 返回的 `(None, None)` 与 `(9, 4)` 比较而失败。两轮 JSON 分别为 `artifacts/migration-20261105/owned-model-imports.json`、`owned-model-cpu-imports.json`；报告器退出 0 仅说明保存证据成功，各模型状态均为 blocked，不记导入通过。

注册目标合同已由 598 增为 599（新增模型 resolver）。禁 GPU 后重跑完整严格 loader，实际退出 1、0 registered / 0 applied：插件初始化失败并出现 HCU detection `No CUDA GPUs are available`，完整 599 项合同未通过。禁 GPU 重跑既有 `tests.runtime.dsv4_registry_current` 也在官方导入期设备查询失败，退出 1，没有执行那五项测试。证据为 `artifacts/migration-20261105/runtime-no-gpu.json`、`runtime-no-gpu.log`、`dsv4-registry-no-gpu.log` 和 `no-gpu-checks.json`。此前 598/598 仅是旧工作树快照，不能沿用为本轮通过结论；不扩大 mock 伪造严格加载成功。

### V4 签名消费者复核

保留原签名诊断的退出 1，不整体豁免 28 个 mismatch。消费者与分支证据追加在 `artifacts/migration-20261105/signature-consumers.json`：candidate block amax 的 wrapper/JIT 成对替换并传 dtype，FP4 split 的 wrapper/JIT 成对替换并传 page8，owned scoring 调 owned flat logits 时不传 width_align，owned backend 调 source projection 时传 run_compressor/captured_index。这些只证明已审消费者的调用形状，未验证 kernel、Graph 或外部调用。

官方 `DeepseekV4HipRadixBackend` 不继承被替换的 owned backend 类，两个旧 workspace 调用仍缺新必选参数。现有 HCU [dsv4 registry](../hcu_sglang/layers/attention/dsv4_registry.py) 将常规及 draft 路由转到官方 trtllm factory，其默认分支构造已替换 base backend；直接/备用 HipRadix 路径仍有兼容风险，未声明支持。官方 DenseBlocks 的两个调用仍传 width_align；DeepGEMM prefill 分支受 `is_cuda()`、SM100+ 与可用 FP4 indexer gate 约束，官方 `is_cuda()` 要求 `torch.version.cuda` 非空，未经改写的 ROCm 构建选择 torch 分支。owned backend 仍构造该 DenseBlocks 与独立 HCU candidate indexer，不能把全部 DenseBlocks 消费者视为已经替换；手动/其他平台开启该分支仍有风险。上述真实路由测试在本轮无 GPU 环境被导入阻塞，静态条件不升级为运行证明。

V2 MLP 保留 fork 的 `input_quant_args` / `gateup_pre_quant` 位置顺序；已审 shared_experts 消费者使用 gateup_pre_quant 关键字，GLM 等 input_quant_args 调用使用关键字。旧外部位置调用兼容性仍未证明，没有为了签名工具结果重排参数破坏来源语义。

本轮台账新增/更新七个已审 GLM/Muse/shared visual 模型键为 in_progress；V4 与模型领域其余条目不因此完成。未运行 GPU 用例、完整 GLM/Muse 模型、模型精度/性能 A/B、真实 CP/TP/Graph、wheel、网页视觉验收；不 commit、不 push、不终止他人进程。


### 模型侧收尾门禁

完成上述代码、处置与交接文档后执行台账重建及 `--check`、导览重建及 `--check`、`ci/l0_gate.py`、`git diff --check`，均退出 0。L0 为 574 Python / 8 Skill / 0 errors；非 ignored Markdown 的本地标题锚点静态检查为 249 项、0 错误。执行台账当前 pending 2880、implemented 44、in_progress 60、upstream 26；这是全仓快照，不是模型完成比例。原始命令和输出保存在 `artifacts/migration-20261105/model-final-gates.log` / `model-final-gates.json`，锚点检查为 `model-markdown-anchors.json`；不是浏览器渲染或视觉验收。来源身份复核见 `model-input-identities.json`：两份来源 SHA 未变且工作树干净，插件 main 工作树包含本轮与并行运行时修改、尚未提交。


## 2026-11-05 GLM 专用运行时 owned 接线补齐

运行时负责人完成五个专用模块迁入后，模型侧只改 [GLM 模型](../hcu_sglang/models/glm5_next/model.py) 的五个 import，并同步 [来源合同](../tests/contracts/owned_model_sources.json) 的可逆替换；计算 body、注册条件与回退规则保持原样。上一节“五个专用依赖未齐”是迁入前的历史状态，本节更新接线状态，不据此宣称完整模型可运行。

| DAS 模块（相对 sglang.srt.layers） | owned 依赖 | 支持条件与验证边界 |
|---|---|---|
| `layer_boundary.hcu_legacy` | [glm5_next_layer_boundary.py](../hcu_sglang/layers/glm5_next_layer_boundary.py) | HCU GLM 的 LayerCommunicator/模式/上下文接口；未执行完整 decoder |
| `utils.glm5_next_cp` | [glm5_next_cp_utils.py](../hcu_sglang/layers/glm5_next_cp_utils.py) | GLM CP metadata、split/gather/scatter；未执行真实 collective |
| `communicator_glm5_next_cp` | [communicator_glm5_next_cp.py](../hcu_sglang/layers/communicator_glm5_next_cp.py) | CP communicator 与 KV prefetch；未执行 CP 通信 |
| `communicator_glm5_next_mhc` | [communicator_glm5_next_mhc.py](../hcu_sglang/layers/communicator_glm5_next_mhc.py) | MHC communicator；未执行 mHC kernel |
| `communicator_glm5_next_mhc_cp` | [communicator_glm5_next_mhc_cp.py](../hcu_sglang/layers/communicator_glm5_next_mhc_cp.py) | MHC+CP communicator；未执行组合路径 |

五个目标共 14 个导入符号均在 owned 源码顶层定义或导入；来源合同逆转全部替换后，与固定 DAS 来源文件逐字节相同。依赖内部接线和计算实现归运行时负责人维护，模型侧未改其文件。映射、模型与五份依赖 SHA256 记录在 `artifacts/migration-20261105/glm-owned-wiring.json`。来源与官方 SHA 沿用上一节；这是共享脏工作树上的增量验证，不复用旧 598/598 加载结果。

本轮仅运行禁 GPU 的检查，不运行 GPU kernel、真实通信、完整模型或精度性能 A/B。Muse 专用依赖仍待运行时负责人交接。具体门禁和导入结果见本节后续记录。

实际执行 `/opt/saltstack/salt/bin/python3.10 -m unittest discover -s tests -v`：133 项、120 通过、13 skipped，8.886 秒，退出 0；包括六份来源 SHA256/替换合同与 owned 注册/NextN 合约。13 项 skipped 仍因未启用 `HCU_PLUGIN_RUN_LAYER_FIX_TESTS`。原始日志为 `artifacts/migration-20261105/unittest-glm-owned-wiring.log`。

容器 Python 3.10.12，官方源码模式 `PYTHONPATH="$OFFICIAL/python:$PLUGIN"`、`SGLANG_PLUGINS=hcu`，禁用 HIP/ROCR/CUDA 可见设备且无 mock，分别执行真实 `hcu_sglang.register()` 与 `importlib.import_module("hcu_sglang.models.glm5_next.model")`：两项均 blocked，报 `RuntimeError: No CUDA GPUs are available`，子进程退出 1，`cuda_initialized_after=false`。完整调用与 traceback 保存在 `artifacts/migration-20261105/glm-owned-wiring-import.json` / `.log`；外层报告器退出 0 只代表证据已保存。五个 owned 依赖模块可用不等于完整模型导入通过。本轮不重跑严格 599 项加载，旧成功结果仍不可沿用。

本轮结构门禁使用 Python 3.10 执行 `tools/build_migration_execution.py` 及 `--check`、`tools/build_guide.py` 及 `--check`、`ci/l0_gate.py`、`git diff --check`，并检查本节新增相对链接与台账引用锚点。逐条命令、退出码和输出保留在 `artifacts/migration-20261105/glm-owned-wiring-gates.json` / `.log`。导览仅做构建与快照检查，浏览器渲染、交互和窄屏视觉未验证。


## 2026-11-05 Muse Spark owned 依赖接线补齐

运行时负责人交付 Muse 依赖后，模型侧只改 [text 模型](../hcu_sglang/models/muse_spark/model.py) 14 处和 [omni 模型](../hcu_sglang/models/muse_spark/omni.py) 4 处 import，并同步 [来源合同](../tests/contracts/owned_model_sources.json)。覆盖顶层以及 int4、megamoe、FinalizePushWorkspace、processor 等延迟导入；两个计算 body、继承关系、原注释和 docstring 不变。上一节的 Muse 依赖未交接描述属于早期快照，当前映射如下；路径存在和静态符号可用不等于完整运行验证。

| DAS 来源依赖 | owned 依赖 | 支持与验证边界 |
|---|---|---|
| `sglang.kernels.ops.muse_spark.*` | `hcu_sglang.kernels.ops.muse_spark.*` | 接通八个直接引用的 kernel Python 模块；未执行 kernel |
| `sglang.srt.layers.muse_spark.*` | `hcu_sglang.layers.muse_spark.*` | 接通七个专用 layer 模块；BF16/FP4/int4/megamoe、audio/NaViT 完整计算未验证 |
| `sglang.srt.configs.muse_spark` | [configs_muse_spark.py](../hcu_sglang/configs_muse_spark.py) | 使用同一 owned config 定义；自动配置发现与 checkpoint 加载未验证 |
| `sglang.srt.multimodal.processors.muse_spark_omni` | [processor](../hcu_sglang/multimodal/processors/muse_spark_omni.py) | 在原方法内部延迟导入，processor 反向引用 owned omni；预处理完整执行未验证 |

来源仍固定为 DAS `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e` 和官方 `baa9befa381511cd4d56f83980951e5a45fffc6d`。18 个 import 位置及依赖 SHA256 保存在 `artifacts/migration-20261105/muse-owned-wiring.json`，两份模型逆转已声明替换后与固定来源逐字节相同。未修改运行时负责人所属文件；已通知其在本轮接线后复验 processor。运行时负责人的 Helion 缺包/门控证据单独保留，模型侧不将其转述为本轮 kernel 验证。HCU 仍按架构严格注册，缺依赖透传，不以官方回退掩盖异常；两个模型条目保持 in_progress。

实际执行宿主 Python 3.10.19 的 `-m unittest discover -s tests -v`：135 项、13 skipped，因新增运行时来源合同的 32 个 subTest failure 而退出 1。失败位于 `test_runtime_owned_sources.RuntimeOwnedSourceTests.test_only_declared_deltas_change_fixed_source_bytes`：合同列了文件中未出现的反向替换，严格 `assertIn` 报错；已通知该合同负责人修复，不放宽断言或改来源注释来迎合合同。原日志 `artifacts/migration-20261105/unittest-muse-owned-wiring.log` 保留；这不是 32 个独立测试方法，也不记全量通过。

容器禁 GPU 运行 `tests.runtime.owned_models_cpu`：11 项通过，0.048 秒、退出 0，CUDA 未初始化；原日志 `artifacts/migration-20261105/muse-owned-wiring-cpu.log`。这是原样 AST 定义的隔离 CPU 行为检查，不是完整模型。容器使用官方源码模式，HIP/ROCR/CUDA 可见设备均设 -1、无 mock；真实 `hcu_sglang.register()`、text 和 omni 整模块导入均被官方 `fp8_kernel -> is_gfx1250_supported -> torch.cuda.get_device_properties(0)` 阻塞，子进程退出 1、CUDA 未初始化。完整命令与 traceback 在 `artifacts/migration-20261105/muse-owned-wiring-import.json` / `.log`，外层报告器退出 0 只表示证据保存成功。

本轮门禁使用 `tools/build_migration_execution.py` 及 `--check`、`tools/build_guide.py` 及 `--check`、`ci/l0_gate.py`、`git diff --check`，另单独运行模型来源/注册合约并检查本节链接。逐项结果记录到 `artifacts/migration-20261105/muse-owned-wiring-gates.json` / `.log`。这是含并行修改的脏工作树，不复用旧加载结论；未重跑 599 项严格加载，未执行 GPU、真实通信、完整模型、精度性能 A/B、wheel 或浏览器视觉验收，未 commit/push。


### Muse 来源合同修正后全量复验

运行时负责人修正冗余反向替换并收紧 import 检查后，模型侧经 SSH 使用宿主 Python 3.10.19 再执行 `-m unittest discover -s tests -v`：136 项、123 通过、13 skipped，9.360 秒、退出 0。来源合同及对应测试文件在本轮运行前后 SHA256 一致；校验三份文件为 runtime source 合同、其结构测试和 model source 合同。原 135 项/32 个 subTest 失败日志保留，不覆盖历史失败记录。新日志 `artifacts/migration-20261105/unittest-muse-owned-wiring-recheck.log`，命令/返回码/前后哈希在 `artifacts/migration-20261105/muse-owned-wiring-recheck.json`。

这次只复验仓库单测，未重跑完整模型导入、GPU 或 processor，不改变此前导入 blocked 的边界。追加记录后重建执行台账和导览并检查快照，运行 L0 与 `git diff --check`；逐条结果在 `artifacts/migration-20261105/muse-owned-wiring-recheck-gates.json` / `.log`。统一 CSV 追加新结果，保留原 fail 行。


## 2026-11-05 运行时目标映射、批量 FA 与 GLM/Muse 依赖收口

### 固定来源、对应表与剩余工作

本节覆盖运行时负责人分配的 Attention、量化、mem_cache、distributed/disagg，以及本轮优先接入的 GLM/Muse 依赖。来源固定为 DAS `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e` 与官方 `baa9befa381511cd4d56f83980951e5a45fffc6d`；插件从 `1eb4abaa3a3ad5bb7bbc9a8b758488639f4db1c7` 的共享脏工作树继续，保留其他负责人的未提交修改。

逐目标结果写入 [307 行历史对应表](../plans/hcu-runtime-target-review-20261105.csv)，非平凡适配写入 [20 行差异清单](../plans/hcu-runtime-differences-20261105.csv)。每条含注册目标、DAS 路径/符号、执行台账状态、Python 等价快照、Hook 类型、owned 定义、审查结论和已有测试索引。映射只证明静态对应与注册链，不等于整文件或模型等价。

| 审查分类 | 目标数 | 含义与边界 |
|---|---:|---|
| Source/声明归一后 AST 一致 | 163 | REPLACE 定义在函数名、native 名称空间、super、文档缩进的显式规则下一致 |
| 已记录的控制差异 | 20 | 逐项列出描述符、继承、V4/GDN/缓存适配理由；其中模型侧项继续由模型负责人审调用方 |
| AROUND/AFTER 适配 | 120 | 保留当前有效 original，不把包装器文本与 Source 执行体混作 AST 相等 |
| 原生复用/装饰器后的 owned 方法 | 2 | `get_kv_class` 方法体两侧一致；Mooncake send 可定位到 owned 实现 |
| 官方独有、DAS 不适用 | 2 | `compress_hip` 的两个官方兼容目标在 DAS 无来源文件 |

本轮更新 43 个运行时 decisions 键并重建执行台账。扩展后的本领域 307 条文件级记录为 `implemented=47`、`in_progress=19`、`pending=238`、`upstream=3`；因此仍有 257 条 pending/in_progress，不能宣称全量迁移或阶段一已完成。`sglang-das-python-equivalence.json` 仍只表示固定 official/fork 的整文件 AST 关系，没有用来覆盖人工审查或生成运行结论。

### 批量 paged-varlen 的实际修复

[flashattention_interface.py](../hcu_sglang/layers/attention/flashattention_interface.py) 原先在进入 paged varlen 分支前执行 `q.view(-1, max_seqlen_q, ...)`。`max_seqlen_q` 是批内最大长度，不是拉平 Q 的固定切分长度；c4 实例总 token=1138、max=1083，必然不能整除。DAS 对应接口明确要求这种形态直接使用 `cu_seqlens_q` 和 page table。修复仅把既有 `q_4d` 构造移到 varlen 返回之后，provider 参数和 decode 的单 token reshape 不变。

新增 [flashattention_interface_cpu.py](../tests/runtime/flashattention_interface_cpu.py)，使用真实 CPU tensor 和 mock provider 保留公开入口/输出边界。修复前 c4/c8 分别报 `invalid for input of size 3495936/3686400`，c1 通过；修复后 c1/c4/c8 及 decode 共两个测试方法通过。真实 GPU/模型复跑留给后续验证，本阶段未执行。

### GLM/Muse owned 依赖

| 范围 | 本轮实现 | 验证边界 |
|---|---|---|
| GLM legacy/CP/communicator | 新增 `glm5_next_layer_boundary.py`、`glm5_next_cp_utils.py` 和三个 `communicator_glm5_next_*` 模块；复用已有 DSA、V4 CP、Mamba page-plan 与 fused-RMS owned helper | GLM text/vision/NextN 实际导入通过；未运行 decoder、CP/TP collective 或 kernel |
| Muse layers/kernels/config/processor | 迁入 9 份 layer、23 份 kernel Python、1 份 config 和 1 份 processor；统一 owned import 前缀 | text/omni/processor 实际导入通过；未执行完整模型或媒体预处理 |
| Muse Helion 兼容 | [muse_spark_spec.py](../hcu_sglang/kernels/muse_spark_spec.py) 固定 DAS backend 描述，并只补 `forward_helion` 映射/优先级 | 常规依赖导入及 CPU 能力拒绝通过；容器缺 `helion` 包，四个可选实现仍明确报错 |
| Muse 媒体配置 | [encoder_preprocessing.py](../hcu_sglang/multimodal/encoder_preprocessing.py) 以冻结 dataclass 子类增加 `preprocess_off_event_loop`、`audio_sample_rate`，经正式类 REPLACE Hook 接入 | 新旧字段、默认值和公开应用通过；未验证 encoder DP 或实际媒体执行 |

[运行时来源合同](../tests/contracts/runtime_owned_sources.json) 固定 40 份 GLM/Muse/spec 来源 SHA256、目标哈希及仅允许的可逆差异；[结构测试](../tests/test_runtime_owned_sources.py) 同时检查来源恢复、语法和家族 import 漂移。Muse 使用的 8 份 JIT `.cuh` 经 `tools/audit_jit_resources.py` 逐字节和模式核对，全部为 `identical_owner_only`，没有新增编译或数值证据。

### 本轮实际验证

所有命令均经 SSH 在 NMZ_zz_26 执行；容器为 `yangyn_sglang_plugin_1022`，容器 Python 3.10.12，官方源码模式使用 `PYTHONPATH="$OFFICIAL/python:$PLUGIN"` 和 `SGLANG_PLUGINS=hcu`。宿主解释器为 `/opt/saltstack/salt/bin/python3.10`（3.10.19）。日志保存在 `artifacts/kimi-runtime-closure-host-20261105/`；该目录不是 Git 跟踪证据，长期交接需保留对应路径。

| 命令/检查 | 实际结果 |
|---|---|
| 宿主 `python3.10 -m unittest discover -s tests -v` | 136 项、123 通过、13 skipped，9.285 秒，退出 0；日志 `unittest-final.log` |
| 容器 `python3 tests/runtime/flashattention_interface_cpu.py` | 2 项通过，12.679 秒；`flashattention.log` |
| 容器 `python3 tests/runtime/glm_muse_dependencies_cpu.py` | 5 项通过，24.727 秒；`glm-muse-cpu.log` |
| 容器 `python3 tests/runtime/muse_media_config_cpu.py` | 1 项通过，41.727 秒；`muse-media-config.log` |
| 容器设 `SGLANG_USE_LIGHTOP_CHANNEL_FP8=1` 后运行 `tests/runtime/compressed_fp8.py` | 14 项通过，1.216 秒；`compressed-fp8.log`；缺省 opt-in 条件下的失败不记为代码回归 |
| 真实 register/apply 后导入 GLM text/vision/NextN、Muse text/omni/processor | 6/6 `IMPORT_OK`；`owned-model-imports.log`。导入期 CUDA 已初始化，但没有运行 GPU kernel |
| `tools/validate_plugin_runtime.py` 使用最新目标合同 | **600 registered / 600 applied / 0 unapplied / 0 errors**，4 个具名 seam；`strict-loader.json` |
| `tools/audit_jit_resources.py` | Muse 8/8 资源来源字节和模式一致；`jit-resources.json` |
| `tools/build_migration_execution.py` 及 `--check` | 重建与快照一致，退出 0 |

未验证：全部 GPU kernel/Graph、完整模型服务、真实 CP/TP/PD collective、GLM/Muse 数值与性能、可选 Helion 执行、媒体数据执行、wheel 安装、浏览器视觉验收。其他 pending/in_progress 条目保持原状态；未 commit、未 push，未修改 DAS 或官方来源树。

结构收尾时执行 `tools/build_guide.py` 及 `--check`、执行台账 `--check`、`ci/l0_gate.py` 和 `git diff --check`，均退出 0；L0 为 620 个 Python 文件、8 个 Skill、0 errors。导览已重建并通过快照检查；该结果不是浏览器截图或交互验收。原始输出见 `artifacts/kimi-runtime-closure-host-20261105/structure-final.log`。


## 2026-11-05 KV CPU 缺口补齐（进行中）

本轮接手 KV/PD/distributed/native 共 104 条；其中本范围 CPU-first 为 19 条。新增 [CPU 方法实现](../hcu_sglang/mem_cache/cpu_cache_methods.py) 与 [窄 Hook](../hcu_sglang/mem_cache/cpu_cache.py)，由 allocation 注册入口接入。当前抽取 token-major Mamba 共享 entry 偏移、超大多模态条目拒绝、HF3FS/SeaweedFS Mamba TP 键、TensorCast packed 检查、StorageAttachment 双窗口阈值/清理及 Mamba external-linker 生命周期，保留官方类身份和非 HCU original 回退。来源定义与文件 SHA 见 [独立合同](../tests/contracts/cpu_cache_sources.json)。此处为进行中记录，尚未完成本轮验证；不记 verified，也不据此关闭其余条目。


## 2026-11-05 CPU 缓存许可、来源登记与文件头门禁

按本轮合规审查，给 [CPU 方法副本](../hcu_sglang/mem_cache/cpu_cache_methods.py) 原样恢复 Tensorcast 的 SPDX 与 SGLang 版权两行，并补入 [第三方清单](../THIRD_PARTY_NOTICES.md)。方法正文未改动；[逐定义合同](../tests/contracts/cpu_cache_sources.json) 继续固定 20 个来源定义，不把局部抽取描述为整文件等价。

[运行时来源合同](../tests/contracts/runtime_owned_sources.json) 新增 extracted_files 与 adapter_files：前者固定 CPU 方法目标 SHA、七个来源文件 SHA 并引用逐定义合同；后者登记 CPU Hook、模型 registry 和五个 entries 文件的目标 SHA、所接来源 SHA，明确为本仓原创接线，不伪装成来源整文件副本。原 files 的可逆差异语义保留。两份来源合同同时保存来源原始头部行供离线检查。

[L0](../ci/l0_gate.py) 按来源注册表检查：来源路径、SHA256、原始头部与 source_declaration 必须齐全；有声明时逐字保留，只有一种声明时保留该声明；无声明时明确登记“来源无声明”，不补造归属。抽取副本逐来源登记，原创接线仅核验来源记录，不强制复制来源头。新增 [门禁回归](../tests/test_l0_migration_headers.py) 覆盖无声明、部分声明、元数据缺失、正文伪造头部、替换原版权、空文件绕过和原创适配边界；扩展 [来源测试](../tests/test_runtime_owned_sources.py) 核验新登记与去重。

本轮验证结果随后追加。当前仍未完成 StorageAttachment 初始化、Mamba 正式 linker 接线和其余 104 条审查；本节许可修复不改变运行时完成状态。未执行 GPU、完整模型、性能、wheel 或浏览器视觉验收；不删除快照，不 commit/push。


许可门禁首轮实际执行（宿主 `/opt/saltstack/salt/bin/python3.10`，Python 3.10.19）：`ci/l0_gate.py` 返回 1，649 个 Python、8 个 Skill，新增严格规则报告 60 个文件的 116 条 SPDX/版权缺项。逐文件回读固定 DAS 来源并核对 SHA，60/60 匹配；56 份来源本身两项皆无，4 份只有一种声明。这些不是本轮新增认定的“丢失来源声明”。未为通过门禁而改造这些来源副本；需要区分“完整保留原有声明”和“来源本身补齐声明”的合规处理。

首轮全仓单测 150 项，失败 2、错误 2、跳过 13，10.351 秒，退出 1。三处整文件来源合同问题已交回对应负责人；本轮新增测试对量化逐定义合同的格式兼容随后修正，原始失败日志保留。聚焦门禁回归 9/9、CPU 来源合同 3/3 通过；不以此代替全仓通过。原始命令、返回码、日志与来源头核对分别保存在 `artifacts/license-gate-20261105/results.json`、`focused-results.json`、`source-header-audit.json` 及同目录日志。


来源规则最终收敛为 Chief 确认的“有声明逐字保留、无声明登记但不补造”。回读两份合同全部来源并核对固定 SHA 后，在每条 source/source_sha256 旁增加 source_header_lines 与 source_declaration（“来源无声明”或“保留来源声明”）；extracted_files 按每个来源分别登记，adapter_files 只做溯源，不声称源头部应复制到原创文件。

按最终规则首次复验：`/opt/saltstack/salt/bin/python3.10 ci/l0_gate.py` 返回 0，652 个 Python、8 个 Skill、0 errors；`/opt/saltstack/salt/bin/python3.10 -m unittest discover -s tests -v` 执行 157 项、142 通过、2 失败、13 skipped，10.123 秒，返回 1。两项失败均为 GLM kpool/planner 目标 SHA 与合同不同，已交原负责人核对；文件头规则 14 项全部通过。最终规则日志及命令返回码见 `artifacts/license-gate-20261105/source-aware-results.json`、`l0-source-aware.log`、`unittest-source-aware.log`；首轮严格规则失败保留为历史证据，不覆盖。


## 2026-11-05 KV 等待状态与 Mamba linker 构造入口（进行中）

在既有 [CPU Hook](../hcu_sglang/mem_cache/cpu_cache.py) 中增加 StorageAttachment.__init__ 的 AROUND 适配：先调用官方 original，再仅在 HCU 且 cache 缺字段时创建 _storage_prefetch_wait；已有等待状态不覆盖，非 HCU 不新增字段。另将 DAS UnifiedCacheLinkerWrapper.__init__ 原文抽入 [方法副本](../hcu_sglang/mem_cache/cpu_cache_methods.py)，通过正式 AROUND target 接入；只在私有 globals 字典中将 supported components 增加 MAMBA，避免污染 native 全局支持集合与非 HCU original 回退。目标列表和 [逐定义合同](../tests/contracts/cpu_cache_sources.json) 同步扩展为 21 个原文定义、9 个组，来源注册表追加 linker 来源与最新目标 SHA。

[CPU 行为测试](../tests/runtime/cpu_cache_cpu.py) 增加等待状态初始化/保留、Mamba 构造接纳、其他组件拒绝及 HCU 调用后非 HCU 仍拒绝 Mamba。新增构造测试使用显式 namespace/cache fixture，并非完整 linker 模块导入；原来的两个 leaf 模块仍经真实 HookRegistry 应用。完整 storage/Mamba 的禁 GPU 导入仍有既知依赖阻塞，未扩大 mock 绕过。完整 linker 的 match/load_back、host-prefetch 与错误协商方法尚未迁入，不能据本节将该来源整文件标记 implemented/verified，也不能关闭全部 104 条。

本节代码的实际复验结果随后追加；未运行 GPU、模型服务或性能测试，未 commit/push。


断网前构造入口版本的实际结果已落盘：容器禁 GPU 执行 `python3 tests/runtime/cpu_cache_cpu.py`，11 项、0.036 秒、全部通过，退出 0；包括 GPU 未初始化断言。宿主执行 `python3.10 -m unittest discover -s tests -v`，157 项、143 通过、1 失败、13 skipped，10.080 秒，退出 1；剩余失败为 GLM kpool/planner 反向恢复来源 SHA，交原负责人修正。完整命令与返回码见 `artifacts/kv-cpu-20261105/constructor-results.json`，对应 `constructor-runtime.log` 与 `constructor-unittest.log`。随后的 SSH 超时退出 255，未执行远端读写；节点恢复后继续复验，不用断网前失败日志覆盖后续结果。


节点恢复后复验构造入口版本：宿主 `/opt/saltstack/salt/bin/python3.10 ci/l0_gate.py` 返回 0，652 个 Python、8 个 Skill、0 errors，72.999 秒；同解释器执行 `-m unittest discover -s tests -v`，157 项、144 通过、13 skipped，13.654 秒，返回 0。GLM planner 的旧来源合同失败已由原负责人修复；日志和精确命令见 `artifacts/kv-cpu-20261105/resume-gate-results.json`、`resume-l0.log`、`resume-unittest.log`。这是新增 CP 登记前的结果，后续改动另行复验。

## 2026-11-05 CP LayerSplit 四份完整文件审查

回读固定 DAS HEAD，核对 CP LayerSplit 包入口、deepseek_v4_layout、pool_base、transfer 四份现有 owned 文件。逐字节反转声明的 import 差异后均恢复来源 SHA，计算正文没有其他变化；来源没有声明时仅登记“来源无声明”。完整映射和静态调用方见 [本批审查](../plans/hcu-kv-cpu-review-20261105.json) 与 [来源合同](../tests/contracts/runtime_owned_sources.json)。V4 sizing 和 CP pool 调用 layout、CP pool 继承 base；公开 setup-state 适配及 owned protocol/Mooncake 使用 transfer。

本批只将上述四个文件处置为 implemented，不标记 validated/verified。prefill/decode 仍有两处 Source transfer import，保留为独立 PD 接线缺口；CPU 数值、真实 CP/PD、多进程、GPU、模型和性能未验证。104 条任务仍在进行，未 commit/push。


CPU-first 逐文件复核新增六条 implemented：token_major、multimodal_cache、HF3FS、SeaweedFS、TensorCast、StorageAttachment。逐项比较固定官方与 DAS 的全部叶子定义，实际变更定义集合与 CPU 抽取合同逐项相等；同时人工审查完整文本 diff，记录额外类型导入和构造状态。六条使用 allocation → cpu_cache → 公开方法 Hook，保持 native 类身份。StorageAttachment 连接现有 window_cache_init/window_pipeline 的双窗口入口；完整 linker 仍是单独缺口。对应来源/官方 SHA、变更定义和适用验证边界追加到 [本批审查](../plans/hcu-kv-cpu-review-20261105.json)。此批已登记十条文件实现（四个完整副本、六个完整差异审查），均未标记运行时等价；104 条中其余 94 条仍未因本批审查而关闭。


### 2026-11-05 KV 预算公式核对与 CPU adapter H1

本批在新增 10 条 implemented 登记后补跑检查；这些登记不代表 validated 或 verified，其余 94 条仍在迁入审查。`hcu_sglang/mem_cache/cpu_cache.py` 为本轮原创接线，按 H1 补 HYGON Copyright 与 Apache-2.0 SPDX，并更新来源合同 adapter SHA；抽取方法的来源声明与来源 SHA 不变。

固定官方 `baa9befa381511cd4d56f83980951e5a45fffc6d` 与 DAS `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e` 核对：`KVCacheConfigurator._profile_available_bytes`、`_resolve_memory_pool_config`、`compute_post_capture_kv_resize` 三个函数逐字相同。普通非 GLM/非 MLA 的 MHA cell-size 在 4 组合成 CPU 输入下相同；无 spec、无 replay、无 fused draft 的 Mamba 预算在 54 组合成输入中 51 组相同。另 3 组为关闭 radix cache 且 max_running_requests=1、attn_dp_size=2：官方得 0 槽后 RuntimeError，DAS 以 max(1, ...) 得 1 槽。该边界差异不是本轮 A/B 已命中的证据。探针首次缺 get_memory 夹具，第二次触发该真实边界异常，最终探针记录两侧返回或异常并断言差异仅发生于此边界。

证据：`artifacts/kv-cpu-20261105/budget-formula-review.json`。本次是从固定源码抽取函数并注入显式 namespace 的 CPU 公式探针，不是完整模块导入、真实 Hook、实际显存采样或模型验证。实际 A/B 日志及输入预算仍待核对，max_total_num_tokens 差 0.24% 的原因未确定；不据文件级函数计数推定漏迁。不运行 GPU，不 commit/push。


### 2026-11-05 TreeComponent 完整差异审查

`unified_cache/components/base.py` 全部差异已审查：`refresh_lru` 的 match 改 if/elif 在 WALKDOWN 空/非空、MATCH_END 空/非空、INSERT_END、非法 phase 六组显式 CPU namespace 夹具下返回/调用/异常相同。`create_buffer_pipeline` 反转 owned pipeline import 后 AST 相同，已有正式 AROUND 与 `init_hicache` 主/辅助 SWA consumer 接通；新增 SWAComponent 仅 TYPE_CHECKING。本条从 in_progress 改 implemented，累计本批 11 条、余 93 条仍未闭合，不晋升 validated/verified。证据存 `plans/hcu-kv-cpu-review-20261105.json`。未运行完整 controller/storage、PD/GPU、模型或性能验证。

前一检查批：execution build/check、guide build/check、git diff --check 与 L0 退出 0；157 项全仓单测失败（13 skip，3 个失败断言，涉及 GLM dsa_backend.py/kpool/indexer.py 目标 SHA 漂移），已交对应负责人核对，不能记全仓通过。日志：`artifacts/kv-cpu-20261105/review-gate-results.json` 与 `review-unittest.log`。


### 2026-11-05 A/B 容量差的实际预算证据

只读核对仓外 `test-results/runs/dump-20261105` 的 A/B server 日志与 server_info。TP0 加载前可用显存 A=135.50 GB、B=135.64 GB；加载后 A=120.42 GB、B=120.71 GB；日志权重占用 A=15.08 GB、B=14.93 GB。mem_fraction_static 同为 0.842461484375，按已确认相同的预算采样公式，日志四舍五入数推算预算增量约 0.268 GB。Mamba 槽 A=652、B=654，SSM 日志 45.91→46.05 GB；KV token 3418432→3426496，内存 telemetry 52.162→52.285 GB。依据后者估算每 token 16384 bytes 时，8064 token 对应 0.123046875 GiB，和 Mamba 增量合起来与预算增量相容。该 16384 值由四舍五入 telemetry 推算，尚非实际运行时 cell_size 采样。

本配方 radix 启用、attn_dp_size=1、spec/replay 关闭，因此 CPU 公式探针发现的小请求边界不适用。现有证据支持先核预算输入/驻留显存，不能仅凭 0.236% 容量差定性每 token 布局或构造漏迁。未采集实际未舍入 free memory/预算/每槽字节，未验证该差异与 logprob 偏差的因果关系。完整字段、SHA 和原日志行号：`artifacts/kv-cpu-20261105/ab-memory-budget-evidence.json`。

### 2026-11-05 LRU 夹具证据更正

复查发现前批 refresh_lru namespace 夹具误把 phase 与 node 的位置传反，六组旧输出全部落入非法 phase，不能支持正常分支验证；撤回该批六组行为通过结论。已保留 superseded_lru_fixture_outputs 并按真实签名 (self, phase, node, root_node) 重跑：WALKDOWN 空值无调用、非空仅 reset_node_mru；MATCH_END 空/非空均 reset_node_and_parents_mru，断言 node/root/predicate 身份；INSERT_END 无调用；非法 phase 仅 ValueError。六组官方与 DAS 结果一致且符合独立预期。证据：artifacts/kv-cpu-20261105/lru-corrected-fixture-results.json 与 plans/hcu-kv-cpu-review-20261105.json。范围仍限固定源码方法与显式 namespace，真实 Hook/完整模块/GPU/模型未验证；implemented 状态不变。


### 2026-11-05 page-plan 原文恢复与 CPU 证据边界

固定 DAS HEAD 的 layer_split.py 仅含 MainKVPagePlan、_build_unique_physical_pages、build_main_kv_page_plan 三个定义。本轮在 [既有 owned 模块](../hcu_sglang/mem_cache/dsa_cache_layer_split.py) 恢复来源 docstring 与完整声明，三个定义含 decorator 逐字相同；未修改 page-plan 算法。来源 SHA256 为 1c253a2104e15179f8585d35c17f70c3bcc41cabaf1dad041d9aa61ea23efa30。在 [定义合同](../tests/contracts/runtime_reviewed_definitions.json) 增加第九组，并在 [来源合同](../tests/contracts/runtime_owned_sources.json) 明确仅覆盖这三个定义；其余 LayerSplit pool 不据此关闭。正式 dsa 注册、DSA utils 与 GLM CP 调用链已静态审查。[本批审查](../plans/hcu-kv-cpu-review-20261105.json) 和 [迁移处置](../plans/sglang-das-decisions.json) 登记本条 implemented，累计 12 条，104 条中其余 92 条未因本批工作闭合，不晋升 validated/verified。

容器 Python 3.10.12、Torch 2.11.0，HIP/ROCR/CUDA_VISIBLE_DEVICES 均为 -1。完整模块测试 dsa_plan_fix 尚未加载：首次导入触发 GPU 属性探测，加入显式硬件夹具后仍在 fp8_utils 的 capability 比较报 TypeError；两次均退出 1，不能计通过。改用固定源码显式 namespace，来源与 owned 各 7 例：空批、history/current 去重与 dummy 过滤、重复 request、非法 page、负 length、请求行不足、表容量越界；包含 dataclass 对象身份断言。合计 14 例通过，进程退出 0、4.217 秒，CUDA 未初始化。日志及脚本为 artifacts/kv-cpu-20261105/page-plan-mha-cpu-diagnostic*；完整导入失败日志 page-plan-runtime* 保留。此结果不代表真实模块/Hook/CP consumer 执行。

同一 namespace 探针核对 MHA.set_kv_buffer：DAS 有 FP8 clamp，插件继承官方方法而在 _store_kv_layer 实现 HCU page 布局。首次探针预期官方将越界值写成 NaN，但该断言失败（退出 1、4.855 秒）；复核实测 Torch 2.11.0 CPU 的直接 cast 已将 +/-1000 饱和为 +/-448，两侧写入字节均为 [126,254]、NaN 数为 0。因此未复现 CPU 数值差异；仅登记静态 clamp 差异和 HCU/真实模型范围未验证，不记实证缺陷，不作为 r04 归因。失败日志 page-plan-mha-cpu-probe* 与新诊断产物分别保留，诊断中的 runner import 与官方 storage callback 均为显式夹具。

未验证：完整模块/正式 Hook 的 page-plan CPU 用例、真实 CP/PD、多进程、GPU、模型数值与性能、wheel、导览浏览器渲染/交互/窄屏。未改 MHA 算法，未启动服务，未 commit/push。容量差按已确认口径归档：A 权重驻留后可用显存少 0.29 GB，KV 少 8064 token，Mamba 槽 652 对 654，原因未定位，不视为已对齐。本轮门禁结果随后追加。


page-plan 批实际门禁（宿主 Python 3.10.19）：execution build/check、guide build/check、git diff --check 全部退出 0；L0 653 个 Python、8 个 Skill、0 errors，23.294 秒；全仓 unittest 157 项、144 通过、13 skipped，测试 10.134 秒、进程 10.293 秒，退出 0。此前 GLM 合同三处失败本轮未再出现，不覆盖旧失败记录。命令、返回码和合同前后 SHA 见 artifacts/kv-cpu-20261105/page-plan-gate-results.json；此结果是下面两份来源合同登记前的快照。

### 2026-11-05 cleanup 与 CP utils 完整副本审查

[cleanup](../hcu_sglang/mem_cache/cleanup.py) 与 [CP utils](../hcu_sglang/mem_cache/cp_cache_layer_split/utils.py) 均与固定 DAS 来源逐字节相同，无反向替换；SHA256 分别为 a3a347d73b6cdf1e48f4ef446d6b7ec51f14f17afac969d4abdf42731c3c0b3c、ebc80c484f4abbb729a3c32853e14c63b38e1b0c14689a72f28ac8a1679ecec8。两来源无声明，只在 [来源合同](../tests/contracts/runtime_owned_sources.json) 登记“来源无声明”，不补造归属。现有 common/PD prefill 与 CP pool_base/layout 调用方已审查；PD 正式注册路径为顶层 disaggregation → disagg.integration → prefill.register_hooks。回读共享处置表发现这两条原已 implemented，因此本轮仅增强审查/来源合同和 CPU 证据，不重复算状态晋升。本批审查共覆盖 14 条 implemented，104 条中另 90 条未因本批审查闭合；均不晋升 validated/verified。

新增原创 [CPU 回归](../tests/runtime/cleanup_cp_utils_cpu.py)，按 H1 添加版权与 SPDX。容器源码模式、SGLANG_PLUGINS 为空、三组 GPU 可见开关均 -1，真实导入 owned cleanup、CP utils 与 pool_base；9 项测试通过，测试 0.055 秒、进程 9.397 秒，退出 0，CUDA 未初始化。覆盖完成结果对象身份及只释放一次、未知副作用隔离、显式无副作用失败重试、非 abort 回退、重入释放隔离、独立子步骤，以及 0..20 层/1..8 分片的连续不重叠覆盖、宽度均衡、owner 映射、真实 pool_base 全局偏移与非法输入；配置开关测试明确使用 parallel-state fixture。完整命令与日志：artifacts/kv-cpu-20261105/cleanup-cp-utils-runtime-results.json、cleanup-cp-utils-runtime.log。

生产运行时代码本节未改；仅增加来源审查/合同与测试。真实 request allocator、副作用失败恢复、完整插件注册、scheduler/PD 通信、CP 多进程、GPU、模型数值与性能仍未验证。新增登记后的最终门禁另行追加；未 commit/push。


### 2026-11-05 cleanup/CP 合同门禁与断联后文档收尾

cleanup/CP 来源合同登记后的断网前门禁已核对原始 JSON 与日志：宿主 `/opt/saltstack/salt/bin/python3.10` 执行 execution build/check、guide build/check、`git diff --check` 均退出 0；L0 为 654 个 Python、8 个 Skill、0 errors，进程 19.669 秒；全仓 unittest 157 项、144 通过、13 skipped，测试 10.021 秒、进程 10.178 秒，退出 0。四份跟踪输入（runtime_owned_sources、runtime_reviewed_definitions、cleanup_cp_utils_cpu、dsa_cache_layer_split）的门禁前后 SHA 完全一致。完整命令、返回码、输入 SHA 与日志索引见 `artifacts/kv-cpu-20261105/helpers-gate-results.json`，原始输出为同目录 `helpers-l0.log`、`helpers-unittest.log`。上述结果产生于断网前，不代表恢复后新代码已通过；旧失败与探针预期错误的日志保留，不覆盖。

原定最后一段开发记录追加在 SSH 连接阶段超时，退出 255，远端脚本未执行，追加后的检查当时未运行。2026-11-05 16:02 +08:00 恢复后重新读取本记录并独立执行 `ls -l`、`sha256sum`，五份已报告证据均存在，大小、SHA、修改时间与命令原始输出写入 `artifacts/kv-cpu-20261105/evidence-path-verification-20261105T1602.json`。此前群报告的远端相对路径被 Markdown 按本机工作目录解析，所给本机链接无效；后续群报告统一用代码块提供已核验的远端绝对路径。

本次只追加开发记录并重建导览；追加后的 guide build/check、execution check、diff check 与 L0 按实际返回码独立保存至 `artifacts/kv-cpu-20261105/closing-doc-gate-results.json` 和同目录 `closing-*.log`，不复用断网前通过结果。恢复后全仓单测由并行负责人另行复验，本次文档收尾不声称其通过。未验证浏览器渲染、交互和窄屏布局；运行时/GPU/模型验证边界维持前文，未 commit/push。


### 2026-11-05 阶段一 closure 生成与防漂移门禁

新增 tools/build_runtime_closure.py，将 triage 的 257 个文件与当前 decisions 合并为每文件一行的 closure；未登记默认 pending，分类旧状态单列保留。输出保存完整 decision、实现/测试/证据字段及规范化输入摘要。ci/l0_gate.py 使用当前解释器调用 --check；decisions 变化而未重建时退出非零，包括范围外 decision 内容变化。--check 不写文件，禁止输出覆盖输入。新工具与 tests/test_runtime_closure.py 按 H1 添加原创版权和 SPDX。

原 closure 为 307 符号行、72 来源文件，已逐字节保留到 plans/hcu-runtime-target-review-20261105.csv，不将符号数混算为 257 文件进度；历史刷新记录保留原 target_table 并增加保留后路径。旧 decision 的符号证据引用按 docs/migration.md 的说明读取历史表。本轮不改共享 decisions 的状态或来源证据，不据生成快照宣称新的运行时实现或数值等价。

宿主 Python 3.10.19 首轮 build/check 退出 0；257 文件当前为 implemented 58、in_progress 12、pending 172、upstream 4、separate_component 11。聚焦 unittest 8 项通过，测试 0.323 秒；包含临时目录中修改 decision 后 --check/L0 检查失败、重建后恢复、检查不写产物及输入保护。日志为 artifacts/runtime-closure-20261105/focused-results.json 和 focused.log。文档与最终全仓门禁另存本目录 final-gate-results.json；完成后的实际汇总追加于后。

工具不导入运行时，不跑 GPU；完整 attention/KV、模型精度/性能、wheel、网页浏览器渲染和交互均未因本工具验证。不 commit/push。


closure 最终全仓门禁实绩（2026-11-05 16:16 +08:00，宿主 Python 3.10.19）：execution build/check、closure build/check、guide build/check、git diff --check 全部退出 0；L0 656 个 Python、8 个 Skill、0 errors，43.806 秒；全仓 unittest 165 项，152 通过、13 skipped，测试 10.471 秒、进程 10.631 秒，退出 0。七份关键输入（decisions、triage、生成器、回归、L0、两份来源/定义合同）前后 SHA 一致。历史符号表 SHA256 402683cd6128d9b9ca245fe7bd3da109f300b90e18b5b032830f8ac9ff065be2，与保留前原表一致。逐命令记录 final-gate-results.json、final-l0.log、final-unittest.log 位于 artifacts/runtime-closure-20261105/。本段追加后重新 build/check 导览、closure/execution check、diff check 和 L0，独立留存于 doc-close-results.json；不以此前结果代替追加后的检查。


### 2026-11-05 memory_pool 81 项定义与注册链审查

固定官方/DAS memory_pool.py 对比，函数叶子定义为 33 changed + 48 added = 81，不将包含方法的 class 节点重复计入。逐项来源行号/AST摘要、owned候选、目标位置与已审查注册链写入 artifacts/memory-pool-review-20261105/definition-review.json 及 .csv。38 项找到对应 owned 目标及静态接线（Mamba/MLA/DSA 33、MHA builder 替换子类 5），其余 43 项保留继承差异、API缺项或接线待核；10 项完整 AST 匹配。该映射仅审查，不代表执行通过，不改 decisions，也不关闭 memory_pool 整文件。

Mamba constructor、_should_fuse_slot_ops、conv_window_dedup_enabled 三定义与来源完整AST相同；allocation → mamba_pool.register_hooks → HookRegistry AROUND 链已审查，HCU门控下用 FunctionType 重绑 native globals。MHA 仍继承官方 constructor/set_kv_buffer，并由子类 _store_kv_layer 负责 HCU page 布局；DAS显式 FP8 clamp、capture/alt_stream调度、prefix-valid默认分支及metadata API差异保留。extra buffer stride 的来源 stride(0) 与 owned shape乘积差异尚未行为验证。Hybrid root 的23个差异定义未在已审查链确认相应constructor/代理安装；子pool同名方法不能当root接线。DSA _should_allocate_index_layer 来源增加@property但仍要求参数，作为静态签名差异保留，未擅自复制或修复。GLM/DSA、MiniMax和当前Qwen3.5普通MHA的可达性分别保留，不混作容量归因。

2026-11-05 16:29 +08:00 容器源码模式，SGLANG_PLUGINS为空、HIP/ROCR/CUDA_VISIBLE_DEVICES均为-1；只尝试导入真实memory_pool、owned Mamba/MHA。进程8.935秒、退出1：官方fp8_kernel导入时 is_gfx1250_supported 调用 torch.cuda.get_device_properties，报 No CUDA GPUs are available，尚未执行到池构造或Hook。未通过逐层mock将失败改称通过。命令与完整错误为 import-probe-results.json、import-probe.log，脚本 import_probe.py 均位于上述证据目录。

容量差仍为原因未定位：A/B token 3418432/3426496、Mamba槽652/654；缺少未舍入预算、每槽字节和真实运行时构造采样，不视为已对齐，不据AST相同排除构造问题，也不据API缺项解释数值差。未修改生产运行时代码、未跑GPU、未启动服务、未commit/push；本轮文档追加后的guide build/check、closure/execution check、diff check、L0实绩独立保存到同目录 review-doc-gate-results.json，不复用前批结果。


### 2026-11-05 memory_pool 剩余43项静态接线复核

在第一轮81项叶子定义中精确筛选43项，独立保存 artifacts/memory-pool-review-20261105/wiring-review.json 与 wiring-review.csv，不覆盖第一轮证据。逐项记录DAS源码及decorator、官方定义/基类、owned同名候选、调用上下文、guard和触发配置。扫描489个owned Python文件、673个动态写入候选，输入逐文件留SHA；写入前重新核对。首次生成因并发修改flashattention_interface.py被摘要检查拦截，退出1且未写产物；重新采样后全部输入稳定，生成退出0。43项来源AST摘要均与第一轮一致，CSV/JSON定义集合一致。

官方模块中32项缺对应定义，11项保留官方实现差异；已审查入口、Mamba/MLA/DSA注册与MHA builder未见这43项对应覆盖。此为静态覆盖清单，不能计为43个实证缺陷。23项属于Hybrid root：builder只替换full-attention child，child的LayerSplit方法不自动变成root代理。prefill与dsa_utils部分调用用getattr默认值回退，可能跳过分片/compact；GLM backend的PAGED位置转换、HCU/KPool写入与tail调用则有直接API调用，需该backend与实际Hybrid root组合才触发，尚未执行验证。普通非GLM get_kv_layer_ids分支仍与DAS相同。

MHA继承官方setter而覆盖_store_kv_layer，已有HCU页布局接线；DAS clamp、非HCU-FA的NHD scaled-FP8 writer和prefix-valid默认分支分别保留差异，不混作当前配方归因。owned prepare_hicache_staging虽直接读取get_device_pool_infos，但全插件仅找到定义；cache_retention只接build_kv_cache，官方prepare入口继续使用旧pool协议，因此尚不能据缺provider判定当前启动必失败。MXFP8、MLA、MiniMax与当前普通MHA配置分别登记。DSA来源@property仍带local_layer_idx参数，官方保持可调用method、owned donor仍传参；只记录来源签名差异，不盲复制。

未改生产运行时代码、decisions或已有证据，不晋升validated/verified，不关闭memory_pool整文件或104条范围。禁GPU导入阻塞沿用前批失败记录，本轮不重试；实际Hook、CPU构造、GPU/模型数值与性能、CP/PD多进程均未验证。652/654槽和8064token差仍原因未定位，等待运行时dump。文档门禁按实际命令及返回码独立留存同目录 wiring-doc-gate-results.json 与 wiring-*.log；本轮只新增证据和文档，不重跑全仓单测，不以16:16的165项/13skip代替本批验证。

## 2026-11-05 A 侧 FA wrapper 对齐与单卡冒烟

按“以 DAS 为基准”的口径对齐插件 `flashattention_interface`：decode 分支改为与 fork `L2577` 相同的 vendor `vllm_flash_attn_with_kvcache`（11 个参数：q/k_cache/v_cache/block_table/cache_seqlens/softmax_scale/causal/window_size/softcap/return_softmax_lse/num_splits，不传 layout/descale/kv_cache_dtype）；extend 分支保留 `vllm_flash_attn_varlen_func`，`fa_version=2`（与 fork 硬编码一致），并透传 `q_descale/k_descale/v_descale/softcap/s_aux/seqused_k/block_table`。`layout` 与 descale 在 fork wrapper 中只在 triton 分支转发，非 triton 的 vendor 调用不含它们，因此 A 不传与 B 一致。

单卡冒烟（卡 1，`Qwen3-0.6B`，`--tp-size 1 --page-size 64 --mem-fraction-static 0.08 --attention-backend fa3 --disable-cuda-graph`，rust-1.92 env 且 `PATH` 追加 `/opt/dtk/bin`）：17:25:02 `The server is fired up and ready to roll!`，`POST /generate` 返回 200；随后停止，端口 30091 释放。extend（vllm varlen）与 decode（vllm with_kvcache）都在真实请求中执行，无 TypeError。对照：默认 page size 会以 `RuntimeError: Paged KV cache block size must be divisible by 64 at gfx936 platform` 退出，该约束已写入 [安装文档](installation.md#41-gfx936-上的-page-size-约束)。

冒烟日志中的 `Failed to apply hooks to sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store._validate_storage_config` 已定位为容器缺可选依赖 `tensorcast`（`import tensorcast` → `ModuleNotFoundError`），HookRegistry 将导入失败表现为父包缺属性；官方 0.5.22 存在该 package，目标串本身正确，由 gpt61 按“缺依赖跳过可选 Hook”处理。

未验证：27B 模型下 r01/r04 的重跑与 dump 对拍（卡 0 仍被他方冻结任务占用）；`_trim_padded_q_for_batch_api`/`_restore_padded_q_rows`（SP padding 裁剪，本配方 no-op）未迁入；量化 static/TENSOR 与 `w8a8_strategy` 2/3/4 路径未验证。未 commit/push。

### 2026-11-05 SWA 默认属性、PD 接线与 Tensorcast 可选依赖

本批修复三处接线，并完成两个 CPU-first 文件的处置登记。所有改动只影响 HCU 门控路径或缺失可选依赖时的注册行为。

`hcu_sglang/mem_cache/swa.py` 在 HCU 门控内为 `BaseSWAKVPool` 设置 `request_window = None` 类属性默认值，不替换类对象、方法或抽象方法集合，已有值不覆盖。该默认值对应 DAS 相对固定官方文件唯一的差异：DAS 只增加这个可选类属性，没有可调用定义变化。五个容器内 CPU 用例覆盖既有实例获得属性、类与方法身份不变、非 HCU 基线不变、已有适配值不被覆盖、子类与实例同名属性保持私有、CUDA 未初始化。

`hcu_sglang/disagg/prefill.py` 与 `decode.py` 的 `configure_v4_transfer` 改从 owned `hcu_sglang.mem_cache.cp_cache_layer_split.transfer` 导入，和同段 `DeepSeekV4TokenToKVPool` 的 owned 导入一致。此前两处仍指向 Source 命名空间，属于插件内消费者未用 owned 实现的真实接线缺陷；两文件各自的 33/27 个未覆盖定义仍未完成，处置保持 pending。

`hcu_sglang/mem_cache/cpu_cache.py` 的 Tensorcast 可选钩子在 `find_spec("tensorcast")` 为 `None` 时跳过注册，目标串保持 `sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store._validate_storage_config` 不变。原因已在容器内实测：官方树有该模块，`import` 报 `ModuleNotFoundError: No module named 'tensorcast'`，HookRegistry 随后把它表现为父包缺少属性。只跳过缺依赖，不吞其它探测异常；三项 CPU 回归覆盖缺依赖不注册、依赖可用时仍注册同一公开目标、安装损坏时抛错。

`unified_cache/components/mamba.py` 完成整文件差异审查并登记 implemented：5 个 changed + 2 个 added 叶子定义，六个 DAS 定义逐字抽取并经公开 AROUND 目标接线，`refresh_lru` 的 match 到 if/elif 改写经七组官方/DAS 显式 namespace 夹具逐项相同。triage 记 9 个 uncovered 是含模块级语句的计数，顶层函数与类方法审计为 7，未按 9 凑数。`cpu_cache_cpu.py` 另补两项包装回归：非 HCU 回退保留当前有效原函数、辅助方法在原构造器前安装且重复注册不改变身份。

同一轮静态审查登记五项待迁入缺口：`hicache_key_scheme.py`（fork-only 8 个新增定义，无 owned 副本）、`linker_fault_injection.py`（fork-only 4 个定义，测试用途）、`hybrid_cache/linker_pool_assembler.py`（2 changed + 4 added，V4.1 与 Mamba/SWA direct-linker 组构造缺失）、`storage/mooncake_store/mooncake_store.py`（6 changed + 2 added，含 `layout_hcu` 页断言）、`unified_cache/unified_tree_core_interface.py`（两个抽象方法增加可选 `component_type` 参数，owned window 实现已带同名默认值，未复制 ABC 声明）。七项完整差异与逐符号证据在 `artifacts/kv-cpu-20261105/cpu-followup-review.json`。

验证边界：`tests/runtime/swa_defaults_cpu.py` 5 项、`tests/runtime/cpu_cache_cpu.py` 16 项在容器源码模式下通过，GPU 可见开关均为 -1；`tests/test_pd_transfer_namespace.py` 1 项提供静态导入合同。没有执行真实 Hook apply（Tensorcast 依赖未安装）、V4.1 RequestWindow 构造与 encoder replay、unified_cache 控制器、storage 后端、PD 多进程、GPU、模型数值与性能；完整 `components.base` 导入仍被 fp8_kernel 的设备探测阻塞。本批不 commit、不 push。

### 2026-11-05 mooncake_store 迁入与容量差归档

`storage/mooncake_store/mooncake_store.py` 的 8 个变更/新增叶子中，7 个已逐字抽取到 [cpu_cache_methods](../hcu_sglang/mem_cache/cpu_cache_methods.py) 并经 [cpu_cache](../hcu_sglang/mem_cache/cpu_cache.py) 的公开 AROUND 目标接线：两个 client-HTTP 辅助（按物理调度进程 rank 取端口、叠加 dp 偏移）、`MooncakeStoreConfig.from_file` 与 `load_from_extra_config`（缺省 `protocol` 回退 `MOONCAKE_PROTOCOL.get()` 而非其 default）、`MooncakeBaseStore.register_buffer`（tuple/list 逐项注册）、`MooncakeStore.__init__`、`_get_hybrid_page_component_keys`（`DevicePoolEntry` 一页一对象分支）。

`MooncakeStore.register_mem_pool_host` 未迁入：DAS 相对官方只多一条接受 `layout_hcu` 的 assert，但该方法体有两处 `super()`，当前 FunctionType 抽取无法重绑。`layout_hcu` 经 `hcu_host_pool.py` 可达，因此这是真缺口，只是当前配方不启用存储后端、短期不命中。该缺口按具名叶子记录，不按文件整体记完成。

`mem_cache/hicache_key_scheme.py` 处置为 excluded，依据是上游自身没有启用：DAS 全树对该模块 import 数为 0（仅模块自身命中）、DAS `server_args` 未声明 `--hicache-storage-key-scheme`、模块外没有任何消费路径。保留原行不删；reason 写明「DAS 侧无消费者，插件侧不迁；若上游后续接线需重新评估」。

同批核对 A/B 容量差的归账：dump 轮 token 差 8064 × 16383.6 B/token = 0.123 GiB，Mamba 差 2 槽 = 0.140 GiB，合计 0.263 GiB；输入端差 0.29 GiB（加载前 0.14 + 权重多占 0.15），token 比例 0.2353% 与加载后可用显存比例 0.2402% 一致。同一轮 TP1 上 A/B 四个数完全相同（135.93 / 120.85 / 15.08），A 两个 rank 自洽；TP0 相对自身 TP1 的间隙在四轮里从 −0.14 变到 −1.98 GiB。静态扫描插件 500+ 文件、13 处 rank 门控 GPU 分配，加载路径上无按 rank0 分配的代码。因此该容量差按“环境噪声、未定位”归档，不作为尾延迟归因，证据在 `artifacts/kv-cpu-20261105/ab-capacity-accounting.json`。

验证边界：抽取契约 3 项、容器 CPU 缓存回归 20 项通过（含新增 4 项 mooncake 行为用例），GPU 可见开关均为 -1。未验证：真实 mooncake wheel 与 store、HiCache 存储流量、`layout_hcu` 注册、GPU/模型数值与性能；容量差与尾延迟的因果关系未验证。本批未 commit、未 push。

### 2026-11-05 models 纳入清单与 Qwen3.5 空钩子清除

此前阶段一清单只覆盖 layers、mem_cache、disaggregation、distributed、kernels 五类共 257 个文件，`srt/models` 一行没有；而 DAS 的 `srt/models` 相对固定官方有 113 个文件存在定义级差异。把 models 排除在等价声明之外等于宣告断言不完整，因此本批按当前目标配置（Qwen3.5 / TP2 / PP1 / attn_dp_size=1）的运行时可达性纳入全部 113 个文件。

可达性判定用运行时路径而不是静态 import 闭包：加载的模型类只有 `Qwen3_5ForConditionalGeneration`，因此运行路径上仅 `qwen3_5.py` 一个文件；其余 112 个文件属于其它模型，在当前目标配置下不执行。两者都按 `not_applicable` 记入 triage 并附具体理由，而不是逐个声称已迁移。

`qwen3_5.py` 的三处差异在本配置下判定为 no-op：`Qwen3_5GatedDeltaNet.forward` 的 `forward_mode`→`logical_forward_mode` 只在 DP relabel 时不同（`attn_dp_size=1` 两值相同）；`refreshes_rope_cos_sin` 在 DAS 全树没有读取方，仅 `qwen3_5.py` 与 `interns2_mobius.py` 两处赋值；`_refreshes_rope_cos_sin` 只被那处赋值调用。未验证 DP relabel 与 PP>1 的 stage 行为。

同时清除插件侧的空钩子：`hcu_sglang/models/qwen/qwen3_5.py` 的 AROUND 直接 `return original(...)`，四个 import 均未使用，DAS 的 qwen3_5 也没有接 fused RMSNorm+RoPE。已从 [插件入口](../hcu_sglang/__init__.py) 移除该模块的导入与 `register()`，并从目标合同删除该字符串，新增 [回归](../tests/test_qwen3_5_hook_removed.py) 断言入口不再引用该模块、全插件没有模块注册该目标、合同不再声明它。源文件本体尚未删除，等用户确认后再删，避免违反“删文件先问用户”。

清单分母变化：triage 257 → 370，closure 同步为 370，当前分布 implemented 73 / in_progress 23 / pending 155 / separate_component 114 / upstream 5。其中 in_progress 从 14 增至 23、upstream 从 4 增至 5，是因为 10 个 models 文件原本已有 decision 但未被旧清单计数，不是本批新增实现。

验证边界：本批只做静态差异、可达性判定与注册清理；没有运行 Qwen3.5 逐层 dump、DP relabel、PP>1、GPU、模型数值或性能验证。113 个文件的 not_applicable 是配置范围判定，不是等价性证明。未 commit、未 push。

### 2026-11-05 配置范围外行的措辞约束

112 个 models 文件按配置范围判为不可达，全部以 `not_applicable` 记入清单，但这类行最容易在后续阅读中被误当成“已处置完成”。为此对这 112 条统一写入 `review_reason: 配置范围外，非等价证明`，并把同一措辞放在 `rationale` 开头；`qwen3_5.py` 作为配置内可达文件使用单独措辞，不带这句，避免被统计进来。

closure 的 `rationale` 与 `decision_json` 两列都携带该字段，因此可以按措辞精确统计。新增回归断言 decisions 中恰好 112 条 review_reason 等于该措辞、路径都在 `srt/models/` 下且排除 `qwen3_5.py`，并断言 closure 的 rationale 列同样数出 112 条且 group 全为 models。这条约束写入测试后，任何人把配置外行改成“已完成”或改动措辞都会使单测失败。

验证边界：措辞与统计是台账约束，不构成任何等价性证明；这 112 个文件的实际行为仍按其所属模型进入目标配置时重新评估。未 commit、未 push。

### 2026-11-05 判据B：噪声基线容差等价

layer 0 在 gfx936/BF16 KV 环境下首层最大绝对差 1.562e-2，至 layer 63 相对差 88.7%。**该 1.562e-2 未做 ulp 归一化**：在 |x|≈1 处约合 4 ulp；"端点比值约 16%/层"是由 layer 0 与 layer 63 反推的比值（1.562e-2×1.16^63≈181），**没有中间层实测斜率，不能当作观测结论**。跨 kernel 构建的栈上把“逐位相同”作为验收条件不可维护，但也不能凭主观阈值判“这只是舍入”。本批把验收口径改为**用同侧重跑噪声做基线**：

1. B 侧（原生 DAS）在同环境、同 prompt 集、同 greedy 参数下独立启动两次，得到 B1/B2，逐条比较 input logprobs 的 max|Δ|、p99|Δ|、p50|Δ|、超差位置数与最差位置，并记录文本是否逐字符一致；
2. 用同一组指标比较 A-vs-B；
3. A-vs-B 的 max|Δ| 与 p99|Δ| 均不超过 B-vs-B 对应值 → 判数值等价，位置级超差记“数值放大”；任一指标显著大于 B-vs-B → 仍有真问题。

“显著”的操作化倍数暂定 2 倍，需 Chief 批准后才写进结论；未批准前只报原始指标。**若 B-vs-B 基线为零（逐位相同），则 A-vs-B 的任何差异都不能归因于噪声，判据B不成立**，必须回到算子定位（GEMM 后端、`layer_boundary` 融合）。文本级分叉（r02/r08）不因此自动豁免，需给出分歧 token 处 top1/top2 logit 间距，证明扰动足以翻转近 tie。

配套实验（26 上做，一轮只改一个变量，命中条件只有 `max_abs == 0`）：① 把 `bf16_gemm_backend`/`fp8_gemm_runner_backend` 固定为与 B 相同实现；② 若①不为 0，只停 `layer_boundary` 钩子、其它全留。gfx936/47 的观测不得外推到 26。

**后续实测（2026-11-06）：判据B 已作废。** B-vs-B 两轮（47、同 prompt、同 greedy、各重启一次）逐位相同——27/27 张量键一致、五层 max_abs 全 0.000e+00、64 层哈希全等、embed 相同。同侧噪声基线为零，故 A-vs-B 的差异不能归因噪声；数值结论只剩两种：找到算子并对齐，或如实记为“数值不等价、源头未定位”。判据定义、作废依据与未验证项见 `artifacts/kv-cpu-20261105/criterion-b-noise-baseline.json`。未 commit、未 push。

### 2026-11-05 closure 分母并集与缺记录门禁

代码审核家用交集/差集量出台账口径问题：closure 的 370 行里，149 行在 decisions 中没有处置记录却被默认记成 pending（占四成）；另有 144 个 decision 键不在分母内。因此“370 个文件全部有处置状态”此前无法验证。

本批按 Chief 的批准改了 [closure 生成器](../tools/build_runtime_closure.py)：分母改为 `triage ∪ decisions`（787 条补入后自动并入同一并集），仅存在于 decision 的路径以 `decisions_only` 分组进入；triage 有行但无 decision 时 execution_status 记 `missing_decision`，**构建与 `--check` 均以非零退出**，不再默认填 pending。[L0](../ci/l0_gate.py) 除 closure 检查外，另起一项直接统计 `missing_decision` 条数并作为门禁失败项。

整改前基线（本次实测）：分母 514 条 = triage 370 ∪ decisions 365（交集 221）；`missing_decision` **149** 条；分布 implemented 154 / in_progress 61 / pending 6 / separate_component 114 / upstream 30。注意其中的 implemented 154 与旧口径 closure 的 74 是不同分母，不可互相印证。

因此 L0 当前按设计**为失败状态**，失败项为「149 条缺处置记录」。这不是回归，而是把此前被默认值掩盖的缺口暴露出来：阶段一在补齐这 149 条处置之前不应判完成。回归同步更新：新增并集分母、`missing_decision` 不再默认 pending、L0 计数的断言。

未验证：149 条处置本身尚未补写；787 条扩围未并入；并集分母下各分组的完成度未重算。未 commit、未 push。

### 2026-11-06 hybrid_pool_assembler 13 定义迁入

`hybrid_cache/hybrid_pool_assembler.py`（2753 行）是当前唯一被判定“启动路径可达且无覆盖”的大件：目标模型为混合 GDN/FA，启动日志 `UnifiedRadixCache hybrid_ssm=True`，该构造路径调用 `attach_hybrid_pool_to_unified_cache`。DAS 相对官方有 9 个 changed + 4 个 added 定义。

本批按“只抽 13 个定义、其余 helper 走官方”的方案迁入 [cpu_cache_methods](../hcu_sglang/mem_cache/cpu_cache_methods.py)：`_MambaStrategy.build`、`_MambaSwaStrategy.build`、`_MambaSwaStrategy.build_direct_linker_pool_group`、`_build_mha_mla_host_pool`、`build_full_draft_pools`、`build_hybrid_mamba_stack`、`build_hybrid_mamba_swa_stack`、`build_hybrid_swa_group`、`build_kv_host_pool`、`build_mamba_host_pool`、`_mha_host_inputs`、`attach_hybrid_dsa_pool_to_hiradix_cache`、`attach_hybrid_minimax_sparse_pool_to_hiradix_cache`。9 个 changed 走公开 AROUND 目标，4 个 added 作为 helper 安装。

**闭包断言由脚本证明**（[checker](../tools/check_hybrid_pool_closure.py)）：13 个定义共引用 16 个模块级 helper/类（`build_pool_entry`、`_split_hicache_size`、`_get_allocator_type`、`_build_pool_entries`、`_swa_allocation_callbacks`、`_uses_unified_page_envelope_host`、`_swa_layer_mappings`、`_mtp_swa_device_pools`、`_mtp_swa_sidecar_specs`、`_stage_local_layer_mapping`、`_evict_mamba_for_device_alloc`、`_evict_swa_for_device_alloc`、`build_kv_only_stack`、`build_host_pool_group`、`build_minimax_sparse_hicache_stack`、`StackBuildResult`），逐个断言官方与 DAS **AST 相同**（all_equal=true），证据 `artifacts/kv-cpu-20261105/hybrid-pool-closure.json`。因此残留调用走官方实现行为等价，不会出现“DAS 函数体 + 官方 helper”的混搭。

**负例测试**证明该断言能拦住混搭：篡改官方 `_get_allocator_type` 后 checker 报 `all_equal=false` 并指出该 helper；删除官方 `build_pool_entry` 后同样报错。抽取契约同时按 SHA 逐定义校验，可逆变换回源字节。

合规：`hybrid_pool_assembler.py` 在 DAS 与官方都无许可证头，因此 donor 与 glue 不补造 upstream 头、不加来源/迁移注释；来源与定义映射只写在 `cpu_cache_sources.json`。`cpu_cache_methods.py` 已在 `runtime_owned_sources.json` 的 `extracted_files` 中登记（10 个来源，含本文件，target SHA 固定）。

边界：本批只做静态与 CPU/契约验证。真实混合池装配、`attach_hybrid_pool_to_unified_cache` 的实际调用、GPU 与模型数值均未验证，因此记 `implemented`，**不记 verified**。未 commit、未 push。

### 2026-11-06 hybrid_pool_assembler 迁移尝试与回滚

按“只抽 13 个定义、其余 helper 走官方”的方案完成抽取后，接线前做依赖核查时发现一个会直接导致 A 启动崩溃的依赖：抽取的函数 `build_mamba_host_pool`、`build_hybrid_mamba_stack`、`build_hybrid_mamba_swa_stack` 都调用 `buffer_info.validate(...)`，而官方 `DevicePoolInfo.buffer_info` 的类型是 `MambaStateBufferInfo`，**没有** `validate` 方法；`MHABufferInfo`/`MLABufferInfo` 这两个类本身是 DAS 新增，官方模块里根本不存在。

可达链是实打实的：`attach_hybrid_pool_to_unified_cache` → `_select_strategy(...).build` → `_MambaStrategy.build` → `build_hybrid_mamba_stack` → `build_mamba_host_pool` → `buffer_info.validate()`。也就是说，哪怕 helper 闭包全部 AST 相同，**类型层面的 API 差异仍会炸**。

处理：把 `hybrid_pool_assembler.py` 的 13 个定义抽取与 9 个 AROUND 接线、以及 `device_pool_info.py` 的 2 个 validate 一并回滚，仓库回到无接线状态；`tools/check_hybrid_pool_closure.py` 与 `tests/test_hybrid_pool_closure.py`（16/16 helper AST 相等 + 两条负例）保留，作为后续迁移的有效证据。两文件状态回到 pending，并写明前置条件：先迁 `MHABufferInfo`/`MLABufferInfo` 类、加宽 `DevicePoolInfo`，再配套 `memory_pool.get_device_pool_infos`，然后才允许重做装配器接线。

教训（写给后来者）：**闭包断言只覆盖“函数级”等价，覆盖不了“被调用类型上的方法是否存在”**。以后做抽取式迁移，除 helper 闭包外还要断言“被调用对象的方法在官方类型上存在”，否则代码在静态层面全绿、运行时才炸。

未验证：回滚后本轮未启动服务；真实混合池装配路径仍未在 GPU 上验证。未 commit、未 push。
### 2026-11-06 注册目标可解析性检查、mooncake 注入式接线与运行时集合重判缺口

运行时对账把「静态扫描已足够」推翻（静态 421 / 运行时 632 / static_only 74 / runtime_only 285），本轮按 R10 补齐三件事。

**一、扫描器的未判定桶从计数变成可追溯**

`tools/scan_hook_targets.py` 原来只输出 `unresolved: 45`，没有明细，无法复查。现在每条未判定记录 `{file, line, expr}`，45 条全部落盘，分布最多的是 `models/deepseek/hook_utils.py`(8) 与 `disagg/transport_mooncake.py`(4)。同时新增 `--check-resolvable`：按固定官方树（`../input-code/sglang/python`）静态解析每个注册目标能否在目标模块上 resolve，`unresolvable` 非空即非零退出。解析覆盖模块级绑定、类体成员、以及跨模块导入的基类（`from X import Y` 递归）；解析不到的基类进独立桶 `missing_but_opaque_base`，不并入任何既有分类。

先在 `tests/test_hook_target_resolvability.py` 立两道自测关卡再出数：合成树上「已知存在必须解析到」（含 `if` 块内函数、跨模块继承方法）、「已知不存在必须报缺失」，外加对真实官方树回归两条（`fp8_kernel.scaled_fp8_quant` 必须解析到、`cp.utils.enable_cp_v2` 必须报缺失），5 项通过后才允许跑全量。

三桶结果（421 个目标，实跑退出码 1）：

```text
resolvable 360 / unresolvable 21 / undetermined 45（targets 426 / unresolved 37；其中 9 条非 Hook 的 .register 已单列，5 个 l2norm 目标已并入，1 条 write_mtp_seed 属「插件先安装再注册」）
unresolvable: missing_module_attribute 21 + missing_class_attribute 1
```

22 条不可解析按 owner 分布：`dsa_utils.py` 11、`cp_runtime.py` 9、`index_topk_share.py` 1（`write_mtp_seed`）、`distributed_backend.py` 1（`GroupCoordinator._init_attn_cp_overlap_group`）。

**返工记录（写给自己和后来者）**：这份数字出过两次错。第一次把 `scaled_fp8_quant` 判成缺失——解析器只扫模块顶层语句，漏掉 `if` 块里的模块级定义，属于假阳性；第二次把 190 条因基类在别的模块而落进未判定桶，是收得太窄。两次都是「工具给自己一个看起来正常的结论」，所以现在改成先跑自测关卡再出数。这些是 DAS 独有的模块级或类成员名字，却按 AROUND 目标注册到官方模块上，与 mooncake 那次同类。**这不是已观测到的 apply 失败**：运行时对账只报了 2 条 mooncake 的 AttributeError，因为这 23 条所在模块在 dump 时未被导入；按「注册目标必须能在目标模块 resolve」的判据，它们属于潜在静默不生效，构建期就该拦住。

**二、mooncake 两处从 AROUND 改为注入式接线**

`_get_mooncake_client_http_port` / `_get_mooncake_client_http_setup_kwargs` 是 DAS 在官方 0.5.22 之外新增的**模块级**函数，官方模块没有这两个属性；原接线把它们当 AROUND 目标注册，注册成功、apply 时 `AttributeError`。现改为 `_MODULE_HELPERS` 注入：命中该模块的钩子在构造绑定命名空间时用 `setdefault` 注入这两个函数（已存在则不覆盖），owned `MooncakeStore.__init__` 通过同一命名空间解析到它们；同时禁止 `class_name` 为空的组注册 `parent.__init__`。

回归：`tests/runtime/cpu_cache_cpu.py` 新增 2 项——注入后两个名字可用且嵌套调用可解析，以及注册表里不再出现这两个模块级目标、但仍有 `MooncakeStore.__init__`。全套 22 项通过。

**三、用运行时集合重判剩余缺口（定义级）**

新增 `tools/audit_gap_reachability.py`：对每个缺口文件用官方树与 DAS 树做定义级 AST 比较得到变更定义清单，再逐个与运行时注册目标集合精确匹配（`module.Name` 或 `module.Class.method`，含子属性前缀）；同时对 DAS 树做导入普查（解析失败、DAS 树不足 1000 文件均直接报错，不静默当空）。

对 23 个仍在 pending/in_progress 的缺口文件（旧 36 条减去 13 条已归档）：

```text
23 个文件 / 92 个变更定义 / 87 个定义在运行时集合中没有对应注册目标
5 个定义已被运行时覆盖：PagedTokenToKVPoolAllocator.__init__/alloc_extend/alloc_decode/free
                       + kv_cache_builder.build_kv_cache
2 个文件在 DAS 树内无静态导入方：custom_all_reduce_v2、triton_symm_mem_ag
  （记为「无静态导入方」，不等于死代码——动态导入未排除）
21 个文件存在「有导入方但定义未覆盖」
同名定义出现在其它模块的情况：0 条（排除「换个模块注册也算覆盖」）
```

另有 `tools/build_runtime_gaps.py`：按 owned 全量/抽取来源、公开 Hook 目标两路覆盖来源给 `changed`/`fork_only` 文件分桶，无法判定时单列并非零退出。全仓口径实跑为 covered_owned 84 / covered_hook 148 / 无覆盖 1193 / 未判定 37，与本批 23 条是两个分母，不可混用。

**验证边界**：本轮工具与接线只做静态与容器内 CPU 验证（GPU 可见开关均为 -1）；真实 Hook apply 只来自 kimi 在 GPU 上的那次 dump（648 注册 / 630 applied），本轮未重跑；23 条潜在不可解析目标未做运行时确认；`static_only 74` 的逐条理由尚未写。未 commit、未 push。
### 2026-11-06 fla l2norm 整文件迁入、两机可用性实测与扫描器伪命中清理

**一、`sglang/kernels/ops/attention/fla/l2norm.py` 整文件迁入**

DAS 在该文件里给 `l2norm_fwd` 加了 vendor 快车道：`from sgl_kernel import l2norm` 成功，或 `torch.ops.sgl_kernel.l2norm` 存在，且输入是 CUDA fp16/bf16/fp32 时，改用该算子，否则回落到原有的两个 Triton kernel。固定官方 0.5.22 只有 Triton 实现，插件此前没有这个文件，因此 A 一直走 Triton、B 走 vendor 算子，而两侧调用点相同（`gdn_backend` 三处 `use_qk_l2norm_in_kernel=True` → `chunk.py`/`kda.py` 调用 `l2norm_fwd`）。

迁移方式是逐字节复制到 [l2norm_source](../hcu_sglang/layers/attention/fla/l2norm_source.py)（`sha256 c8142677d81757f23de083f6952a3d81d7f1bba159f2440f3b0645655059f1d1`，与源文件相同），导入语句无需改写，`try/except` 回落逻辑原样保留。[l2norm](../hcu_sglang/layers/attention/fla/l2norm.py) 注册五个 REPLACE 目标：`l2norm.l2norm_fwd`、`l2norm.l2norm`、`l2norm.l2_norm`、`chunk.l2norm_fwd`、`kda.l2norm_fwd`。后两个是调用方模块里的已绑定名字——`chunk.py` 在模块导入时 `from ...l2norm import l2norm_fwd`，只替换定义模块的属性不会改变它的本地引用，因此必须同时替换调用方属性；钩子 apply 时 `pkgutil.resolve_name` 会先导入 `chunk`，`chunk.l2norm_fwd` 按目标名排序先于定义模块，替换结果对后续调用生效。

注册写成五条字面量调用而不是遍历 `_HOOKS`：循环里的目标对静态扫描器是不可求值表达式，会被记进未判定桶——那正是本轮在治的盲区。

五个目标先用 `scan_hook_targets.py --check-resolvable` 对固定官方树逐个验证可解析（5/5 resolvable），再登记 `runtime_owned_sources.json`（`reverse_replacements` 为空，反向重建等于源字节）与 `hcu_source_alignment_targets.json`。`plans/sglang-das-decisions.json` 该文件由 pending 改 in_progress，closure 重建后 in_progress 66→67、pending 646→645、missing_decision 0、`--check` rc 0。

**二、26 与 47 两台机器的 `sgl_kernel.l2norm` 可用性（实测，互相矛盾的说法已裁定）**

```text
26（nmz26，容器 yangyn_sglang_plugin_1022，镜像 562c59c8…，torch 2.11.0，sgl_kernel 0.4.6.post1）
  直接 hasattr(torch.ops.sgl_kernel, "l2norm")          → False（新进程、未导入 sgl_kernel 时）
  from sgl_kernel import l2norm                          → ImportError: cannot import name 'l2norm'
  执行 DAS 原始 try/except 之后                          → _has_l2norm_cuda = True，算子 sgl_kernel.l2norm
47（bw47，容器 sglang-agent，同版本 sgl_kernel）
  同样路径                                                → _has_l2norm_cuda = True，算子 sgl_kernel.l2norm
```

结论：**两台机器上 vendor 快车道都可用**。26 上出现「不可用」的读法来自检查顺序——`from sgl_kernel import l2norm` 虽然抛 ImportError，但这次失败导入已经把 sgl_kernel 的算子库加载进来并注册了 `torch.ops.sgl_kernel.l2norm`，所以 DAS 的 `except` 分支随后 `hasattr` 得到 True。新进程里先直接 `hasattr`、或先查 `torch.ops.sgl_kernel`，才会得到 False/ModuleNotFoundError。因此「这条 1 ulp 分歧是 47 特有、26 上两侧都回落 Triton」不能用该现象支撑，仍须在 26 上跑 A/B 的 layer 0 对比。

**三、扫描器伪命中按接收者剔除**

`sglang` 树里任何对象的 `.register(...)` 原来都被当成 HookRegistry 注册。现按接收者判定：只接受 `HookRegistry`、`*.HookRegistry`、以及从 `*hook_registry` 导入的别名；其余单列 `non_hook_register` 桶（不并入目标、不并入未判定）。实测 9 条：`poller.register`（disagg/transport_mooncake.py 两行各两次）、`atexit.register`（cpu_cache_methods.py、window_cache_init.py）、`self.kv_event_lora_names.register`、`AttentionBackendRegistry.register`、`registry.register`。未判定桶因此从 45 降到 37，目标 421→426（含新增的 5 个 l2norm 目标），`resolvable 360 / unresolvable 22 / undetermined 44`。

**四、运行时 dump 增加 owner 与 apply 失败明细**

`tools/dump_runtime_hook_targets.py` 现在记录每个目标的 `{module, name, file}`（取自钩子对象的 `__module__`），并在 `--apply` 时逐个目标捕获 `_apply_target` 的异常到 `apply_failures`，不再只依赖日志里的 `Failed to apply hooks`。这两项正是把 37 条未判定与 285 条 `runtime_only` join 起来、以及判断 mooncake 是否闭合所缺的字段。**该工具需要在真实 GPU 进程里跑，本轮未执行**。

验证边界：本批完成静态、CPU 行为与合同验证（`tests/test_l2norm_migration.py` 5 项、`tests/test_hook_target_owner.py` 2 项；全仓 186 项通过 / 13 skip；L0 passed，671 Python / 8 Skill / 0 errors）。**未验证**：真实 vendor 算子执行、Hook apply 后 A 侧 op 集合是否出现 `sgl_kernel.l2norm`、layer 0 及后续层 `max_abs` 是否归零、性能影响、`chunk.py` 已绑定引用的实际替换效果。A 侧基线指纹 `7919f29e` 在迁移生效后必然失效，需按新代码重取两次自洽基线。未 commit、未 push。

**五、`static_only 74` 逐条归因到具体门控**

新增 `tools/explain_static_only.py`：对每条只出现于静态扫描的目标，定位它在插件里的注册语句（AST，含 for 循环字面量展开、f-string 与常量拼接求值），再向上收集外层条件链（`if` 条件、循环迭代式、所在函数），输出 `{target, line, guards}`。74 条全部定位到注册点，退出码 0（无“无注册点”“owner 未知”）。证据 `artifacts/kv-cpu-20261105/static-only-guards-20261106.json`。

按机制归类（合计 74）：

```text
56  能力门控：官方没有该 DAS-only 符号 → 注册被跳过，改为 setattr 直接安装实现
      dsa_utils.py 11        if hasattr(U, name)              U = sglang.srt.layers.attention.dsa.utils
      cp_runtime.py 9        if hasattr(C, name)              C = sglang.srt.layers.cp.utils
      transport_common.py 7  if hasattr(N.CommonKVManager, '<name>')
      transport_mooncake.py 29  if hasattr(N.MooncakeKVManager, '<name>')
    与 `--check-resolvable` 的 22 条不可解析同源：官方树没有这些属性，hasattr 为假，所以既不注册也不报错。
 1  显式按名字排除：distributed_backend.py 的循环里 `name != '_init_attn_cp_overlap_group'`
16  注册函数没有调用点
      layers/dp_attention.py 10   ← 聚合函数 `hcu_sglang.layers.register_hooks()` 无任何调用者
      current_v4/backend.py 5、current_v4/compressor.py 1
                                  ← 二者的 `register_hooks` 也无调用点；同包的
                                    `hcu_backend/integration.py` 用 `_FORMAL` 表对
                                    `DeepseekV4AttnBackend` 等做 class REPLACE，功能上可能已取代，
                                    但“已取代”需要 DeepSeek V4 负责人确认，本轮不据此下结论
 1  空钩子已移除：models/qwen/qwen3_5.py 的 `register` 已从插件入口删除（文件本体待删）
```

两条需要决策的发现（本轮只报告，未改行为）：

1. `hcu_sglang/layers/__init__.py::register_hooks()` 没有任何调用者，入口是逐个 import 子模块再调用各自 `register`。后果是 `dp_attention` 的 10 个 hook 在任何配置下都不会生效——运行时集合里 `dp_attention` 目标数为 0，与此一致。当前目标配置 `attn_dp_size=1` 不依赖它，所以不影响现有结论；但 DP attention 一旦启用，这批 hook 是静默不生效。
2. `current_v4/backend.py` 与 `current_v4/compressor.py` 的 `register_hooks` 同样没有调用点。它们注册的是 `DeepseekV4AttnBackend.__init__/_forward_attention/_forward_prefill_sparse/enter_late_layer_tail/init_forward_metadata` 与 `_low_ratio_compress_fused`，而同包 `hcu_backend/integration.py` 已用 class REPLACE 接管该类。两种解释（旧路径被取代 / V4 缺 6 个 hook）在静态层面无法区分，需要模型负责人按 V4 用例确认。

**验证边界**：归因是静态与集合差集推出来的，`dp_attention` 与 `current_v4` 两组“无调用点”在运行时集合中确为 0（`runtime_targets.json` 查询），但没有跑 DP attention 或 DeepSeek V4 用例来验证功能影响；`hasattr` 型 56 条的“官方无该符号”与本轮 `--check-resolvable` 的 22 条不可解析互相独立地指向同一批名字。未 commit、未 push。

**六、V4 六个 hook 的取代判定与 DP attention 缺陷登记**

按「静态能判的不留给人拍脑袋」，逐个核对 `current_v4/backend.py`(5) 与 `current_v4/compressor.py`(1) 的 `register_hooks` 所注册的目标是否已落在 `hcu_backend/integration.py` 的 class REPLACE 方法面里。

结论：**6/6 全部 superseded**，证据 `artifacts/kv-cpu-20261105/v4-hook-supersession-20261106.json`。

```text
class REPLACE 条目      integration.py:32-34  DeepseekV4AttnBackend -> hcu_backend.backend.DeepseekV4AttnBackend
注册动作                integration.py:603    HookRegistry.register(target, getattr(owned[module], name), HookType.REPLACE)
替换类定义              hcu_backend/backend.py:765，共 107 个方法，基类 AttentionBackend/C4IndexerBackendMixin/CompressorBackendMixin
六个方法在替换类中的位置
  __init__                784-886     （17 个语句）
  enter_late_layer_tail   1165-1203
  init_forward_metadata   1442-1454
  _low_ratio_compress_fused 1857-1887
  _forward_attention      2642-2739
  _forward_prefill_sparse 2741-2796
```

注意判定边界：这是**方法面覆盖**，即替换类自身定义了同名方法；方法体是否与旧 hook 行为等价，静态判定不了，需要 DeepSeek V4 用例验证，本轮未验证。旧 `register_hooks` 及其 `around_backend_init`/`around_init_forward_metadata`/`around_enter_late_layer_tail`/`_forward_attention`/`_forward_prefill_sparse` 因此成为无调用点的死代码；**删除需用户批准**，本轮不删。

DP attention 缺陷按裁定只登记不改动，台账措辞：``启用 DP attention（`attn_dp_size>1`）时，`hcu_sglang/layers/dp_attention.py` 的 10 个 hook 静默不生效：`hcu_sglang.layers.register_hooks()` 无调用者；当前配置 `attn_dp_size=1` 不可达，运行时集合中 dp_attention 目标数为 0。不删除该函数（删除等于放弃 DAS 已有能力），待支持 DP attention 时接线并补 DP 配置验证。``

未验证：DP attention 配置下的注册与行为、DeepSeek V4 用例下 6 个方法的功能等价。未 commit、未 push。

**七、l2norm 闭环：47 上从 layer 0 到 layer 63 逐层零差**

kimi 在 47（gfx936、BF16 KV）上完成验证：迁移落地后 A 侧两次独立启动（`/flush_cache` 后）自证 `A_m1 = A_m2 = 4f33666a`，`max_abs 0.0`；新 A 与 `B_ref` 在 `|1081` 键上逐层比较，layer 0/1/4/8/16/32/48/63 全部 `max_abs = 0.000e+00`，两侧哈希同值（`4f33666a / 544deb94 / 40b91b8e / e49ccccd / 2bdb650c / 4a5dbfe6 / c7d8f2d4 / 85a35a27`）。迁移前的旧 A 基线 `7919f29e` 作废。

因此本条的定性是：**这是「少迁一个文件」导致的系统性差异，不是 kernel 不兼容**。DAS 给 `fla/l2norm.py` 增加了 vendor 快车道，插件没有迁这个文件，A 就一直走官方 Triton，B 走 vendor 算子；两侧调用点相同、权重逐字节相同，差异只来自同一个调用点落到不同实现。迁入后级联放大消失，原先按端点反推的「约 16%/层」放大率不再成立。**不需要在「逐位一致」和「融合性能」之间取舍**：vendor 算子本来就比 Triton 快。

**方法沉淀（可提前发现同类问题）**：这类缺口可以用「运行时 op 集合 A-vs-B 对账」提前暴露——**B 有而 A 没有的算子即嫌疑**。本轮的 `sgl_kernel.l2norm.default` 就是先从那轮 ops 差集里看到的（当时因 R8 未通过只作旁证），再由三树静态比对定位到具体文件。以后每次 A/B 轮次都应顺带落一份 op 集合差集，作为「少迁实现」类缺口的早期信号；注意该差集本身不能作为数值结论，只能作为线索，数值结论仍须双侧 R8 通过后按 ulp 归一化给出。

**边界（如实框住）**：结论只在 47 上成立；26（gfx938、fp8 KV）才是目标环境，未复核；性能影响未测；`r02`/`r08` 两条文本分叉在新代码下未重跑；旧 A 代码下产出的一切数值结论（BF16 `r04` 18.64 nats、各轮 layer 0 `1.562e-2`、`r02`/`r08` fail、`f1dc72a1` 相关轮次）随根因修复全部失效，须用新代码重跑后才可再次引用。未 commit、未 push。

**八、开源合规补登：FLA MIT 全文、四个原创文件许可证头与来源合同重钉**

代码审核家预审报的 A1 是阻塞项：FLA（flash-linear-attention）来源未登记、MIT 全文缺失。本轮补齐：

```text
licenses/{fla,deepep,flashmla,fast-hadamard-transform}.LICENSE
                                        四份第三方许可证全文 + fast-hadamard 的 AUTHORS，逐字未改；分别取自固定 commit
                                        fla 9e82a80f(2026-11-04)、deepep f6aa38fe(2026-10-21)、
                                        flashmla 395cb3af(2026-10-27；代码来源为 fork 分支 endurehero:will_fp8_mr @7fafcd21，PR #54 未合并)、fast-hadamard sgl-project fork 7fd811c2(2025-10-06；与 Dao-AILab e7706faf 正文逐字节相同)，
                                        取全文日期 2026-11-06；落盘后用 curl+sha256 与上游原文逐个比对，四份全部一致
THIRD_PARTY_NOTICES.md §1               新增「flash-linear-attention（MIT）」条目：仓库 URL、许可证副本链接、
                                        11 个本地文件路径、HYGON 修改说明
```

本地文件按实际 grep 结果登记 **11 个**，不是预审里列的 8 个：`layers/attention/fla/` 下 `chunk_delta_h_source.py`、`chunk_o_source.py`、`fused_recurrent_source.py`、`layernorm_gated_source.py`、`l2norm_source.py`，`layers/attention/linear/kda_kernels/` 下 `chunk_delta_h.py`、`chunk_intra.py`、`chunk_intra_token_parallel.py`、`fused_norm_gate.py`、`kda.py`、`l2norm.py`——预审漏了 `layernorm_gated_source.py`、`chunk_intra.py`、`chunk_intra_token_parallel.py`、`fused_norm_gate.py` 四个。登记按实际文件走，不照抄清单。

B1：四个原创文件补 H1 头（`Copyright (c) 2026 Hygon…` + `SPDX-License-Identifier: Apache-2.0`）：`models/registry.py`、`models/glm5_next/entries/text.py`、`models/glm5_next/entries/nextn.py`、`models/muse_spark/entries/omni.py`。这四个都在 `runtime_owned_sources.json` 的 `adapter_files` 段，加头后 SHA 变化，已同步重钉（`d1695321→b392227d`、`32a791bd→c0763865`、`356277f6→02eb3318`、`71fba808→da67fd38`）。`adapter_files` 不参与「源文件头必须保留」的检查，因此加自家头与源声明义务不冲突。

B2 经核实**不成立**，不改文件：预审称这四个文件「未进来源合同」，实际都在合同里且 SHA 已钉——

```text
hcu_sglang/layers/quantization/kv_cache.py      adapter? 否，extracted_files，1 个来源，target_sha256 已钉
hcu_sglang/layers/quantization/moe_wna16.py     extracted_files，1 个来源，已钉
hcu_sglang/layers/quantization/auto_round.py    extracted_files，1 个来源，已钉
hcu_sglang/mem_cache/cpu_cache_methods.py       extracted_files，9 个来源，已钉
合同总行数 79 = files 64 + extracted_files 6 + adapter_files 9；migration_header_errors 实跑 0 条。
```
差异来源应是预审只数了 `files` 段（64 行），把 `extracted_files`/`adapter_files` 段漏在外面。


**R13 纠正记录**：最初为对齐本地文件头的 2023-2025，曾把 FLA 许可证正文的版权行改成 2023-2025，这违反“第三方许可证正文严禁修改”。已撤回：`licenses/fla.LICENSE` 恢复上游逐字原文（2023-2026），本地文件头与许可证年份的差异如实写进 `THIRD_PARTY_NOTICES.md` 的「版本口径」，不写进许可证文件。另补 DeepEP、FlashMLA 两份 MIT 全文与 fast-hadamard-transform 的 BSD-3 全文（此前已登记但缺全文）。

验证：`migration_header_errors` 实跑 0 条；全仓单测 `Ran 187 tests, OK (skipped=13)`；`L0 passed`（672 Python / 8 Skill / 0 errors）。未验证：TensorRT-LLM、FasterTransformer、marlin、FastVideo 四家存在 main/PR/无版本引用，按「逐文件实际来源链」结论暂不取全文，标来源链未确认；pytorch 3 处与 flashinfer 的 topk_v2.cuh 判为注释级/信用引用，需在登记表写明排除理由。未 commit、未 push。

**九、阶段一判据整改：措辞补齐、缺口表分栏、分母 593 与 qwen3_5 的口径纠正**

测试校对家独立核对阶段一判据时发现两处状态冲突与本 bot 的三处数字问题，本批按裁定整改。

**一、`separate_component` 措辞补齐（11 条）与 qwen3_5 的口径纠正（1 条）**

`review_reason` 未写「配置范围外，非等价证明」原文的 12 条里，11 条按原文改写：`muse_spark/{helion_norm_fp4_quant,helion_norm_gate,helion_post_expert,helion_qk_norm}.py`（HELION 仅 `CapabilityRequirement.CUDA`，HCU 不命中）、`dsv4/compress_hip.py`（固定 DAS 无此文件）、`intel_amx_backend.py`/`xpu_backend.py`（registry 仅 intel_amx/XPU 条目导入）、3 个来源测试文件（`test_flash_with_topk_idx.py`、`test_fp8_attn_gemm.py`、`aibrix_kvcache/unit_test.py`、`tensorcast_store/test_tensorcast_store.py`）。

第 12 条 `models/qwen3_5.py` **不能写那句**：它是目标配置**可达**的文件，差异只是 no-op（`attn_dp_size=1`/`pp_size=1` 下 `forward_mode` 重标两值相同、`refreshes_rope_cos_sin` 无读取方），写「配置范围外」是不实陈述。改为 `pending` 并带明确 todo：在 `attn_dp_size>1`、`pp_size>1` 下验证三处差异是否仍为 no-op，可达则迁移对应定义。因此 `separate_component` 由 593 变 **592**，repo 内对该措辞的机械校验计数由 591 更新为 **602**（`tests/test_runtime_triage.py` 两处断言同步）。

**二、真缺口表与 `separate_component` 不再双记**

原 `mem-cache-no-coverage-20261106.json` 的 36 条与 decisions 状态冲突：13 条同时被记为缺口又被关闭为配置范围外，且**所有条目都没有待办字段**。新增 gap-ledger v2（本地历史产物 `artifacts/kv-cpu-20261105/gap-ledger-20261106-v2.json`，未随仓库提交）（schema 2）分两栏，36 = **open 23 + closed 13**：

```text
open_gaps 23  = 22 pending + 1 in_progress；每条带 todo，且 open/closed 交集为 0
closed   13   = 配置范围外已关闭；每条保留 reason + evidence
evidence 字段 = test-results/ab/dump47/A.log:29（A 侧实际生效 server_args 全量 dump）
```

13 条的可达性用**实际生效参数**逐条证明（不是推断）：
```text
disaggregation_mode='null'                    → decode_kvcache_offload_manager、decode_schedule_batch_mixin、fake/conn、mori/conn
hicache_storage_backend=None                  → aibrix_kvcache_storage
enable_unified_cache_external_linker=False    → unified_cache_linker、mooncake_direct_linker
enable_unified_memory=False                   → unified_memory_pool
speculative_algorithm=None                    → unified_draft_pool
kv_cache_dtype='bf16'                         → pool_host/mha_mxfp8
模型 layer_types 无滑窗层（uses_mamba_radix_cache=True、disable_hybrid_swa_memory=False）
                                              → pure_swa_radix_cache、swa_memory_pool
allocator/base.py：唯一差异方法 get_extend_allocation_demand 的唯一调用点在 A 侧无 Hook 且调用数为 0
```

**三、环境口径**：闭环与门禁脚本需 Python 3.10（`from __future__ import annotations`），宿主机默认 `python3` 是 3.6，直接跑会 SyntaxError 并被误判为回归；容器 `yangyn_sglang_plugin_1022` 内 `python3` 为 3.10.12，本轮全部门禁均在容器内执行。

验证：closure 重建 `files 1464 / implemented 130 / in_progress 66 / pending 646 / separate_component 592 / upstream 30 / missing_decision 0`，`--check` rc 0；全仓 `Ran 187 tests, OK (skipped=13)`；`L0 passed`（672 Python / 8 Skill / 0 errors）。未 commit、未 push。

**十、阶段二支撑：0.6B 逐层 dump 驱动参数化与 pending 待办补齐**

**一、`ab.sh` 新增 `layerdump` 子命令（0.6B / 27B 同一脚本，参数全走环境变量）**

```text
AB_MODEL                模型路径（容器内路径）
AB_TP                   TP 大小（config.env 由写死 TP=2 改为 ${AB_TP:-2}）
CARDS                   卡号（沿用原有变量）
AB_MEM_FRAC             显存比例（沿用；26 上 0.6B 建议 0.12–0.15）
AB_KV_DTYPE             KV 类型（沿用；26 目标 fp8_e4m3）
AB_CONTAINER            容器名（config.env 新增 ${AB_CONTAINER:-yangyn_sglang_plugin_1022}，47 上为 sglang-agent）
ATTN_DUMP_TOKENS        dump 命中 token（默认 1078；0.6B 短请求需显式传小值）
ATTN_DUMP_MODEL_MODULES 需要打层钩的模型模块（默认 sglang.srt.models.qwen3_5,sglang.srt.models.qwen3）
PORT_A / PORT_B / RUN_ID / OUT 沿用
```

`layerdump` 的流程：清掉容器内旧 dump → 起 A → 等就绪 → 发 `requests.jsonl` 首条请求 → `docker cp` 取回 `layer-dump / layer0-ops / layer0-weights / layer0-taps / tensors.pt` → 停 A → 对 B 重复 → 容器内跑 `compare_dumps.py` 比对。两侧 PYTHONPATH 前置 `dump_shim` 并注入 `ATTN_DUMP_SIDE/TOKENS/MODEL_MODULES`，仍只 kill 自己 pidfile 里的进程。

26 上 0.6B 的调用示例（`/mnt/qwen3/Qwen3-0.6B` 为容器内实际路径）：
```bash
AB_MODEL=/mnt/qwen3/Qwen3-0.6B AB_TP=1 CARDS=3 AB_MEM_FRAC=0.14 AB_KV_DTYPE=fp8_e4m3 \
ATTN_DUMP_TOKENS=32 PORT_A=30091 PORT_B=30092 RUN_ID=26-0p6b-dump \
./ab.sh layerdump
```

**二、dump shim 支持非 Qwen3.5 模型**

原 shim 只 patch `sglang.srt.models.qwen3_5` 里的 `Qwen3_5LinearDecoderLayer/Qwen3_5AttentionDecoderLayer`，0.6B（`sglang.srt.models.qwen3`）不会挂任何层钩，dump 会是空的。现新增 `patch_generic_model`：按 `*DecoderLayer` 命名约定包装模块内所有 Decoder 层，Qwen3.5 专有 taps 仍由 `getattr(..., None)` 保护、缺失即跳过；模型模块列表由 `ATTN_DUMP_MODEL_MODULES` 控制。Gated DeltaNet 的 `_forward_input_proj` tap 仍是 Qwen3.5 专属，0.6B 不会走。

**三、pending 待办补齐**

按要求先补 `srt/configs/qwen3_5.py`，随即审计发现**646 条 pending 中 644 条都没有 todo**（不止这一条）。按统一模板补齐，模板含该文件的 group / relation / 未覆盖定义数：

```text
核对 N 个未覆盖定义（group=…, relation=…）在声明配置下的实际调用可达性：
可达则迁移到 owned 实现并补回归与登记，不可达则按配置范围外归档并写明理由；核对前保持 pending
```

补齐后 `pending=646 / 缺 todo=0`，closure 一致（`--check` rc 0）。

验证边界：`layerdump` 与 shim 改动**只做了语法与静态检查**（`bash -n ab.sh`、`ast.parse` shim），没有 GPU 窗口的实际执行；0.6B 在 26 上能否正常出 dump、`ATTN_DUMP_MODEL_MODULES` 是否覆盖其层类，待 kimi 实跑确认。未 commit、未 push。

**十一、B 侧 0.6B 类型错误定性（只定性，未改 fork 源码）**

现象（kimi 实测，26 上 0.6B）：B（原生 sglang-das）启动后请求即崩 `RuntimeError: expected scalar type BFloat16 but found Float`，**fp8_e4m3 与 bf16 两种 KV 都一样**。

静态排查结论与已排除项：

```text
排除 1  KV dtype 参数——两种 KV 类型报同一个错，且报的是 BFloat16/float 而不是 Float8
排除 2  fork 的融合 qk-norm-RoPE 路径——models/qwen3.py:70 _use_fused_qwen_bailing_rotary
        取 SGLANG_USE_FUSED_RMS_ROTARY（默认 false），ab.sh 未设置；:299 分支不进入
排除 3  enable_fused_set_kv_buffer 差异——DAS models/utils.py 在 _is_hcu 下返回 cuda_enabled（要求 _is_cuda，
        在 HCU 恒 False），而插件 hcu_sglang/models/model_utils/methods/utils.py 的同名实现是同一逻辑，
        A/B 两侧一致，不构成差异源
排除 4  确定性/RL 的 fp32 norm 分支——models/qwen3.py:216-222、:486-495 都由
        get_exec().deterministic.rl_on_policy_target is not None 门控，本配方未开启
```

**已由 kimi 在 47 上取到的 traceback 校正（该假设作废，lightop RMSNorm 无关）**：

```text
RuntimeError: expected scalar type BFloat16 but found Float
  qwen3.py:417 forward → qwen3.py:341 forward_prepare_native → kernels/fused_op.py:653 forward
  → rotary_embedding/base.py:593 forward_cuda → base.py:158 _launch_rotary_embedding → :136 _launch
  → sgl_kernel/elementwise.py:433 rotary_embedding
```

静态对照后机制明确：`rotary_embedding/base.py` 的 fallback 分支里，`_is_hip` 为真时
`self.cos_sin_cache = self.cos_sin_cache.to(query.device)` **只搬设备、不转 dtype**（缓存保持 fp32），
随后 `_launch_rotary_embedding(self.fallback_rotary_embedding, …, self.cos_sin_cache, …)` 把它交给
`sgl_kernel.rotary_embedding`，而该算子要求与 bf16 的 query 一致 → 报此错。

一处必须写准的边界：**这段 fallback 代码在固定官方 0.5.22 与 DAS 中逐字相同**（官方 base.py:159-171、DAS base.py:281-293 与 444-452/582-590 两处分支一致），因此它**不是 DAS 私有缺陷**。

**A 与 B 的分支关系（已用代码与单测坐实，不再是推理）**：A 也走 fallback 分支，差别在被调用的 fallback kernel 不同。插件 `hcu_sglang/layers/rotary.py:100-107` 在 `RotaryEmbedding.__init__` 之后（HcuRotaryFallback，AFTER 钩子）把 `layer.fallback_rotary_embedding` 换成 `HcuRotaryFallback`；该包装器 `:90-98` 在调用原 kernel 前把 `cos_sin_cache` 转成 `query.dtype`（带 memo，缓存对象本身不改）。B 没有这层包装，fp32 cache 直接进入 `sgl_kernel.rotary_embedding`，与 bf16 query 冲突 → 报错。

这条已固化为 CPU 回归 `tests/test_rotary_fallback_dtype.py`（3 项）：fp32 cache 被转成 query dtype、dtype 一致时原样透传、转换结果按缓存对象 memo 复用；同时断言传入的 cache 张量本身未被改写。即在无 GPU 的环境下也能复现并证明“A 有转换、B 没有”。

风险边界：触发条件是「该模型的 rotary 在 HCU 上落到 fallback 分支」。**不排除其它小模型/小配置同样触发**；
遇到同一条错误可直接复用上面这串调用栈判断，不必重新排查。

验证边界：以上为静态排除，未实际运行；`0.6B 在 47 上 B 是否同样失败`是区分「模型相关」与「机器相关」的关键实验，由 kimi 的 47 对照顺带给出。未 commit、未 push。

**十二、阶段二验收记录（47：精度与性能；26：精度已验证、性能未验证）**

本节的数字全部来自 47（gfx936、BF16 KV、Qwen3.6-27B-INT8-w8a8、TP2）与 47 上的 0.6B 缺陷复现。**26 的精度结论见本节下文（A1=A2 与 A=B 文本均 8/8 逐字一致、逐层 dump 45 键张量全等）；26 的性能未验证（用户要求停止，只有一轮标定，不作结论）。**

**一、精度（47）**

```text
l2norm 迁入后 A/B 逐层 dump：layer 0/1/4/8/16/32/48/63 全部 max_abs = 0.000e+00（|1081 键）
A 侧两次独立启动逐位相同（A_m1 = A_m2 = 4f33666a）；旧基线 7919f29e 作废
确定性文本：r02、r08 在新代码下 A/B 逐字一致（8/8 请求；独立复算 id/text/usage 哈希
  r02 两侧 d3582051242f、r08 两侧 d372fb258587）
```

**二、性能（47，同配方）**

每侧 3 次独立启动（新进程，每次 5 轮），轮内算分位数后取轮次中位数，再对 3 次启动取中位数；
R19 门禁剔除 r02 单独入批等无效轮后，源为 `artifacts/kv-cpu-20261105/perf-quantitative-v3.json`。
下表由本 bot 直接读该文件的 A/B 三次启动数组重算（非引用他人汇总）：

```text
output tok/s  -0.926%   B 优      req/s      -0.926%   B 优
TTFT p50      +0.381%   B 优      TTFT p99   +1.117%   B 优
TPOT p50      +1.440%   B 优      TPOT p99   -0.421%   A 优
E2E  p50      +0.119%   B 优      E2E  p99   +0.949%   B 优

中位数差范围      -0.926% ~ +1.440%
方向               B 在 7 项上数值略优；A 仅 TPOT p99 一项略优（不得写“A 略优”）
同侧极差最大      A 0.52–2.70 pp / B 2.97–5.30 pp
结论              未观测到可分辨差异；不给差异上界；不得写成“性能等价”
```

口径变更（必须与旧表述区分）：

```text
旧判据 ≤3% pass / 3–10% warn / >10% fail  → 失效（噪声地板已大于 3%）
旧“噪声地板 2.3–6.8pp”                    → 作废（由两个不同窗口的两轮对比得出，含跨窗口环境差异）
现行判据                                 → 以同侧多次独立启动的极差为尺度；中位数差不得表述为“差异上界”
```

**三、26（目标环境）——精度已验证、性能未验证**

26 上已取得（本轮之前的窗口，本 bot 本轮独立复算）：

```text
文本   A1/A2/B1 三次运行的 8 条请求：A1==A2 8/8、A==B 8/8（按 text + prompt_tokens + completion_tokens 哈希；
       r02 99 字符/54 token，r08 430 字符/225 token）
张量   test-results/ab/dump26/{A,B}/tensors.pt：45 个键两侧全存在、形状一致、torch.equal 全等（mismatched 0）
       注意 layer.json 只含 layer_0_in / layer_63 两个哈希键且两侧相同，kimi 报的“8 层×3 行”来自 tensors.pt
性能   仅一轮标定（test-results/runs/verify26-cal-A）：该轮 8 并发 token 集 {r01 4, r02 42, r03 18, r04 6,
       r05 4, r06 32, r07 51, r08 208}，与 47 不同 → 26 性能「未验证（用户要求停止）」，不进性能结论表
遗留    layer.json 的 sha 与 fa.json 在 26 上存在首调用捕获缺陷（A 的 json sha 与自身张量重算不符；
       FA 页 1/2 不同但 layer_4 逐位相同），这两份快照不能用于 A/B 比对，已记工具 todo
```

27B TP2 仍起不来：`yuanyuan` 的 `rye_sglang_dpsk-v41` 占 119.5–120.8 GB/卡（Total 147.44 GB，余 26.6–28.3 GB），
`rest_memory` 为负 → 容量不足，非参数问题。0.6B 替代路径不可用：B 侧 fused rotary dtype 缺陷（见第十一节）。

**R18 口径（批组成）**：26 上同一请求串行 54 token、并发 42 token；47 是 33/54。这是批形状影响输出的又一次体现，
两侧行为一致、各机自身稳定，**不写成跨机差异证据**。


**推送前基线**：`HEAD = 6005118`（164 文件，+64136/−1628，分支 main，ahead 1 / behind 0，未 fetch）。
本节与交付记录的文档同步（SHA、性能 R19 口径、26 结论、R18 表述）作为其后的一次补丁提交；
push 到 origin 仍未执行，等用户明确确认。

**四、已知不透明点：插件不是透明替换**

两侧在 HCU 上都走 `rotary_embedding/base.py` 的 fallback 分支，差别在被调 kernel：A 由插件的
`HcuRotaryFallback`（`hcu_sglang/layers/rotary.py:85-107`）在调用前把 fp32 `cos_sin_cache` 转成 `query.dtype`
（带 memo、不改原张量），B 无此包装，fp32 cache 直接进入 `sgl_kernel.rotary_embedding`。
证据为插件代码 + `tests/test_rotary_fallback_dtype.py`（3 项 CPU 通过），**不是运行期取证**（两侧分支命中日志未打印）。
含义：27B 上逐位一致只说明该模型上这层转换无副作用，**不等于插件对所有模型都是 no-op**。

未 commit、未 push。

## 2026-11-06 Kimi K3.1 SituGLU W4A8 Triton MoE 支持补登（11-05 实测）

完成模块/行为：kimi-k3.1 channel-wise W4A8（`SlimQuantW4A8Int8TritonMoEMethod`）支持 SituGLU 激活。`fused_experts_impl_w4a8_triton` 新增 `activation="situ"` 分支，数学对齐上游 `SituAndMul.forward_native`（`gate = beta*tanh(gate/beta)*sigmoid(gate)`，`up` 在 linear_beta 置位时软限幅为 `linear_beta*tanh(up/linear_beta)`，fp32 计算后回投原 dtype）；`_apply_triton`/`apply_ep` 从 `moe_runner_config.gemm1_alpha`/`gemm1_clamp_limit`（上游 `kimi_k3.py` 以 `activation_situ_beta`/`activation_situ_linear_beta` 填入）管线化双 betas，缺省 beta=1.0、linear_beta=None。新参数 keyword-only 带默认值，silu/gelu 分支与既有调用点不变；非 situ 模型即使配置了 gemm1_* 也不受影响（betas 仅在 situ 分支消费）。

新增 `tests/runtime/w4a8_triton_current.py::W4A8KimiSituTests` 两项：CPU 契约测（`_apply_triton` 将 situ+双 betas 原样转交 impl；无 `gemm1_*` 的 silu 形 config 走缺省不崩）与 GPU 数值测（kimi-k31 形状 latent MoE，EXPERTS=8/LATENT=64/moe_intermediate=128，beta=1.25/linear_beta=8.0，对拍 fp32 参考实现，rtol/atol=0.03，无 CUDA 自动 skip）。

实际验证（2026-11-05，nmz26 `lizhg_k31` 容器，gfx936 单卡）：`tests/runtime/w4a8_triton_current.py` 5 项（既有 3 + 新增 2）全部通过（12.91s）。

未验证项：`apply_ep` 的 betas 管线无独立单测（与 `_apply_triton` 对称实现，靠代码路径保障）；kimi-k3.1 端到端服务、TP>1/EP 并行、其他激活/量化模型回归（仅既有 3 测）与性能均未验证。


## 2026-11-04 保留 HCU 底座已声明的 ServerArgs 字段

完成模块/行为：`hcu_sglang/server_args.py` 在安装兼容属性前，通过底座的 `field_names` 查询真实声明字段。已声明的 `minimax_opt`、`hy3_sp` 保留底座实现；仅未声明字段沿用现有兼容处理。HYGON-AI/sglang-das 的 `v0.5.22.plugin`（`15c7204b7a3b`）已声明这两个可写构造字段，原先用只读 property 覆盖后，`ServerArgs` 初始化报 `AttributeError: property ... has no setter`，GLM-5.3 CT W4A8 服务无法继续启动。

实际验证：在 DTK26.10 配套镜像中导入指定底座的真实 `ServerArgs`，执行 `ensure_choices` 后，以 `minimax_opt=True`、`hy3_sp=True` 构造成功且值保留。组合服务已越过此错误，随后因缺少 Python `soundfile` 依赖退出；构造检查日志为任务产物中的 `server-args-constructor-check.log`，服务对照日志为 `server-combined-v2.log`、`server-combined-v3.log`。底座跟踪源码未修改，未新增单测。

提交前检查使用本仓 L0、迁移规划及导览快照检查、修改文件 Python AST 与 `git diff --check`。本修复尚不代表完整模型、GPU kernel、Graph 或 P/D 服务通过；这些仍需组合环境实际验收。

## 2026-11-04 复用原生 attention 通信组

完成模块/行为：`distributed_backend.around_initialize_model_parallel` 调用底座初始化后，仅补建缺失的 `_ATTN_DP_TP` 和按需 CP overlap 组。指定 HCU 底座已建立 DP-TP 组，原来的二次初始化断言使八个 rank 全部退出；现保留该组，不改写其 rank 布局或通信器。未提供该组的底座沿用原有补建逻辑。

实际验证：DTK26.10 八卡组合服务 `server-combined-v4.log` 在该断言退出；加入守卫后，`server-combined-v5.log` 已越过并行初始化，开始加载真实 GLM-5.3 CT checkpoint。此时尚未完成权重加载、请求和 kernel 验收；CP/DP 多节点组合与 overlap 模式继续随目标 P/D 配方验证。提交范围为通信初始化守卫、本记录及导览快照，未新增单测。

## 2026-11-04 原生量化能力与 index-K 模式复用

指定 HCU 底座已提供 FP8 block reference、LightOp channel-wise helper、compressed FP8 输入能力和 INT8 scale 归一化。旧注册流程遇到这些符号就抛错，导致后续 KV pool、模型及 MTP 钩子没有执行。现保留已有能力，仅在缺失时安装；完整原生 dense-mxscale 能力存在时，整组保留其配置状态与 GEMM 消费者。两份 FP8 helper 的函数 AST 相同；INT8 归一化仅有模块限定名和 staticmethod 装饰差异。本轮仅调整注册所有权，未修改算子数学。

同时优先复用底座的 `IndexKCacheMode`，避免原生 KV pool 与 plugin indexer 对同名枚举做 `is` 比较却不相等，选错 BF16/INT8/FP8 缓存分支。底座不提供该模块时继续使用插件定义；模块内部缺依赖照常报错。

实际验证：`plugin-bootstrap-v3.log` 注册完整结束、退出 0；原生量化 helper 对象未被替换，后续 DSA pool 和模型钩子已登记，枚举三个模式与原生具有同一对象身份。该检查不加载权重，不代表模型生成通过。此前 v5 八卡 CT 权重已加载完成，warmup 失败的缓存接线问题需在完整注册后重新验证。未新增单测，完整 GPU/Graph/P/D 与其他模型精度仍待验收。

## 2026-11-04 RoPE JIT 的 HIP 向量访问

DTK26.10 的 `float2` 含匿名 union，RoPE overlay 在 GPT-J 和 NeoX 分支对它做结构化解包，hipcc 拒绝编译。改用同一向量的显式 `x/y` 字段，保留计算表达式、数据类型及布局；没有改动底座源码或量化过程。

在节点 21 直接执行实际 JIT API，对 BF16/FP16、GPT-J/NeoX、连续/带 stride 视图的 8 种组合做 FP32 参考计算并按输出类型舍入；全部通过，Q/K 最大绝对误差均为 0。证据为任务产物 `rope-jit-check-v3.log`、`rope-jit-check.json`。组合 v7 已越过 RoPE，随后停在独立的 MLA concat CUDA 头文件缺失问题。本次仅提交 RoPE 兼容修复和记录，未新增单测；完整模型及 P/D 验收继续进行。

## 2026-11-04 MLA concat JIT 的 HIP 拷贝与空批次

MLA concat overlay 的 CUDA 专用头文件和 PTX 访存阻止组合 v7 warmup。沿用 adapter 既有可移植处理：HIP 使用现有公共头、逻辑 32-lane ID 与相同类型的直接加载/存储；保留 CUDA 路径及原版权来源说明。TensorMatcher 在 HIP 校验 ROCm 设备，launch 使用 256 threads，并对零元素输入直接返回，避免零 grid 导致 invalid configuration。拼接始终只拷贝原始 BF16 位模式，没有反量化或修改激活量化。

节点 21 的实际 JIT API 检查完成 11 个输入组合：Q 的 8/64 heads、连续/带 stride、空/非空，以及 K 的广播拷贝和空输入；全部与 Torch 拼接精确一致。8 个非空组合另在 graph 捕获后改写输入、清空输出，再 replay，结果仍精确一致。证据为 `concat-mla-check-v4.log`、`concat-mla-check.json`。检查只注册所需 JIT overlay、使用实际编译核；全模型验证继续进行，未新增单测。

## 2026-11-04 CT W4A8 的 DeepEP 低延迟 INT8 分发

四节点 v3 的 prefill PP2/CP8 已就绪，decode TP16/DP16/EP16 在目标 graph 捕获时由原生 W4A8 消费者拒绝激活类型。指定底座在 `CompressedTensorsConfig.get_quant_method` 中为合格的 CT channel W4A8 + DeepEP + DeepGEMM 选择 `HCUW4A8Int8DeepEPMoEMethod`，保留 CT checkpoint 配置与加载格式，复用其 W4A8 专家实现；该方法已声明 normal/low-latency INT8 和 producer event。独立 CT scheme 的 BF16 normal 配置不是此次 DeepEP 实际入口。

首次修复 b6d8448 给低延迟 setter 补写配置并放开 DeepGEMM 下的 W4A8 vendor 分支，v4 仍由同一消费者校验拒绝。随后 bootstrap 检查确认 native/TBO 均实际引用 plugin dispatcher，但 adapter 的旧 `HCU_DEEPEP_BRIDGE` 已把 vendor `quant_type` API 包成 `use_fp8`，使 plugin 的 ABI 检测为 false。原始失败和 CPU 检查保留于 `candidate-pd/server-v4-node23.log`、`candidate-pd/dispatcher-bootstrap-v1.json`；setter AST 检查仅证明配置赋值，不证明实际路径生效。

现撤去多余的 setter 补写，保留唯一必要的 plugin 行为：已有 `hcu_w4a8_int8_dispatch` 契约可进入 vendor INT8 分支，即使 runner 为 DeepGEMM。关联 adapter 配方设置已有 `HCU_DEEPEP_BRIDGE=0`，保留原生 vendor API，让底座/plugin 处理两种 API。没有修改 checkpoint、重打 SlimQuant 标签、删除 INT8/FP32-scale 校验或关闭目标 graph；实际 P/D 复验继续进行。未新增单测。

## 2026-11-04 MTP 融合 EH norm 的 HIP launch bound

decode v5 已越过原 INT8 错误，但 MTP graph 的实际日志提示 hidden6144 的 EH norm 按 768 threads 启动，而设备代码缺省 launch bound 为 256；不能将后续 graph 捕获完成视为该算子数值有效。overlay 在 HIP 下以现有 `host::norm::get_cta_threads` 声明准确 launch bound，保留线程数、归约、权重乘法与输出拼接；CUDA 路径保留原声明。TensorMatcher 在 HIP 使用真实 ROCm 设备类型。

节点 23 通过实际 JIT API 完成 20 个组合：BF16/FP16、hidden6144 的 0/1/6/128 tokens、连续/行 stride，以及 hidden7168/8192。按底座既有 FP32 参考和 `rtol=atol=1e-2` 检查全部通过，16 个非空组合各做两次变输入/权重且清空输出后的 graph replay，共 32 次；未改变误差门限。原始 launch-bound 日志、编译日志和结果保留在任务产物 `candidate-pd/server-v5-node23.log`、`candidate-pd/fused-eh-norm-check-v1.log`、`candidate-pd/fused-eh-norm-check-v1.json`。此组件检查不代表完整 MTP、生成或 P/D 通过，未新增单测。

## 2026-11-05 MTP 校验结果的 attention DP 同步范围

组合 decode v7 完成 EAGLE 5/1/6、target verify/draft decode/draft extend graph 捕获和 16 个 DP warmup。首条固定 33-token 随机采样请求中 prefill 已返回，decode 的 16 个 scheduler 停在结果事件同步；已保留各 rank 的 py-spy 栈、原始请求与日志。底座 `eagle_sample` 对 idle batch 在同步之前返回，而 `_sync_eagle_verify_outputs` 按 `parallel.dp_size == 1` 选择整个 TP 组。当前底座将 replica DP 与 attention DP 分开，实际配置是 replica DP1、attention DP16；只有接收该请求的 attention DP rank 进入校验广播，其余 rank 不参加该 collective。

在现有 distributed backend 通过正式 AROUND Hook 为 HCU attention DP>1 选择底座已有的 attention TP/CP 组及同样的广播顺序、src0。独立 attention DP 的请求不共享校验状态；TP/CP 组内仍同步完整 packed verifier state。非 HCU、未启用 DP attention、attention DP1 保留当前有效原实现。没有改变采样、接受判定、随机数、量化或组的构造；SGLang 跟踪源码未改。新增 Hook 同步到当前目标合同；上游按 attention DP 正确区分同步范围后可删除本补丁。

实际底座 helper 经插件发现和 HookRegistry 应用完成 6 个 CPU 组选择检查：覆盖当前 replica DP1/attention DP16、TP4/CP2、replica DP2、attention DP1、关闭 DP attention 和非 HCU。当前配置的原 helper 选择 full-TP 广播，修复后不跨独立请求广播；TP/CP>1 保留同一 packed state、src0 和顺序。已缓存别名随 Hook 更新，公开 `eagle_sample` 对象未变。该检查只用 CPU recorder 替换 group handle，不能证明 GPU collective 或模型精度；结果为 `candidate-pd/eagle-sync-check-v2.json`。首版探针未调用插件发现，其断言失败已保留，未作为运行时反例。项目 L0、导览/迁移快照、Python 语法和 diff 检查通过；完整 GPU 请求和候选 32-rank profile 尚待复验。本次未新增单测，也未执行完整本地测试套件。原始 v7 阻塞证据位于任务产物 `candidate-pd/hang-v7/`。

## 2026-11-05 GLM CT W4A8 的 BF16 RoPE 选择

当前固定 HYGON v0.5.22.plugin 底座在 adapter 的 CUDA 兼容语义下构造 `use_fallback_kernel=False` 的 `RotaryEmbedding`，原有仅包装 fallback 的 Hook 没有覆盖此分支。真实首层 Q/K/位置输入相同，但原生与候选 RoPE 输出不同；两组八 rank 的原生输出可由 BF16 cache 与 BF16 逐步舍入精确重建。

新增默认关闭的 `SGLANG_HCU_USE_LIGHTOP_ROPE`，在 HCU BF16 构造后准备已安装 LightOp 公共接口，复用 typed cache memo，并通过正式 `forward_cuda` AROUND Hook 消费；原始 FP32 cache 保留给框架和 indexer。由 GLM CT W4A8 配方显式启用，不改变其他配方默认值。融合 KV-store 调用保留当前有效原实现。源码不写回框架或 vendor 包，不新增算子实现，不修改量化方案。

第一版复用旧 head 分块的插件探针失败，未进入模型服务。进一步同输入对照确认：两组八 rank 的公共 LightOp 完整 64-head 调用均精确匹配原生，拆成 8-head 调用则有 23861 个输出元素不同。因此正式路径保留 LightOp 自有分发，不沿用为旧 sgl_kernel host 编译边界设置的 head 分块；记录为 `rope-lightop-component-v2.json`。

DTK26.10/Python3.12 镜像中的真实插件发现、构造和 AROUND 分发已通过：两组八 rank 的 Q/K 共 32 项逐元素匹配原生；同一 Graph 更换输入后重放两组 Q/K，也均逐元素匹配。FP32 canonical cache 对象未变，typed memo 为 BF16，关闭开关后的构造保持原分支。记录为任务产物 `rope-runtime-v4.json`；固定参考和捕获哈希见 adapter-das `docs/glm5.3-w4a8-attention-alignment.json`。offsets 与融合 KV-store 保留原语义，但本轮组件不声称覆盖这些调用。移除观测后的 P18/D10 六组模型请求完成，但全部六组 P 与六组 D 数值仍失败；组件通过不等于模型精度通过。

## 2026-11-05 GLM CT W4A8 的 W8A8 GEMM 分发

继续以首层 Q-b 实际相同的 BF16 输入、INT8 权重和 FP32 scale，在两套镜像重放 M8/N16384/K2048 的两组八 rank。公共 LightOp 输出均逐元素匹配原生捕获值；当前插件自行选择的 smooth ASM 路径则逐元素匹配候选捕获值，并有四个元素与原生不同，最大 rank NRMSE 约 8.2e-6。此前 dense MLP 的对照通过不能外推到此形状。记录为任务产物 `attention-probe-v1/qb-w8a8-v1/`。

新增默认关闭的 `SGLANG_HCU_W8A8_USE_LIGHTOP`，由 GLM 配方显式启用，仅跳过既有接口中的 ASM 快路径，复用公共 LightOp 实现。没有改变量化整数、scale、权重布局、输出 dtype 或其他配方默认值。两镜像共 48 份共同量化整数、token scale 和公共 GEMM 输出逐元素一致；真实插件分发的 16 组输出及两次更换输入的 Graph 也逐元素匹配原生，关闭开关后 16 组输出均保持原候选路径。记录为 `qb-w8a8-v1/cross-image.json` 与 `candidate-runtime-v2.json`；后续六组模型数值仍待复测。

## 2026-11-05 GLM CT 的 BF16 gate 与 LightOp 专家路由

P23 两组短前缀各自八 CP rank 的 248 项早期 norm 张量对照均逐元素匹配原生，首个差异出现在第 3 层 MoE 输出；该层输入及下一层的 residual 一致。实际 gate 权重与输入为 BF16，当前 tiny GEMM 返回 FP32；固定原生 HIP CP 分支使用 BF16 `F.linear`。用同一输入和 checkpoint 参数在两镜像重放投影及公共 LightOp 路由，80 份输出张量逐元素一致；BF16 投影等于当前 FP32 logits 的 BF16 舍入，但会改变 65 个有效 token 中 25 个的专家集合。此处是来源分支的受控重放，未单独捕获原生 gate 中间值，不能据此声称 MoE 或全模型精度通过。

新增 `SGLANG_HCU_GLM_CT_BF16_ROUTER`（布尔，默认关闭），在 HCU GLM DSA compressed-tensors gate 构造时生效，为 BF16 权重选择公共 BF16 `F.linear`。FP32 权重沿原有优先分支；无量化、其他模型、Hash-MoE 与 DSV4 不启用该策略。开关关闭后保留当前 gate 实现。已有 `SGLANG_LIGHTOP_TOPK`（默认关闭）接入已有静态 bias、归一化分组路由的公共 LightOp 入口，保留其原有分派优先级；helper 懒加载依赖，动态 bias 或不归一化的调用保留已有路径。GLM 配方显式开启两项；没有修改 checkpoint 标签、专家 W4 与动态 INT8 激活的量化语义。

先前只更换 BF16 gate 的正式探针保留失败结果：专家 ID 相同，但首组 Triton 路由权重有 15 个元素不同，最大绝对差约 6e-8。因此继续复用公共 LightOp 路由，正式分发与 Graph 的结果见下，六组模型验收仍待复测。类型注解改为已导入模块的限定名称，消除原文件两个 F821；未删除动态引用或新增单测。

正式入口复验还发现底座已注册同名 custom op，重复注册会被跳过并落回未导入 LightOp 的底座 helper。插件桥接改用独立的 `sglang::hcu_moe_fused_gate` 名称及既有 fake shape 契约，公共 LightOp 计算与参数保持不变。两个开关在启动脚本中设置，Graph 保留捕获时的 provider，不提供运行中的热切换。

三个固定开关的独立进程完成对照：旧版关闭与新版关闭的 48 份真实输入输出逐元素一致；独立上下文与先前服务捕获之间的少量 FP32 权重 ULP 差异同时存在于旧版，不能归为新回归。最终启用探针 v7 的 16 组 logits、路由权重和专家 ID 共 48 份输出逐元素匹配来源分支及公共 LightOp 参考，两次更改真实输入与有效 token 数的 Graph 重放也精确匹配。无量化、Hash-MoE、DSV4、其他 architecture 的四项构造门控及 FP32 权重优先分支通过；范围检查曾错误保留 GLM architecture 的夹具和先前失败均留档。722 份运行时资产在执行后再次校验一致。组件记录为任务产物 `moe-gate-v1/qualification-v7.json`、`routing-v6-cross-baseline.json` 与 `routing-v7-new-on.json`；全模型精度、有限验证矩阵和最终 profile 仍未验收。

## 2026-11-05 VP 与 LayerSplit 功能归属复核

社区基线为 `baa9befa381511cd4d56f83980951e5a45fffc6d`，DAS 来源为 `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`，plugin 整理起点为 `9faa706d8b8c1ebcec680395634fe21ed2fd0f15`。新增机制归 plugin，社区已有行为的 HCU 兼容归 adapter；社区源码保持原样。

VP 主体和 EAGLE 初始化、消费仍在 fork，plugin 输出字段不构成完整迁移。固定 Git 对象核对确认：社区基线已包含基础 DSA LayerSplit 池及 CP helper，不能把整个模块视为 fork 专有。只审查 DAS 新增的 MainKVPagePlan、紧凑布局、scratch 和传输差异；工厂、模型消费者及生命周期仍须逐项核对。对应路径在 plans/sglang-das-decisions.json 标为 in_progress，运行对齐为未验证。

本轮独立工作区仅整理归属台账，未修改运行时代码或部署。先补齐实际调用、数学语义和 graph 变输入重放证据，再做最小迁移；既有局部检查不能关闭完整功能链缺口。

### 全量盘点工具与范围修正

整理范围覆盖两仓全部代码及固定 fork 与社区的完整差异，VP/LayerSplit 仅为首批审阅条目。build_migration_plan 原来只扫描旧包名 sglang_hcu_plugin 和已移除的 targets_v0_5_22.json，现兼容当前 hcu_sglang 与 expected_hook_targets 合同，并保留历史包名和列表格式。这是盘点工具修复，未改变推理行为。

重新采集得到 2028 个修改路径、903 个 fork 独有路径、79 个社区独有路径；运行时待审阅路径 1614 个，合同目标 599 个，源码映射线索 473 个。目标匹配到来源文件不等于 Hook 已应用或功能验收，均不可据此计算迁移完成率。完整 Python AST 对照覆盖记录中有 1494 项差异、26 项等价和 14 项文件模式或种类差异。

固定 Git 树对照已完成；本地浅历史无法提供 merge-base，历史归因清单仍不完整，不伪造共同祖先。四仓审计、当前树差异和新版全量盘点为独立产物。运行时改动须先满足实际调用与数学语义对齐要求。全量整理完成后，将通用流程写入适用 AGENTS.md，固定提交交主对话验证，通过后再提 PR。现有两个草稿 PR 保留待逐项处置。

本次实际检查：全量盘点 CLI 成功；固定 Ruff 0.12.1 的 CI 策略检查、Python AST 语法检查、迁移执行台账快照、离线导览快照、L0 和 git diff --check。未新增单测；没有用工具检查结果替代运行验证。

## 2026-11-05 VP 原生来源固定

按用户指定，VP 迁移以原生镜像实际实现为准，源码参照为 [maxiaobetter/sglang-das 的 glm5.3-w4a8 分支](https://github.com/maxiaobetter/sglang-das/tree/ac17298aeae1b149a17c4ce964619c782832d6a4)。固定提交与逐文件 SHA256 保存在 [VP 来源记录](../plans/glm53-vp-source.json)。从节点 23 原生容器只读提取的 VP 核心、EAGLE worker、logits processor、speculative 参数校验四个文件均与该提交逐字节一致。

镜像的 worker 显式传入 TP / 节点数作为 local_world_size，并持有 draft_lm_head_vp 状态；不能用之前 HYGON 盘点基线的另一份实现替换。全量 HYGON / 社区差异清单保留其原始基线，VP 的实际迁移来源单独固定，不能将两个来源混作同一版本。

此次完成来源核验，未修改推理代码，未验证迁移后的分发、数值或 graph。后续须以这份镜像实现核对参数入口、状态、collective、logits 消费及 graph 重放，再形成 plugin 运行时提交。

## 2026-11-05 VP 插件迁移候选

新增 `hcu_sglang/speculative_vp/`，归属 plugin，未改社区 SGLang 或 adapter 运行时。核心 `core.py` 与上述镜像文件逐字节相同，保留节点内分组、权重分片连续转置、固定 gather 缓冲区、RCCL 捕获前预热、scale/softcap 顺序与确定性 top-1 合并。

接入链为 CLI → plugin 进程配置 → 配置解析后约束检查 → EagleDraftWorker.init_lm_head → plugin 核心 → LogitsProcessor.forward → EagleDraftWorker.draft_forward。命令行沿用 `--speculative-draft-lm-head-vp-size`，由 `HCU_PLUGIN_DRAFT_LM_HEAD_VP_SIZE` 传给子进程，默认 1；没有修改社区冻结配置字段。社区 attention DP 字段为 attn_dp_size，因此将镜像的旧 dp_size/enable_dp_attention 检查映射到该字段，未放宽 TP=attention DP、CP1、PP1、节点内整组等约束。

初始化在当前有效 init_lm_head 完成共享后读取 draft 模型的 lm_head.weight，worker 与 logits processor 持有同一 VP 实例。LogitsMetadata 的 AFTER 钩子携带原生 spec_input_type 门控；只有 EAGLE_DRAFT 的 decode/idle 使用 VP，prefill、draft extend、普通 LM-head 与关闭状态回到当前有效 original。保留社区 LayerNorm SP 的 gather。输出复用既有 plugin LogitsProcessorOutput，top-1 的紧凑分数不作为完整词表概率使用。

消费者基于固定社区 draft_forward 保留当前接口，只补入镜像的 VP 三处行为：关闭 CUDA topk1 链直接写入、消费 VP token/probability 并递增位置、按真实词表大小检查 token 范围。拒绝采样、动态步数、多层 EAGLE、LoRA、FP32 LM-head、token map 等不支持组合仍拒绝。Graph 继续经 worker 的 draft_forward 调用；未添加另一份 graph runner。

验证状态：源码一致性及静态接口核对已执行；本地 CI 等价 lint、语法与结构门禁记录见本次提交。真实 collective、所有参与 rank 的 idle/non-idle、同输入输出对拍、更换输入 graph replay、初始化共享权重的实际身份及元数据旧引用仍未验证。此候选不得直接替换正在进行精度排查的主对话环境；先补齐上述组件证据，再做整模型验证。全量归属整理仍在进行，VP 候选不代表 LayerSplit 或其他功能已经完成。

## 2026-11-05 LayerSplit 接线与镜像增量候选

核对节点 21 原生容器的 DSA LayerSplit 池、KVCacheConfigurator、ModelRunner 和 CP helper，四个文件与用户指定 ac17298 分支逐字节一致，见 [LayerSplit 来源](../plans/glm53-layersplit-source.json)。既有 plugin 池方法与镜像逐项 AST 对照，除两项环境变量能力缺失及三项社区接口兼容外，方法 AST 相同；该检查不是执行验证。

新增 `mem_cache/dsa_layer_split_integration.py`，把 EAGLE alloc_memory_pool 前的 scratch-source 生产落到 plugin，保留镜像的 prefill、PP1、单层 EAGLE/NextN、非 EAGLE3、目标池类型限制。ModelRunner.init_kv_cache_configurator 通过作用域 ContextVar 将源池传给 plugin 的 configurator 子类，构造结束无论成功失败都 reset；子类用独立字段持续持有源池，不在 slotted 社区实例上动态添字段。替换 configurator 定义和 ModelRunner 已导入别名，避免调用者仍实例化旧类。

共享源存在时，draft 工厂沿当前社区参数接口构造 plugin LayerSplit 池，传入源池与 CP size - 1 的 owner offset；这与镜像的最后一个 CP rank 持有单层 draft 一致。无共享源时调用当前有效父类工厂。已有 CP 预算 helper、scratch geometry 校验、stream 等待、前一使用者缓存失效和独立 Index-K 缓冲区继续复用。没有移动社区基础 LayerSplit，也没有改动社区源码。

池中另补齐镜像原方法的两个缺失分支：`SGLANG_HCU_COMPACT_INDEX_BROADCAST=1` 的历史 Index-K page gather/broadcast/scatter（含 payload record_stream），以及 `SGLANG_HCU_LAYER_SPLIT_SYNC_PREFETCH=1` 的当前流预取。初次迁入时两个变量沿用镜像默认 0；随后按用户要求调整，当前默认见下一节。set_mla_kv_buffer 的 WriteLoc 解包、CPU copy 的 req_pool_index 参数保留社区兼容。

本轮只执行本地 CI 等价 lint、语法、快照和结构检查，不创建单测、不启动服务。真实类绑定、所有 CP rank 的 collective、共享 data_ptr、流切换失效、P/D 层映射及适用 graph 尚未验证，必须先完成组件对齐再交接整模型复测。VP 和 LayerSplit 均是候选；全量归属整理与最终 AGENTS.md 规则落盘仍未完成。

## 2026-11-05 LayerSplit 开关默认开启

按用户要求，plugin 中 `SGLANG_HCU_COMPACT_INDEX_BROADCAST` 与 `SGLANG_HCU_LAYER_SPLIT_SYNC_PREFETCH` 在未设置时均按 1 处理；显式设置为 0 仍可关闭各自分支。不修改运行中节点的环境。紧凑广播仍要求 Index-K remote buffer 和有效 page plan；共享 scratch 的 draft 沿用原有当前流约束，即使同步预取变量为 0 也不取消该约束。

这是相对镜像默认值的明确差异；原生源码 SHA 记录保留，不再宣称这两个方法与镜像 AST 完全相同。本次执行 CI 等价 lint、Python 语法、结构和快照检查、git diff --check；未运行 GPU/graph 或整模型测试。

## 2026-11-05 CT W4A8 分发与计算归属候选

全量导入检查发现，已有 plugin 的 CT 权重方法仍由 fork 的 CompressedTensorsConfig 选择，DeepEP 计算也留在 fork。仅有权重类不能证明社区 SGLang 的路径完整。本次在隔离工作区将该方法移动到 `layers/quantization/compressed_w4a8.py`，旧 SlimQuant 模块只保留导入别名；新入口只匹配 HCU、DeepEP、DeepGEMM、pack-quantized、对称 channel INT4 权重和动态 token INT8 激活。其他配置、无法匹配的投影名称回到当前有效 original，不借用 DSV4 的 checkpoint 别名推断 GLM 语义。

`layers/moe/ct_w4a8.py` 承载 reference normal HIPC、normal Marlin 和 masked 计算，DeepEP core Hook 按权重方法标记接入。保留 CT nibble 顺序转换、offset-8、scale/16、INT8 dispatcher 输出及 FP32 token scale、256 expert alignment、两次 GEMM、激活量化位置、masked producer event；不引入 W4A16 回退。Masked custom op 使用 plugin 专有名称，避免覆盖框架同名注册；vendor API、参数和 fake 输出语义保持来源实现。

代码来源固定为 HYGON `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e`，与 VP/LayerSplit 的镜像来源分开记录，见 [CT 来源记录](https://github.com/maxiaobetter/sglang-plugin-das-fork/blob/feat/glm53-ct-w4a8-moe-20261107/plans/glm53-ct-w4a8-source.json)。normal HIPC 与 masked reference 分支在排除导入路径和 custom-op 重命名后 AST 相同；权重类除导入位置、注册入口外计算未变。该结论不表示容器安装库相同或数值已对齐。对生产者量化中间量、两个 GEMM、combine、空 token、多 rank、MTP、graph 变输入重放仍标记未验证。

同时将 P/D transfer 的两处导入指向已有 plugin 模块，Bailing 的 parallel-state 导入改为社区现有位置；DeepEP FP8 模块去掉整体复制 fork globals，改用明确依赖。没有改动社区源码、主工作树或节点环境，没有通知主对话切换代码，没有提交 PR。

全量整理尚未完成：静态检查还发现 V4 row transport、row-bucket graph、V4 Mega backend 和 GLM5-next 的 fork-only 依赖。不能将它们改个导入名称就记作迁移完成，也不能把这些其他模型路径误记为当前 GLM5.3 的实际算子。继续按来源、消费者与功能门控逐项核查。

本次实际本地检查：Ruff 0.12.1 CI 规则 E9/F63/F7/F82 对本轮累计 20 个 Python 文件通过，AST 语法通过，L0（565 个 Python 文件）通过，migration plan/execution 快照与导览检查通过，git diff --check 通过；未新增单测、未运行 GPU。全量导入复查剩余 10 处 fork-only 模块引用，其中包含有保护的历史兼容分支；该扫描只验证模块存在性，不验证属性或实际执行链。

通用整理和交接约束已先写入两仓隔离工作区的 AGENTS.md，防止后续整理继续扩大职责混杂。规则落盘不代表全量工作完成。当前仍不满足交接主对话切换和整模型测试的条件。

## 2026-11-05 GLM5.3 范围收敛与依赖整理

用户明确“不管其他模型”。本次收尾覆盖非 Flash GLM5.3 CT W4A8 的全部适配链路；其他模型的 fork-only 依赖不再作为本任务阻塞，撤回本轮 Bailing 导入调整，保留其原始基线。完整归属和验证顺序见 [GLM5.3 整理交接](glm53-w4a8-ownership.md)。历史全仓盘点及其 pending 状态不改写。

P/D transport_common、transport_mooncake 和 DP attention 去掉整模块 globals 注入，明确声明所用依赖；保留 plugin 自有协议类及传输 helper 优先级。P/D worker 显式导入自身依赖，直接注册原函数；scheduler 经当前有效 native 命名空间访问其宿主依赖，直接调用 plugin 元数据解析，不再给宿主注入解析函数或重建 FunctionType。两者经命名空间规范化后函数 AST 与本轮起点一致，传输等待、完成通知、异常处理与调度顺序未改。

DSA 条件算子使用宿主模块的明确限定引用，保留原有平台门控；CP、forward batch、VP 邻接 graph 和 DeepEP 模块补齐类型依赖，类型导入置于 TYPE_CHECKING，不增加模型导入环。没有更换量化、内核或数学策略。

本次最终本地检查：累计 39 个改动 Python 文件通过 Ruff 0.12.1 的 E9/F63/F7/F82（ignore-noqa）与 AST 语法检查；plugin L0（565 个 Python 文件）、migration plan/execution 快照、导览快照和 git diff --check 通过。adapter 本轮仅新增 AGENTS.md，L0 通过并保留 31 条原有 soft warning。未新增单测、未运行 GPU/模型测试，未推送或创建 PR。GLM 范围的代码整理交付与实际组件对齐分开验收；下一步交固定提交给主对话按交接清单测试。

## 2026-11-05 GLM DSA Indexer head-gate 同输入分歧

固定 adapter `7ad53f65`、plugin `7972101a`、HYGON SGLang `15c7204b` 的独立 wheel，使用两个归档请求、8 个 CP rank 的真实 layer-0 Indexer 输入和原 checkpoint 参数，对拍原生镜像。16 组有效输入的 2080 个 head-gate 值均有差异：原生投影及编译后的缩放结果为 BF16，候选由于 adapter 的 CUDA facade 进入 FP32 投影分支。两镜像中同一个 BF16 ReplicatedLinear 投影完全一致，分歧定位在 Indexer 的平台分发与舍入位置。详见[组件证据](glm53-indexer-head-gate-alignment.json)。

本次最小修正限定 HCU 的 GLM DSA main/NextN 架构，先调用原有 BF16 weights_proj，再执行原有缩放；保留显式 GLM-next FP32 分支及其他架构路径。没有改公共平台判断、量化方案或 vendor wheel。相同 16 组 fused LayerNorm/RoPE 输入在两镜像与各自实测 cache 下完全一致，未据 cache dtype 差异改动该算子。

修正前两镜像各完成 6 次变输入 graph replay，与各自 eager 一致；这不能证明跨镜像数学对齐。修正 wheel 的同输入及 graph 复验待执行，构造接线、完整 Indexer 和 decode attention graph 仍未关闭；不宣称整模型精度问题已解决。新分支 `fix/glm53-indexer-head-gates` 从固定 plugin 提交隔离建立，原交接分支与两份已安装 wheel 均保留。未新增单测，整模型复测仍等待算子清单关闭。

修正后组件复验：runtime 提交 `a587338d` 的独立 plugin wheel（SHA256 `17060beb…ee827c3`）在相同 16 组真实输入上，投影原始值及编译后的缩放值均与原生逐值一致；6 次变输入 graph replay 与对应 eager 一致。实际构造器字段接线和服务消费者仍未验证，不能外推为完整 Indexer 或整模型验收。33 个原始输入、输出、源码与报告文件已归档并在本地逐份核对哈希，归档标识见组件证据 JSON。修正代码通过 Python 3.12 的项目 L0、migration/guide 快照、Ruff 0.12.1 CI 选项、AST 与 diff 检查；没有运行新增单测或完整本地测试套件。

## 2026-11-06 GLM NextN EH 原生 HCU 分发修正

同一份服务实际输入表明，NextN 的 6 行 draft-extend EH 输入在原生 HCU 的两次 LightOp RMSNorm 加拼接，与 CUDA facade 下融合 EH 分支之间有 2 个 BF16 输出值不同，最大差值为 0.000244140625。此前单行样本通过不能覆盖这条路径。原始输入、失败对拍和已安装文件哈希均保留，见 [EH 分发证据](glm53-mtp-eh-route-alignment.json)。

本次仅在 plugin 自有 NextN 构造器记录物理 HCU、GLM MoE DSA 架构和 compressed_tensors 配置判定，forward 据此恢复原生 enorm、hnorm 和拼接路径。其他配置沿原有分发；不更改公共平台标记、量化方案、RMSNorm 实现或权重。

直接执行自有 forward 到实际 EH projection 边界，使用原服务采集的输入、权重和布局：16 个 rank 共 320 例、160 次实际更新输入 graph 重放，EH 输出及 BF16 projection 均与原生 LightOp 路径逐值一致。最终构造门控版本另通过两个实际活跃 rank 的 20 组输入和 10 次更新输入 graph；GLM 配置命中、非 GLM 与无量化配置不命中。Ruff 0.12.1 CI 选项、Python AST、项目 L0、migration/guide 快照及 diff 检查通过。固定运行时提交 `ae3160f` 的 wheel（SHA256 `bba95325…a74b736`）安装后，经公开构造器与公开 forward 再验 20 组实际输入、10 次更新输入 graph，通过；729 个插件运行时文件中仅本次模型方法文件不同。四节点按原 P(PP2/CP8) → D(TP16/DP16/EP16/VP8，EAGLE5/1/6) 重启，每节点启动前核对 6887 个运行时文件，仅启用 hcu。最终以 curl 直接向 decode 发出冷请求（DP0，4095 输入/64 输出）和热请求（DP8，4096 输入/64 输出，P 命中 4032 token 缓存），两次执行真实 MTP；16 份 prefill 和 16 份 decode 原始 trace 均通过 gzip、JSON、rank 覆盖和本地哈希校验，诊断 hook 事件为 0。原始 ROCTracer duplicate-flow 警告保留，不据此判定 FlashMLA 事件缺失原因；第 2 项按用户分工交由用户查看 trace。未创建单测，未开展整模型精度复测。


## 2026-11-06 GLM5.3 原生 profile 调用接入

对照实际安装的原生 `e37bbf` SGLang 及 16-rank decode profile，修正 CT W4A8 DeepEP recv-hook 模式、masked SiLU 量化 expect_m 参数；将 GLM 主注意力/Indexer 的 RoPE 与缓存 dtype、DSA MLA 拼接接至镜像既有算子。来源、调用链、形状限制、加载归属与逐项状态见 [原生算子分发接入](glm53-native-operator-dispatch.md)。保留 W4A8 权重和 INT8/FP32 scale 契约及 producer-event 依赖，不改环境变量或在线服务。

按用户本轮明确要求跳过同输入对拍、GPU/graph、整模型和性能复验；本轮只做项目 lint、语法和 diff 检查。代码接入与运行对齐分别记录，不宣称 kernel fault 或异常耗时已经修复。


## 2026-11-06 GLM profile 异常耗时与多流门控

对全部 32 份 trace 按 graph 和 stream 分析，adapter 多组相邻算子的长短耗时呈稳定错位，不能据原始名称断言 RMSNorm/SiLU 本体退化。真实额外延迟主要落在 MoE dispatch 到 combine 区间；并观察到一个 rank 约 35ms 停顿与其余 15 rank 的 dispatch 长等待对应。负载分别是 33 与 1 个逻辑请求，保留负载差异和计时关联的解释边界。

实际源码确认 CUDA 兼容标记使 GLM 在 ROCm 多流开关为 0 时仍继承备用流。主 attention/Indexer、MoE、NextN 构造器现在按物理 HCU 与 GLM 架构恢复原生流使用门控，保留主 attention prefill CP 和显式多流分支。分析方法、数据与限制见 [kernel 耗时追查](glm53-kernel-timing-investigation.md)。本次遵循用户要求，不做 GPU/graph/整模型测试，不改环境、不切换服务，仅执行 CI lint、语法与 diff 检查。


## 2026-11-07 GLM INT8 index-K 容量计费

DTK 26.10 的 GLM-5.3 W4A8、TP16/DP16/EP16、EAGLE 5/1/6 服务在全部 Decode rank 实际分配 `int8_scaled` index-K：每层每 token 持久化 132 字节，每个池另有一份 BF16 解量化工作区和 page-claim 数组。旧 `around_compute_dsa_indexer_cell_size` 仅根据 gfx936 缺少原生 FP8 能力，错误进入 BF16 计费分支。主模型 21 个完整 indexer 层按 5376 字节计费，原生为 3029；MTP 按 256 字节计费，原生含工作区为 389。主模型与 MTP 总预算从每 token 55242 字节被放大至 57456 字节。

修正先按池分配使用的 index-K mode resolver 确认模式；INT8 模式委托当前有效框架计费函数，保留工作区、page claim、层共享和 MTP 处理。既有 BF16 兼容分支保留，不修改存储格式、量化计算或服务环境变量。

固定实际安装源码及模型配置的主机侧方法重放复现旧差异；修正后 INT8 主模型/MTP 计费与原生一致，BF16、FP8 和非 HCU 分支结果保持不变，诊断进程没有初始化 GPU。该证据使用显式运行上下文替身，不代替公开 Hook 接线和服务重启验证。在当前 98560 token 容量对应的相同预算区间内，修正后的预计容量为 102464–102528 tokens/rank；实际值待重启测量。在线 GPQA 评测保持原部署，本修正尚未部署，整模型精度与性能不能归为本修正验证。额外 scale 存储和其余显存差异另行追踪。


## 2026-11-07 GLM main 整合的运行态差异复核

在独立 main 整合工作区保留上游新增能力，并核对在线 DTK 26.10 site 的实际源码。CT W4A8 SiLU 量化显式保留在线已验证的 `expect_m=-1`；两次 GEMM 的 expected_m 不变。KV 计费修复已通过源码方法重放与项目 lint，但尚未部署。RoPE BF16 canonical cache 和原生 kernel 分发虽已在候选代码中，当前在线 site 尚不包含；在线精度/性能结果不能直接作为最终 main 组合的验证。

在线 16 rank 的 MEM-only allocator 快照均存在各 76 个 2.25/2.375 MiB scale 分配，与 304 MiB 尾部保护一致；也存在各一个与 FP32/BF16 RoPE cache 大小吻合的分配。分配尺寸吻合尚不能代替张量地址绑定，启动阶段原生/候选对照仍待完成。MEM-only 导出造成约 36–40 秒请求处理停顿，该区间不能用于性能验收。


## GLM W4A8 KV 存储修复（2026-11-07）

本次使用 hcu-plugin-dev 流程，在已部署的 0.5.22.plugin 组合上处理三项明确差异。
INT8 index-K 预算与实际缓存模式使用同一个 resolver，保留 BF16 回退与 INT8 workspace 预算。
GLM 主注意力和 indexer 共享 BF16 canonical RoPE cache，保留已部署的 RoPE 算子分发，避免长期保留 FP32 cache 和 BF16 副本。
CT MoE 的紧凑 scale 仅限 GLM DSA、16 个本地 expert、4096/6144 输出通道及 HIPC/Marlin masked 布局；保留 scale/16、FP32 scale 和 W4A8 权重布局，其他组合继续保留尾部空间。

本次用户明确要求跳过组件对拍，直接进行整模型精度评测。因此，新增 scale 存储和 RoPE 组合的同输入数值、变输入 Graph 对拍标记为未执行，不能引用旧环境的组件结果作为本次通过依据。隔离部署保留原运行版本用于回退；启动、KV 容量及 GPQA 结果待实测记录。

启动检查发现旧候选直接引用的 torch.ops RoPE 符号未在 DTK26.10 组合中注册；本次存储修复因此不引入该分发修改，只统一 canonical cache dtype。


DTK26.10 隔离服务启动记录：D 端 16 rank 的实际预算 cell_size 为 55242 bytes/token，分配容量为 109440 tokens/rank（旧部署 98560，历史 DTK26.041 原生 113792）。模型对象清单确认主模型 150 个 MoE scale 的 storage bytes 等于 logical bytes，RoPE canonical cache 为 BF16、1048960×64，未保留 FP32 canonical cache。P 端实际服务容量为 4184256 tokens。紧凑 scale 的 16-expert 门控只覆盖本轮 D 布局；P 端 32-expert 布局仍保留原保护空间。

旧部署评测在切换时保留 192 个完整评分记录：177 正确、15 错误，准确率 92.1875%；剩余题目中止，不能作为完整 198 题成绩。修复版使用相同 198 题、temperature=1、top_p=0.95、seed=42、max_tokens=65536、并发32，完整精度仍待新结果。


### 后续基准与 main 整理

当前完整 GPQA 正在运行；固定版本的 GSM8K（1319 题、4-shot）、HumanEval（164 题、0-shot、pass@1）和 HMMT Nov 2025（30 题、0-shot）已准备，等待 GPQA 完整结束后依次执行。沿用 thinking、T=1、top_p=0.95、seed=42、65536 输出上限、并发32，每题一次生成。HumanEval 保持 EvalScope 的代码拼接、抽取和官方测试，执行端改为无网络、无宿主挂载、无 GPU 的独立 CPU 容器。数据来源与版本、运行状态保存在本次验收产物；不能将排队状态写成精度通过。

main 整理已同步正在评测的三个 KV 修复，并移除未部署的 raw torch.ops RoPE/MLA 拼接替换；已上线 JIT 拼接与 LightOp RoPE 保持原路径。main 现有其他变化仍须与固定 0.5.22.plugin 部署区分，当前整模型结果不自动代表 main 组合已验收。
## 2026-11-07 Phase 1 去重：fla×2 hook 退役 + W8A8 create_moe_runner 双注册收敛

**fla/GDN hook 退役（A1/A2）**：`hcu_sglang/__init__.py` 不再 import/注册
`layers/attention/fla/linear_attention.py` 与 `layernorm_gated.py`（合计 5 个
REPLACE 目标：chunk_delta_h.chunk_gated_delta_rule_fwd_h、chunk_o.chunk_fwd_o、
fused_recurrent...packed_decode、layernorm_gated._layer_norm_fwd、
gdn_triton.TritonGDNKernel.target_verify）。依据：adapter gdn_aiter 生产级路由已
承接 fwd_h/packed decode（官方 inplace_update 签名兼容），chunk_o/layernorm 上游
triton 即正确路径，target_verify 与前四项同族且已在组合栈验证窗口随整模块禁用
通过（247 tok/s / GSM8K 96%）。`*_source.py` 保留（tests/runtime/
fix_attention_models_gpu.py 直接对拍引用）；l2norm hook 属 #20 新迁移（vendor fast
path），不在本次退役范围。契约 `hcu_source_alignment_targets.json` 同步摘除 5
目标（636→631），新增 `tests/test_fla_gdn_hooks_removed.py` 钉死（入口不引用、
契约不列出、l2norm 保留三项断言）。

**W8A8 create_moe_runner 双注册修复**：w8a8_aiter 与 w8a8_lightop 原本对
W8A8Int8/W8A8FP8 两目标各自注册 AROUND（"Multiple AROUND hooks" 告警、隐式链序
依赖）。新增 `layers/quantization/w8a8_runner_dispatch.py` 单点注册：lightop 请求
时走 lightop 臂，否则 aiter 臂（两臂原有 guard 不变，均含 original 回退）；
两个臂模块移除 w8a8 目标的 create_moe_runner 注册，compressed 目标（原本就单
注册）留在 lightop。新增 `tests/test_w8a8_runner_dispatch.py` 5 项：AST 结构
（臂模块零注册、dispatcher 恰一次、compressed 留守）+ 行为（lightop 优先 /
aiter 回退，mock 臂隔离）。

**本地验证**：`python -m unittest tests.test_fla_gdn_hooks_removed
tests.test_w8a8_runner_dispatch -v` 8/8 通过；全量 L0 + unittest discover 见
提交记录。容器/GPU 组合 e2e 已跑（2026-11-07，nmz26 debug_lzg26，GPU0,1 tp2，
官方 v0.5.22 baa9bef + adapter-das feat/l15 8692273 + 本分支 afef8ad 隔离 wheel）：
服务起服（prefill/decode/verify 图全捕获），create_moe_runner 双注册告警归零，
fla 族 plugin REPLACE 零注册、adapter gdn_aiter 接管日志确认；17×23=391、五言诗正确；
GSM8K 5-shot 200 题 96.5%（组合基线 96.0%）；8 并发×256tok 吞吐 324.6 tok/s
（2026-10-30 组合基线 247.2，不同 GPU 对/日期，方向性 +31%）；单流混合均值
20.3 tok/s、长输出 29.1（基线口径敏感，无回归信号）。

**附带修复（afef8ad）**：组合起服首跑在 prefill graph capture（num_tokens=2048）
崩溃，定位为 a4213fe 四崩修复#1（cu_seqlens_q 无条件走 varlen 入口）在 #20 重写
时被改为 layout 门控、#15 默认翻转后官方布局掉回 paged 入口丟 cu_seqlens_q 的
回归。已恢复无条件 varlen 路由（双布局 page_size 自适应）并补 layout-off 回归
测试（容器 3/3）。其余三项四崩修复核查：#2 已演进为 lightop 直连、#3 仍在、
#4 钩子已整体移除。

遗留（Phase 3 B5 域）：w8a8 apply 族 / compressed apply_weights、process_weights、
eplb、unquantized forward_cuda 仍各有 2 个 AROUND 同目标双挂（24 行告警），按
量化域裁决时同 create_moe_runner 模式收敛。

## 2026-11-07（二）遗留双注册收敛：6 组 AROUND 同目标双挂归一

Phase 1 遗留的组合栈 "Multiple AROUND hooks" 告警（6 目标 ×4 进程 = 24 行）全部收敛，模式与 create_moe_runner 相同（互斥臂→单点分发；非互斥臂→显式链式组合保持原栈序）：

- **w8a8 apply 族**（W8A8Int8/W8A8FP8MoEMethod.apply）：`w8a8_runner_dispatch` 增 apply 分发（`_uses_lightop_runner` 判 lightop 臂，否则 aiter 臂）；两臂模块去注册。
- **compressed apply_weights**（CompressedTensorsW8A8Fp8MoE.apply_weights）：同上分发（lightop `around_compressed_apply` vs aiter `around_apply_compressed_fp8`）。
- **compressed process_weights_after_loading**：非互斥——deepep 臂（legacy channel-fp8 deepgemm 权重准备）+ lightop 臂（channelwise lightop 重打包）都可能在 original 后生效；dispatcher 显式链 `lightop(deepep(original))` 保持原注册序（lightop 后注册=外层）。注册从 `w8a8_lightop` 与 `layers/moe/deepep_fp8` 移除。
- **UnquantizedFusedMoEMethod.forward_cuda**：新 `unquantized_moe_dispatch`（marlin `_enabled()` 优先，否则 aiter 臂）；`unquantized_aiter`/`unquantized_marlin` 去 forward_cuda 注册（后者保留 process_weights 单注册）。
- **handle_eplb_and_dispatch**：新 `eplb/hook_dispatch` 显式链 `placement(integration(original))` 保持原栈序（两臂是不同的独立校验：load_aware 前置条件 + locality_fair 静态派发约束）；integration/placement 去 target 注册，插件入口接线。

**已知未动（PD 配置才激活，本锚点不可验证，盲改违组合验证纪律）**：disagg 族 3 目标静态双注册（`CommonKVManager.get_mha_kv_ptrs_with_pp`、`PrefillBootstrapQueue.__init__`、`setup_state_kv_args`）与 `ServerArgs.add_cli_args` 双 AFTER——待 PD/EPLB 组合窗口按同模式收敛。`add_cli_args` 为 AFTER×2（注册表不告警），语义为两段 CLI 参数追加，合并前需验证 argparse 去重行为。

测试：`test_w8a8_runner_dispatch` 扩至 9 项（apply 分发、compressed apply 分发、compressed process 链序 lightop→deepep→original、臂零共享注册）；新增 `test_unquantized_moe_dispatch` 3 项、`test_eplb_hook_dispatch` 2 项（链序 placement→integration→original + 单注册钉死）。容器 e2e 验证见后补记录。

## 2026-11-07（三）双注册收敛 e2e 验收 + 134 tok/s 异常排查定案

**告警归零**：6d38aae 起服（GPU0,1 tp2，同 v5 MTP 配方）"Multiple AROUND hooks" **0 行**（原 24 行）；17×23=391、五言诗正确；GSM8K 5-shot 200 题 **96.5%**（与 afef8ad 一致）。

**134 tok/s 异常排查（已定案为服务器进程态，非代码非硬件）**：双注册收敛 build 首轮 8 并发仅 134.7/134.2 tok/s（上午 afef8ad 同配置 324.6）。三方同时段对照：A=6d38aae@GPU0,1（重启后）**516.8**、B=afef8ad@GPU2,3 **510.6**、C=6d38aae@GPU4,5 **509.2**——同代码跨卡对、同卡对前后代码全部 ±1.5% 内一致。结论：134 是当日多基准并发混跑后留下的进程态劣化（重启即愈），**代码无回归**；上午 324.6 亦偏低（JIT 缓存冷的首个 boot），第二轮 warm cache 后全栈 ~510-517。教训入库：**同服务器跨窗口 perf 数字不可直比，A/B 必须同窗口双服并跑**（本轮 B/C 对照法）。

**验收汇总（6d38aae）**：起服全图捕获 ✓；AROUND 双挂告警 24→0 ✓；正确性冒烟 ✓；GSM8K 96.5% ✓；8 并发×256tok 516.8 tok/s（同窗口 afef8ad 基线 510.6，+1.2% 噪声内）✓。

## 2026-11-05 DeepSeek MoE 捕获期双流在 HCU 上关闭

完成模块/行为：`hcu_sglang/models/deepseek/deepseek_v2/methods/moe.py` 的 `forward` 在进入 `forward_normal_dual_stream` 的条件中增加 `not C._is_hcu`。HCU 上 CUDA Graph 捕获期不再把 shared experts 放到 `alt_stream`，改走单流 `forward_normal`；非 HCU 平台、eager 模式、`alt_stream is None` 以及已有的 breakable graph 分支行为不变。attention 与 indexer 的 `alt_stream` 不变。

问题来源：官方 sglang 上 adapter 的 `moe_dualstream_gate` 只作用于上游 `DeepseekV2MoE.forward` 这个函数对象（早期为文本补丁，现为运行时换绑，判据 `torch.version.hip`）。plugin 的 REPLACE 钩子替换了整个 `forward`，adapter 的门就不生效了。DTK 上捕获的跨流 MoE 在 replay 时结果错误：eager 正确，decode graph 首 token 正确，之后全是乱码。本修复让 plugin 自带同一个平台门，不再依赖 adapter 对上游函数的适配。

实际验证（HY4 W4A8，单机 TP8，官方 sglang v0.5.22 baa9bef + adapter-das 1a5ff5c + 本仓 2edc8bc，`--disable-prefill-cuda-graph`）：
- 修复前 decode graph 下 chat 探针输出乱码。
- 修复后 chat 探针 3/3 命中，gsm8k 前 10 题 10/10。
- 对照实验：只关 MoE `alt_stream`、保留 attention/indexer 的 `alt_stream`，结果同样正确，说明问题只在 MoE 跨流。

未验证模式：
- prefill graph：开启后输出仍然错误，plugin 关闭时也错，与本修复无关，单独追查。
- PD、DP attention、DeepEP、其他 DeepSeek 系模型、多机。
- 修复前后的 decode 性能 A/B。

已知限制与下一步：
- 本门在 HCU 上放弃了捕获期 shared experts 与 routed experts 的流间重叠。DTK 修复跨流 graph replay 后可以去掉。
- 本次组合验证还依赖：
  - adapter-das 同批 PR（platform_seam、sparse FlashMLA 分块、paged MQA 调度表用 lightop 规划器、HYV4 CT scale 命名、weak_ref torch 路由）；
  - 尚未入库的 `jit_overlay` 问题：与官方一致的影子文件会遮蔽 adapter 的 JIT 补丁，单独处理。

## 2026-11-10 官方 SGLang 0.5.22 上的 GLM-5.3 W4A8 PP + LayerSplit

官方 `release/v0.5.22` 的 DSA 参数检查会拒绝 PP>1 与 LayerSplit 的组合。
plugin 在官方 `handle_model_specific_adjustments` Hook 上仅替换 HCU 的
这一处分支，继续检查 DSA prefill、CP、Mooncake 和 HiCache L3 的约束。
此路径依赖 adapter 的 `rewrite_resolution_function` 来保留已生效的
元数据适配。

P/D 接线补齐 prefill 的跨 PP stage 请求准入和传输完成状态同步、外部 KV
状态初始化，以及 Fake KV warmup 的可选参数兼容。首层 KV 预取按当前 PP
stage 与 forward batch 初始化。INT8 index-K 只为活动索引层分配持久缓存，
并将反量化 workspace 和页 claim 计入容量。历史 KV 读取、CT 权重布局、
DeepEP 事件所属 dispatcher 和 GLM CT EAGLE 采样沿用已核对的 HCU 路径。

节点组合使用官方 `baa9befa`，P=PP2/TP8/CP8/EP8，D=TP16/DP16/EP16，
保留 W4A8、FP8 KV 与 EAGLE。GPQA Diamond 为 181/198，HMMT Nov25 为
25/30，HumanEval pass@1 为 162/164。分数对应合并功能修复及整理变更的
节点快照。当前 PR 在最新主线上的验证范围是本地检查与接口对照，未在新
主线组合重跑 GPU 全量 profile 或精度评测。

## 2026-11-10 HY4 compressed-tensors W4A8 接入 slimquant_w4a8_marlin

HY4 preview 的 compressed-tensors checkpoint 是混合格式：routed experts 为
pack-quantized INT4 channel 权重，indexer `wk`/`wq_b`、MLA、shared experts 与
首层 dense MLP 为 int-quantized INT8 channel 权重，激活均为 INT8 per-token
dynamic。原 `slimquant_w4a8_marlin` 的 CT 分支要求所有 group 都是 INT4，
遇到 INT8 group 返回 None，官方 `_verify_quantization` 报 quantization
method 不匹配；`slimquant_marlin` 只认顶层 `int-quantized`，对该 checkpoint
静默退回 compressed-tensors，MoE 仍走 W4A16 Triton。

改动：
- `SlimQuantW4A8Int8MarlinConfig._supports_compressed_config`：每个 group 为
  4/8 bit channel 对称静态权重 + INT8 per-token dynamic 激活，4 bit 为
  pack-quantized、8 bit 为 int-quantized（group 级 `format` 优先于顶层），
  至少一个 INT4 group。全 INT4 checkpoint 仍满足，行为不变。
- `from_config`：HYV4 架构的 CT checkpoint，或任意含 INT8 group 的 CT
  checkpoint，设 `checkpoint_format=hy4_compressed_tensors_w4a8`，
  `linear_targets` 取所有 INT8 group 的 targets；含 INT8 group 但非 HYV4 时
  明确报错。
- `get_quant_method`：`linear_targets` 非空时，命中的 Linear 走 W8A8 int8，
  其余走 BF16，不再把 INT4 以外的 Linear 一律当 W8A8。
- `normalize_checkpoint_weights`：CT 以 `q + 8` 存储且偶数 K 在低 nibble、
  scale 为真实 per-channel scale；lightop/legacy W4A8 约定为有符号补码、
  偶数 K 在高 nibble、scale/16。转换在 TP/EP 切分之后执行，每个 rank 只转
  自己的分片。Marlin、Triton、Aiter 三个 MoE 方法的
  `process_weights_after_loading` 开头调用。
- registry 的 Lazy 包装：官方 `QUANTIZATION_METHODS` 登记的是
  `LazySlimQuantW4A8Int8MarlinConfig`，CT 且用户显式选
  `slimquant_w4a8_marlin` 时转调真实 Config 的 override，否则上面的校验不会
  被调用。
- `hcu_sglang/models/hunyuan_v4.py`：`HYV4ForCausalLM.__init__` 的 AROUND
  在 `slimquant_w4a8_marlin` 但 checkpoint_format 不符时报错；`load_weights`
  的 AROUND 丢弃 `*.weight_shape`，`*.weight_packed` 去掉 `_packed` 后缀并按
  INT8 view，非 `.mlp.experts.` 的 INT32 2D 张量报错。

验证（S1，gfx938，官方 `baa9befa` + adapter + 本分支，HY4 preview
`Hy4-preview-Channel-INT4-w4a8`，TP8，decode graph 开、prefill graph 关）：
- 8 个 rank 均 `selected_moe_method=SlimQuantW4A8Int8MarlinMoEMethod
  resolved_backend=lightop`，加载 51.46 GB/卡。
- chat 3/3；gsm8k 0-shot 200 题 193/200（W4A16 Triton 同配置 194/200），
  用时 598s（W4A16 为 3364s）。
- prefill profile（8k chunk）MoE GEMM 由约 12.3s/步降到约 0.47s/步
  （同时合入 adapter meta_size 修复后的数字）。

已知限制：
- HY4 config 的 `swiglu_limit=10.0` 在 lightop
  `fused_experts_impl_w4a8_marlin` 中没有对应参数，marlin 路径不做 clamp；
  gsm8k 未见掉点，长输出与其他评测未验。
- 未覆盖 MTP（`hunyuan_v4_nextn`）、DeepEP/EP、prefill graph、PD。

## 2026-11-11 DeepEP clean_buffer 显式传递 top-k

本次基线为 `9543829be6177e4261a2a3902fb6762e5602b881`，起始工作区干净。
`hcu_sglang/layers/moe/deepep_current.py` 的 `DeepEPBuffer.clean_buffer`
向 `clean_low_latency_buffer` 增加 `num_topk=state.num_topk`。
按仓库约定补充本记录并重建离线导览。

在用户指定的 `yangyn_sglang_plugin_1022` 容器（Python 3.10.12，已安装
SGLang `0.5.18+dtk2604.torch2110.2610211227.g414c32`）运行：

- `python3 ci/l0_gate.py`：通过，检查 697 个 Python 文件、8 个 Skill。
- `python3 -m unittest discover -s tests -v`：运行 218 项，报告 11 处失败、
  13 项跳过。失败包括 CLI 的 `field_names` 导入、quant_ops 来源哈希、
  DSA Indexer AST 哈希及 slimquant_w4a8_marlin AST 哈希不一致；本次未修改这些模块或合同。
- `PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 tests/runtime/deepep_current.py -v`：4/4 通过。
- 同样设置 PYTHONPATH 运行 `tests/runtime/deepep_graph_cpu.py -v`：
  `setUpClass` 因已安装 SGLang 缺少 `arg_utils.is_record` 报 ImportError，0 项执行。
- 提取本次实际 `clean_buffer` 方法进行调用检查：低延迟分支准确传递
  `(128, 7168, 256, num_topk=8)`，非低延迟分支不调用清理方法，通过。

日志位于 ignored 目录 `artifacts/clean-buffer-num-topk-20261111/`。
以上是 CPU/接口检查，未验证 GPU 通信内核、真实模型生成或多卡 Graph。
远端托管 Quality Gate 未触发，本次执行本地 CI 配方。
## HY3 Channel FP8 与 NEXTN 适配（SGLang 0.5.22）

### 改动范围

- HY3 主模型和 NEXTN 使用方法 Hook，内部组件仍实例化官方类；调整 FFN 通信边界，并为 NEXTN 使用独立的 `layer_stack`。模型 Hook 目标仅包含 HY3 两个类，保留 HY4 等其他模型的注册条目。
- Channel FP8 的旧版 grouped GEMM 路径显式向 adapter 请求 `quant_group_size=0`；两处 masked DeepGEMM 调用前将 scale 转为连续布局。公共 FP8 路径不限定模型名称，不能据 HY3 通过推断其他模型通过。
- 融合 RMSNorm/RoPE 后恢复 Q 的二维形状，补齐两处 JIT 宏兼容；保留 main 已有的 HIP RoPE 实现。
- EXTEND 输出按本轮实际输入长度 `_original_num_tokens` 裁剪，避免把已缓存前缀计入输出，修复缓存命中后的 NEXTN 负尺寸问题。

### 验证结果

- 2026-11-11 组合验证：官方 SGLang `baa9befa`、plugin `3eb48a8`、adapter `9e616cc`；DTK 26.10、Python 3.12、Torch 2.11。HY3 Channel FP8 在 TP8 / attention DP8、DeepEP、FA3、E4M3 KV、NEXTN 2/1/3 下完成启动，基础问答 4/4 正确且正常结束。静态显存比例为 0.70，prefill Graph 档位为 64/128/256/512/1024，decode 最大 batch 为 16。
- 固定 GSM8K 5-shot 200 题：数值答案正确 197/200，正确且正常结束 196/200，1 题达到 2048-token 上限，0 请求错误。与旧 plugin 分支 `e05a1cd` 的相同样本相比，200/200 完整文本、数值答案和结束原因一致；对照对象不是 sglang-das 0.5.18。
- NEXTN 缓存回归共 128 次请求完整返回，未复现负尺寸 `-64` 或服务退出。合成 token 输入的冷、热缓存完整输出仍有差异，原因未定位；这不等于模型准确率下降，也不能记作数值一致性通过。
- HY3 方法 Hook 7 项、实际模块缓存裁剪 3 项、DeepEP 分发 5 项通过。DeepEP FP8 6/7 通过，其中 GPU scatter/gather 最大绝对误差为 0；剩余为 main 已存在的 `_fp8_max` 全局符号检查失败。
- CPU 全量 unittest：候选分支 217 项、原 main `9543829` 210 项；同条件均为 82 failures、7 errors、17 skipped，失败和错误节点集合无新增。已有失败涉及历史来源快照、缺少外部输入与运行时导入；全量测试未通过。L0、导览快照和 `git diff --check` 通过。

### 依赖与限制

Channel FP8 依赖配套 adapter 的 DeepEP 量化粒度透传。上述环境还使用部署目录中的 DeepGEMM v4 兼容扩展，该扩展不在两仓提交内；AITER custom all-reduce 初始化存在回退警告，通信后端未验收。公共 ForwardBatch、FP8 路径改动尚无 HY4 回归证据。

早期 0.9 显存配方曾 OOM，8 并发曾出现 NEXTN padding 失败，未按这两个原始配方复验。环境变量治理、与 sglang-das 0.5.18 的同条件性能和精度对照仍未完成。上述模型结果不能代替这些验收项。
## 2026-11-12 PR #41 的评审修复与变基后验收边界

本轮将 PR 变基到 plugin main `b7d4356`，保留该基线的 DeepEP clean-buffer top-k 与 nightly wheel 变更。`architecture.html` 的冲突通过 `tools/build_guide.py` 从合并后的源文档重新生成。四个 MQA 运行时修改文件与旧 PR head `d557ddd` 逐字节一致；本轮没有改变 LightOp API、分组门控、默认配置、索引顺序或量化语义。

按 reviewer 点名的范围增加 `tests/test_mqa_request_split.py`，直接加载零依赖规划器，不导入框架或设备库。九项 CPU 测试通过，覆盖空输入和全零 Q、单请求、部分分组优于两种极端划分、阈值相等时保留更多分组、完整切片回退、零 Q 请求跨 K 边界与两端裁剪、一字节预算的逐行分块、零预算和非法长度/阈值。它们验证规划与坐标，不代表 native kernel、Graph 或整模型通过。

提交前按 CI 的 Ruff 0.12.1 E9/F63/F7/F82（ignore-noqa）规则检查全部 PR Python 文件，并执行语法、L0、导览重建与 `--check`、`git diff --check`。GitHub CI 独立记录结果；没有在本机运行完整测试套件。

变基后完整组合的真机重验明确为 **pending**，由 [#47](https://github.com/HYGON-AI/sglang-plugin-das/issues/47) 记录两 PR 的提交、实际安装库与输入绑定，以及全部实际工作 rank 的调用链、小样本 logits/消费者对照和 P0/P1 延迟接受条件。历史 `glm5.3-mqa-validation-20261111.json` 保留原有来源和 frozen #27/#62 组合边界，未改为当前 head 已通过；历史 HMMT/HumanEval 差异仍未归因。

本轮浏览器截图、交互与窄屏复核未执行：浏览器工具的 URL 安全策略拒绝 `file://` 协议。已执行导览重建与快照检查，未将结构检查记作截图验收。
## 2026-11-12 PD prefill inflight 队列 rank 一致性

**问题来源**：输入为 `prefill-consensus-fix.patch`（来源目录 `zhongxing/zombie-process`），对方基线为 `sglang/srt/disaggregation/prefill.py`。该路径的上下文只存在于本仓扩展后的 `hcu_sglang/disagg/prefill.py`（`external_kv_cleanup_done`、`_all_reduce_polls`、`FAKE_BOOTSTRAP_HOST` 均为本仓自有符号），因此按路径映射落到本仓，不进入 sglang-das 的 `disaggregation/prefill.py`。

**改动范围**：仅 `hcu_sglang/disagg/prefill.py`，共 5 处。

- `process_batch_result_disagg_prefill` 的 external KV 清理分支：队列入队不再依赖 rank 本地的 `cleanup_done`，改为一律入队。
- `process_batch_result_disagg_prefill` 的 abort 分支：`bootstrap_host == FAKE_BOOTSTRAP_HOST` 时先调用 `req.disagg_kv_sender.abort()`，让假 sender 从 `WaitingForInput` 进入 Failed，避免 abort 落在 forward 期间时永久滞留。
- `process_disagg_prefill_inflight_queue` 的 poll 循环：`external_kv_cleanup_done and external_kv_response_sent` 的请求直接判 done，不再走 sender poll。
- `process_disagg_prefill_inflight_queue` 的循环之后：done/undone 划分改为 TP、CP 两组各一次 `_all_reduce_polls` MIN 汇总；按 RID 累计超过 60 秒未清理的请求每分钟告警一次；`rank_consensus_checker.assert_same` 记录 done/kept 数量。
- `handle_bootstrap_failure`：入队同样不再依赖 `cleanup_done`，去掉已被消费的返回值。

同时把 `handle_bootstrap_failure` 中不再使用的 `cleanup_done` 局部变量删除，避免未使用变量告警。

**实际验证（本机，Windows / Python 3.14，无 torch、无 sglang、无 DCU）**：

- `python -m py_compile hcu_sglang/disagg/prefill.py` 通过。
- 新增 `tests/test_pd_prefill_inflight_consensus.py`，9 项 AST 结构契约全部通过：导入来源、TP/CP 两次汇总、汇总位置在 poll 循环之后、done 集合由汇总结果重建、已回包请求提前出队、两处入队不依赖 `cleanup_done`、假 sender abort 在入队之前、告警限频。
- 全量 `python -m unittest discover -s tests`：本次改动前后同为 80 failures / 31 errors / 16 skipped（改动前 202 项，改动后 211 项）。属既存失败，与本次改动无关，未逐项归因。
- `python ci/l0_gate.py` 失败于 `plans/hcu-runtime-target-closure-20261105.csv` 快照过期；stash 本次改动后复现相同结果，确认为既存问题，未重建快照。

**未验证项**：

- 未在真实 PD 分离拓扑上运行。多 rank 下队列长度是否已先一致（all_reduce 要求各 rank 长度相同，否则会 hang 或报错而非修复）未经运行时确认。
- 未复现原始症状（`POST /flush_cache` 返回 400、`#inflight-req` 只增不减）。
- 未跑 `test_zombie_inflight.py` 压测流程。
- 未验证 `rank_consensus_checker` 在未开启 `SGLANG_ENABLE_RANK_CONSENSUS_CHECKER` 时的行为（按实现为 no-op）。
- `_inflight_held_since` 采用首次访问惰性初始化，未在 `init_disaggregation` 中预置；与 `no-getattr-defensive` 风格建议不一致，保留原 patch 写法。

**与健康探针改动的关系**：同目录 `修改方案.md` 的改动 A/B/C 处理 `health_generate` 超时不下发 abort、以及 scheduler 忙时丢弃探针 abort 的问题，本次 patch 不包含；本次只处理 inflight 队列的 rank 一致性与假 sender 回收。两者互补，不能互相替代。

## 2026-11-13 Qwen3.8 GDN HIP kernel-only verify

状态：本地实现，L0 passed；GPU 端到端 A/B 待跑。

- 范围：新增 `hcu_sglang/layers/attention/linear/gdn_hip_bplus.py`（手写 HIP 递推核的 ctypes 封装，含 `dt_bias`/`A_log` bf16->fp32 修复），并在 `gdn_source.forward_extend` 的 target-verify 回退分支接入 kernel-only 路径（`SGLANG_GDN_HIP_KERNEL=1`）。
- 行为：verify 仍写全量 per-draft 快照，行距/槽号与上游 `kernel_dispatcher.target_verify` 一致；不启用 B+ 写省略，也不依赖 0.5.18 的 state handoff（0.5.22 无该机制）。
- 文件：`gdn_source.py`（+322，新增 `forward_extend` 副本与分支）、`gdn_backend.py`（REPLACE 钩子转调 `gdn_source.forward_extend`）、`gdn_hip_bplus.py`（新）、`layers/attention/linear/lib/`（HIP 源 + `build_gdn.sh`）。
- 开关：`SGLANG_GDN_HIP_KERNEL=1` + `GDN_HIP_SO`（或 `lib/gdn_bw_ship.so`）。
- 已执行：`python3 ci/l0_gate.py` passed；`py_compile` passed；`lib/build_gdn.sh` 在 gfx938 上构建成功。
- 未验证：真实服务 target-verify 的数值/吞吐 A/B；CUDA graph 捕获；多卡/PP；B+ 提交链（另开 PR）。

## 2026-11-13 clean 0.5.22 性能栈（hbird / fp8 KV / 草稿头 fp8）

状态：本地实现；c40/2k2k 实测；L0 passed。

- hbird bf16 GEMM：plugin `register()` 显式调用 `hbird_sglang.install()`
  （adapter 的 sitecustomize 先 import 了 unquant，hbird_site 的 _Finder 不再触发）；
  `ModelRunner.load_model` 后 `prepare_model`；`LogitsProcessor._compute_lm_head`
  的 hbird 快路径。实测 packed=256 linear、freed 45.31 GiB、skipped 48。
- fp8 KV：`prepare_paged_mha_query` AROUND 钩子补 k/v descale；varlen fp8
  prefix decode 补 q_descale（1.0）。修复 `For fp8 prefix decode, q/k/v descale
  tensors must be provided`。
- MTP 草稿头 fp8：`models/qwen3_5_mtp_fp8.py`，只替换 draft 模型的 `lm_head`
  （target 的 bf16 head 不动），e4m3 + hbird fp8 整词表一次 launch。
- 1-D mrope：clean 0.5.22 源应用上游 `fix/mrope-1d-fast-path`（逐位相同）。
- c40 / 2k in / 2k out / TEMP=0：677.90 → 844.84 tok/s 聚合
  （16.95 → 21.12 tok/s/req），Mean TPOT 52.80 → 40.98 ms，80/80 成功。
  归因：hbird lm_head +5.0%、fp8 KV +2.0%、hbird linear +16.4%。
- 未验证：GDN B+ 提交链、gemma/qk/rope/zba 融合核、prefill graph
  （breakable prefill 在 HCU 报 CUDA error 35，暂关）、T=0.2 全量
  HumanEval（adapter 的 flashinfer 树采样 JIT 缺口）。

## 2026-11-14 c40 吞吐推进：scatter 调度开销、commit 去同步、handoff 实验

状态：本地实现；c40/2k2k 实测；handoff 默认关（负收益）。

- 上游（sglang 源，patch 见 clean-stack/patches/upstream/）：
  - `mamba_state_scatter_triton.py`：`fused_mamba_state_scatter_with_mask` 的
    BLOCK_SIZE 1024→16384（elem≥65536）。c40 trace 里该核一步 48 ms（1.47M CTA
    的调度开销，6 GiB 拷贝只有 ~250 GB/s）；改后 9.2 ms/次。逐位对拍
    （elem=40000/786432 两条路径 `torch.equal`）。
  - 同文件：conv window scatter 连续化（CONTIG/ELEM + BLOCK 16384 + meta key），
    548.5→169.7 µs/拍，逐位。
- 插件：
  - `gdn_bplus.stash` 记录 host int T，compensate 不再每步 `tdev.item()` 同步
    （+0.7%：1137.36→1145.03）。
  - `compensate_from_stash` 缓存每层 `layer_cache`/`flat`（持平，无回归）。
  - **GDN state handoff（实验，默认关）**：`SGLANG_GDN_HIP_HANDOFF=1` +
    `SGLANG_GDN_HIP_HANDOFF_SKIP_TRACK=1`（仅 `--disable-radix-cache`）。
    verify/compensate 直写 snapshot 行（`disable_state_update` 语义），
    commit 只更新 int32 `prev_step`，conv 窗口照常 scatter；prefill 末尾把
    temporal seed 到 `intermediate_ssm[slot,0]`。
  - 验证：单请求 T=0 与基线逐字符相同；trace 里 ssm scatter 完全消失
    （GPU 内核 9831→9087 ms/10 s）；但同卡 c40 A/B 1080.97 vs 1137.36
    （**净亏 5%**）——commit 的 48 层 Python 循环/launch 串行化把 GPU 从满
    变成等，scatter 的 GPU 时间原本盖住了这段 CPU 时间。默认关，不进产线。
  - CPU trace（12.2 s，148 万事件）：Python self-time 前几名全是等待/锁/
    graph replay，插件 Python 工作 <0.2%，差距不在插件 Python 开销。
- 端到端：clean 基线 677.90 → **1145.03 tok/s 聚合**（28.63 tok/s/req），
  Mean TPOT 52.80 → **29.92 ms**，TTFT 7881.6 ms，80/80 成功；产线 0.5.18
  同口径 1256.78（91.1%）。
- 未验证：handoff 在 radix 开启/PP>1/抢占场景；multi-layer compensate
  （48 launch→1）尚未实现；T=0.2 全量 HumanEval。
