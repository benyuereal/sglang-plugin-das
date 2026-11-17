# 配置管理：路径、开关与运行时状态

> **本文用途**：区分工具参数、SGLang/adapter/plugin 配置的所有权，说明组合运行时的覆盖与记录规则。当前 `hcu_sglang/config.py` 与 `hcu_sglang/environ.py` 已含插件配置和注册逻辑；所有权及状态规范继续适用于当前实现。

## 1. 当前工具的路径配置

| 名称 | 含义 | 要求 |
|---|---|---|
| `REPO` | 当前方法库 | 在仓库根目录用 `pwd -P` 获取；执行 L0、工具测试和脚本的位置 |
| `OFFICIAL` | 指定 SGLang 官方工作树 | 固定 SHA；与清单工具的 official-ref 匹配 |
| `FORK` | DAS 来源树 | 审计保留 dirty；不能作为正式运行时 import 来源 |
| `PLUGIN` | 含业务实现的 plugin 树；当前实现可与 `REPO` 相同 | 当前包为 `hcu_sglang`；支持旧布局的工具按需显式选择 `--plugin-package` |
| `ARTIFACTS` | 本次审计/日志输出目录 | 建议位于 `REPO/artifacts/<task>`，不要落到来源树内 |
| `OUTPUT` | 源码回溯新输出树 | 不与任一输入相同、互为父目录或子目录 |
| `WHEEL` | 本次安装的 wheel 文件 | 安装前核对交付方 SHA256；不从旧任务目录自动选择 |
| `MODEL` / `SERVED_MODEL_NAME` | 模型目录 / 实际服务模型标识 | 按配方填写；响应校验使用服务返回的模型标识 |
| `CONTAINER` / `CONTAINER_USER` | 执行容器 / 合法的 UID:GID | 仅容器方式需要；镜像挂载内路径与宿主路径分别确认 |
| `SOURCE_ROOT` / `VLLM_SKILL_SOURCE` | 历史抽取来源根 / 原 vLLM Skill 根 | 仅来源溯源需要；JSON 中以变量占位符记录，不是工具默认路径 |

在仓库根目录准备本地配置：新建 `.env`，按上表填写本次任务需要的变量（如 `OFFICIAL`、`PLUGIN`、`ARTIFACTS`）及路径；已有 `.env` 时保留并编辑它。来源、产物和模型路径可位于任意目录，不要求互为同级。外部路径填写本机实际绝对路径，含空格的值使用引号。进入容器后按容器内可见位置配置，不能直接沿用宿主路径。

```bash
# 从当前方法库根目录执行；.env 只保存本机配置，已被 Git 忽略。
export REPO="$(pwd -P)"
test -e .env || touch .env
# 编辑 .env 后再加载；仅 source 自己维护、可信的配置文件。
set -a
. ./.env
set +a
# 按任务检查必需项，例如插件能力开发：
: "${OFFICIAL:?请在 .env 中设置官方源码路径}"
: "${PLUGIN:?请在 .env 中设置外部插件源码路径}"
: "${ARTIFACTS:?请设置本次证据目录}"
```

这些是手册中的 shell 变量，不是本库自动读取的环境变量接口。加载 `.env` 是显式操作；实际 CLI 参数以工具 `--help` 为准，设置 `PLUGIN` 不会自动启用运行时插件。只阅读文档或运行离线工具测试时无需配置外部路径。首次初始化插件只需现有官方源码与外部新工程目录；按对应指南检查所需输入。

临时安装或示例工程使用 `mktemp -d`，由系统及 `TMPDIR` 决定位置，不固定临时文件名。需要持久部署时显式指定可写安装目录。历史记录中的 `${REPO}`、`${SOURCE_ROOT}` 等是脱敏位置标识；来源清单仍保留原始版本及文件哈希，不自动解析占位符，也不把当前路径冒充历史路径。原始 ignored 运行日志保持原样，按任务单独保管。

共享文档、Skill、模板与代码不得包含个人目录、私人节点名称或固定任务临时目录。目录结构示意使用相对路径；外部输入用显式变量或 CLI 参数。操作系统接口（如 `/proc`）和标准 shebang 不属于用户目录配置。

## 2. 运行时配置所有权

| 配置来源 | 所有者 | plugin 接入原则 |
|---|---|---|
| `SGLANG_PLUGINS` 与框架参数 | pinned SGLang | 按该版本 loader/CLI 语义使用 |
| SGLang 自有 `Envs` / typed descriptor | pinned SGLang | 有对应 API 时使用 `.get()` / `.override()`，不在 Hook 中散落重复解析 |
| adapter 的配置/快关 | adapter | 引用其固定版本手册；不复制同名注册表或改写它的默认值 |
| plugin 自有能力开关 | plugin 实现 | 集中定义、记录类型/默认/支持条件，独立命名并可诊断 |
| NCCL / driver / launcher / OS 变量 | 外部依赖或部署 | 按所有者的接口消费，不擅自改变语义 |

本仓 `hcu_sglang` 已包含来源插件的配置开关和 `hcu` 入口；具体以当前代码为准，不假定存在统一的 `HCU_DISABLE_OPS` 或额外配置服务。

### 当前 Qwen4 运行时配置

以下字段由 `hcu_sglang/environ.py` 注册到官方 typed `Envs`，在服务启动前设置并继承给 worker；更改后重启服务。默认关闭的性能路径需按支持条件显式启用，checkpoint/INT8 PLE 支持不依赖这些性能开关。

| 名称 | 类型 / 默认 | 行为与支持条件 |
|---|---|---|
| `SGLANG_QWEN4_COMPACT_PREFILL` | bool / false | 纯 EXTEND/IDLE 使用 SUM_LEN，idle rank 保留一行；TP=DP=EP、attention TP/CP=1、Gatherv、无 A2A/TBO、disabled prefill Graph |
| `SGLANG_QWEN4_LOCAL_ROUTING` | bool / false | compact prefill 在量化前的本 rank BF16 输入上选专家；要求非融合 shared expert |
| `SGLANG_QWEN4_ROUTED_INT8` | bool / false | 局部 routing 后传输 INT8 codes/scales，直接交给支持预量化的 AITER moe_c 一次 |
| `SGLANG_ENABLE_DP_GATHER_INT8` | bool / false | Qwen4 FFN 的独立 INT8 transport 入口；SUM_LEN/Gatherv、支持预量化的 quant method |
| `SGLANG_MOE_C_NATIVE_SORT` | bool / false | E129/top-k10、HIP、支持的 block size 使用保留 drop-route 的 native sort；依赖 AITER sorting/conversion API |
| `SGLANG_QWEN4_SPLIT_HC_MIX` | bool / false | gfx936/BF16、1–16 行、HC4/hidden2560/lowrank320 的分阶段 HC mix；保留确定性模式守卫 |
| `SGLANG_QWEN4_FLAT_STATE_SCATTER` | bool / false | gfx936、36 层 FP32 48×128×128、1–16 条 metadata 的连续状态回写；保留 scratch 源行映射 |
| `SGLANG_QWEN4_PREFILL_DIAGNOSTICS` | bool / false | 首次 compact 布局输出 rank/counts，和 routing 日志一起诊断实际路径 |
| `SGLANG_PLE_ETP_REDUCE_SCATTER` | bool / true | 支持的 PLE 分片布局采用 reduce-scatter；其他布局保留 all-reduce + scatter |
| `SGLANG_PLE_ETP_REDUCE_SCATTER_MIN_TOKENS` | int / 2048 | 小于阈值时使用 all-reduce + scatter |

关闭相应性能开关或超出窄形状门控时调用当前有效基线，不绕过 adapter。显式启用 INT8 transport 而缺少必要量化/通信合同，或启用 native sort 却缺少 vendor API 时明确报错。`SGLANG_DP_USE_GATHERV`、Graph、并行与 MoE 后端属于官方配置，按其接口设置；adapter 的 `qsa_mqa` 快关参见 `adapter-das` 仓 `docs/env-vars.md`，不由插件重复定义。完整支持范围与已测组合见 [开发记录](hcu-sglang-development.md#2026-11-04-qwen4-exp-w4a8-与-int8-ple-适配)。

## 3. 新增配置的设计检查

| 检查项 | 要给出的答案 |
|---|---|
| 名称与类型 | 谁拥有；布尔/枚举/数值/字符串，取值范围是什么 |
| 默认与覆盖 | 未设置、显式设置和不支持环境下分别采用什么；显式用户值优先，非法或不支持值给出可诊断结果 |
| 生效阶段 | import 前、进程初始化、请求级还是 forward 级；修改后是否需重启 |
| 依赖与组合 | 需要哪个 adapter/vendor/official 版本；另一个仓是否影响同一状态 |
| 观察与回退 | 日志如何显示最终值、来源和选中路径；停用后回到哪条有效基线 |
| 测试 | 默认、显式覆盖、不支持/缺依赖、进程传播与恢复是否覆盖 |

优先用一个含义明确的配置，不增加彼此冲突的多个布尔项。使用目标版本实际存在的 API；不能因为 DAS Skill 讲了 RuntimeContext 就假定旧 official 也具备同名接口。

## 4. 配置与状态分层

```mermaid
flowchart LR
    E[启动参数与环境] --> C[解析后的进程配置]
    C --> R[资源/stream/buffer 生命周期]
    R --> F[请求与 per-forward 状态]
```

配置、进程资源和每步状态分开。融合 KV 写入标记属于 batch/layer 的生命周期，不放模块全局跨请求复用；Graph 的动态输入不能只在 Python capture 时赋值。若 pinned 版本有 publish/override 机制，按其生命周期写入并记录来源。

平台判定也有所有权：adapter 可能向 SGLang 暴露 CUDA 语义，而真实 torch/编译器仍是 HIP。plugin 的硬件能力判定应使用足够直接且被验证的设备/依赖信息，不能只凭某个 SGLang `is_cuda()` 推断物理设备与 vendor 符号。

V3 的外部运行时输出实际引入 `HCU_PLUGIN_MINIMAX_OPT`，由 `--minimax-opt` 在构造配置前设置并继承给 worker；在 ServerArgs 上仅提供兼容读取，不写冻结记录。该接口最初属于外部 runtime 0.3.0，当前迁入实现见 `hcu_sglang/config.py`；历史行为与验收边界见 [V3 手册](v3-runtime-validation.md)。

官方配置记录可能是 msgspec Struct，不要仅靠 dataclasses.is_dataclass 判断；使用 pinned 版本的 `arg_utils.is_record`/record_fields。CLI 解析后还要 resolve_once 才算完成配置解析。

### 当前 custom all-reduce 参数

当前 HCU 插件提供 `--custom-all-reduce-backend {auto,native,aiter,off}`，默认 `auto`。官方0.5.22的原始 ServerArgs/comm记录没有该字段，插件在 CLI 构造原始记录之前消费参数，并用内部 `HCU_PLUGIN_CUSTOM_ALL_REDUCE_BACKEND` 继承到 worker；不向冻结输入或 `get_exec().comm` 补写属性。直接构造官方 `ServerArgs(custom_all_reduce_backend=...)` 不是已支持接口；程序化调用应在初始化 engine 前使用 `hcu_sglang.config.configure_custom_all_reduce_backend(value)`。

| 规则 | 实际行为 |
|---|---|
| 优先级 | 官方解析后的 `disable_custom_all_reduce=True` 强制 `off`；否则保持显式后端；HCU上 `auto` 且 legacy `SGLANG_USE_AITER_AR=1` 按DAS提升为 `aiter` |
| 生效阶段 | 每个 worker 的官方 distributed bootstrap 先设置原有 flags，再将插件解析结果交给组工厂；一个进程对应一个活动 engine 配置 |
| 组覆盖 | world组始终 `off`；model-parallel组默认取解析结果；显式组级参数优先，不修改其他组成员定义 |
| strict-Fabric | `aiter` + `AITER_AR_TRANSPORT=fabric` 时构造异常原样抛出，disabled communicator 明确拒绝；不静默回落成 RCCL |
| QuickAllReduce | 保留DAS的独立 `SGLANG_USE_QUICK_CUSTOM_ALLREDUCE` 门控；Fabric/auto transport 不并建 IPC QuickAllReduce |
| 原配方别名 | `--nsa-prefill-backend` / `--nsa-decode-backend` 使用原 deprecated alias action，写入当前 DSA 字段，未指定时不覆盖 canonical 默认值 |

后端初始化日志记录 requested/selected transport、disabled、ranks与world size。参数解析通过不代表 communicator 初始化或模型正确；IPC、Fabric、Graph及各组拓扑分别验收，见[开发记录](hcu-sglang-development.md)。

## 5. 变更与排障记录

每次配置改动同步默认值说明、支持范围、测试与启动日志示例。发生冲突时先列出“原始值 → 解析值 → 最终值 → 实际分支命中”，再修改实现。验证模板见 [validation-record](../templates/validation-record.md)，详细状态约束见开发 Skill 的 [runtime-and-env](../skills/hcu-plugin-dev/references/runtime-and-env.md)。

## 6. GLM CT W4A8 的原生 BF16 RoPE 选择

插件开关 `SGLANG_HCU_USE_LIGHTOP_ROPE` 为布尔型，默认 `False`；在 GLM-5.3 CT W4A8 配方中显式设置为 `1`。它在 HCU 的 BF16 `RotaryEmbedding` 构造结束后准备镜像内公共 LightOp `rotary_embedding_fuse`，由正式 `forward_cuda` AROUND Hook 消费；修改后需重新构造服务。默认关闭时保留既有分发，非 HCU、非 BF16 或融合 KV-store 调用保留当前有效原实现。

该路径使用已有 `HcuRotaryFallback` 的 typed cache memo，为 LightOp 提供与 query 同 dtype/device 的缓存视图；框架持有的 FP32 `cos_sin_cache` 及其他 indexer 消费者保持原所有权。LightOp 的缺失或执行错误直接暴露，不在数值失败后静默切换其他算法。此选择不改变 CT W4 权重、动态 INT8 激活或 GEMM 分发。

LightOp 自行选择 launch，不套用旧 sgl_kernel 的 256-thread head 分块；真实 64-head 输入完整调用与拆成 8-head 调用的数值并不相同。offsets 在调用前按原语义加到 positions，返回原地更新的 query/key。

适用证据和待验证范围见[开发记录](hcu-sglang-development.md#2026-11-05-glm-ct-w4a8-的-bf16-rope-选择)。两组八 rank 的真实 Q/K 及更换输入后的组件 Graph 已逐元素对齐原生，完整服务仍待复测。

## 7. GLM CT W4A8 的公共 W8A8 GEMM 分发

`SGLANG_HCU_W8A8_USE_LIGHTOP` 为插件布尔开关，默认 `False`。GLM CT W4A8 配方显式设置为 `1`，使既有 `triton_scaled_mm` 接口使用镜像内公共 LightOp 分发，跳过插件自行选择的 smooth ASM 快路径；公共接口内部仍可能使用 ASM。开关不改变输入 INT8 量化值、token scale、权重或输出 dtype；默认关闭及非 HCU 的行为保持原样。应在服务启动前设置，Graph 捕获后不要切换后端。

真实首层 Q-b 的两组八 rank 输入、权重和 scale 对照中，两套镜像的公共接口均逐元素重建原生输出；插件的 ASM 快路径有四个元素不同。该有限组件证据不证明所有 shape 或完整模型通过，后续分发和模型复测见[开发记录](hcu-sglang-development.md#2026-11-05-glm-ct-w4a8-的-w8a8-gemm-分发)。
