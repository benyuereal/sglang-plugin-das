# 测试契约：真实边界、反例与有效 oracle

> **用途**：按实际改动选择能发现错误的测试，避免接线或孤立算子全绿掩盖模型调用链失败。开发闭环见 [历史经验](../../../docs/development-methodology.md)，分层结论见 [验证手册](../../../docs/validation.md)。

## 1. 从改动选择回归入口

| 改动面 | 回归应保留的真实结构 | 关键断言 |
|---|---|---|
| API/Hook/seam | 顶层注册链、公共分发、实际 reader/readout、绑定 receiver | 预期非空 targets 全量 apply；新可选/keyword-only 参数与返回值完整传递 |
| KV pool/布局 | 构造器→descriptor→写/移/读→backend→native consumer | pointer、stride、byte span、不同 K/V head_dim、页边界与备用 buffer |
| residual/norm | 合法 initial/written/pending、entry wrapper、模型实际 NormReadout/NormQuantReadout | residual 更新一次、pending 被消费、共享 plan 不被修改、额外张量不丢失 |
| collective | 生产者局部贡献→真实 FFN exit/下一消费者 | TP1 和本次 TP>1、空/非空；sum 的 owner、group、次数及最终值 |
| cache/消费者 dtype | canonical buffer 与消费者 cast/视图分开 | dtype/device、源 buffer 替换、memo 复用/失效、其他消费者不受破坏 |
| Graph | 同一真实 kernel/状态链的 eager、capture、变输入 replay | 改输入、位置/slot、cache 后读取新数据；K 的每个 head 只处理一次 |
| 配置/能力选择 | 实际启停和所声明的 adapter/plugin 组合 | 分支命中、完整参数、回退保留当前有效基线 |
| 安装/交付 | 指定源码、隔离 wheel/镜像、embedded | 包数据与 ABI、真实模块路径、外部入口隔离、同一行为合同 |

并非每次改动都跑表中所有行。按影响面选测并记录理由；只对当前需求明确涉及的模型、部署组合和产物给结论。

## 2. Mock 的边界

mock 昂贵硬件或通信时，保留被验证的框架生命周期。只直接调用内部 Hook 会漏掉公共 reader 在 Hook 前抛错；`__new__` 手填 buffer 会漏掉真实构造中的 descriptor 错误。pending residual 用生产者的 `record` 创建，不把 written 状态与独立 hidden 硬凑在一起。

使用实际 readout。NormReadout 与 NormQuantReadout 即使 quant_format 为空也可能调用不同 norm 签名；第三个位置 None、显式/默认 bool 与 keyword-only 都需按调用点覆盖。静态 signature bind 只是候选检查；加 `**kwargs` 后丢弃数据不能算兼容。

归约测试不能停在 TP1 的 router/expert mock。当前社区边界若负责 sum，专家方法应返回局部贡献；用真实 ExitPolicy/FfnExit 验证完整结果和 collective 次数。V3 中预期 8、旧结果 32 的反例比“all_reduce 被调用过”更能验证合同。

## 3. Oracle 与修复前反例

先让同一回归在修复前失败，再保留最小变更和修复后的结果。方法机械搬移与语义修改分开；AST/源码或 `sync_deepseek_methods --check` 只证明所定义的文本映射，不代表行为等价。

探针记录实际对象/文件、实现选择、dtype、shape 与模式。社区算子通过不代表实际 vendor 分支通过；其他 M/专家数失败也不能直接替当前模型定因。调用点和 provider 不一致时先修正探针。

对量化算法区分理想 FP32 oracle、同量化输入/权重 oracle 与安装的 native 基线。运输层应保持原生结果与物理字节；原生近似误差不能靠放宽运输阈值掩盖，也不能把与 native 一致当成完整模型准确率。

验收器也需要真实失败样本。保存原始 JSON，验证数字/关键词偶然出现、控制字符、未闭合思考段或 length 截断都会失败；正常已知答案仍应通过。确定性短响应使用 [validate_generation_response.py](../../../tools/validate_generation_response.py)，开放题另选评分标准。

## 4. 状态与 Graph

输入 token、位置、cache 初态与采样对齐后，比较首次“输入相同、输出不同”的边界。状态等价与 batch-invariance 分开；基础路径、真实 prefix hit、decode hit、跨页/滑窗/淘汰与重算分别验证。用例名称不是命中证据。

Graph replay 不执行 Python forward hooks；capture 中不能以 `.item()` 等主机同步保存证据。探针应证明命中且保留原问题；在 capture 外读动态状态，用变输入/变状态 replay 对照。仅 eager 正确或关闭 Graph 后遇到其他错误，都不能给原 Graph 行通过。

## 5. 服务、基准与交接

服务需分别给初始化、owned ready、真实生成及清理证据。相同 prompt 的短响应、health=200、20 个 Graph 捕获完成或单 kernel 对拍，不替代其他层。资源、协议、构建环境与形态闭环统一见 [历史经验](../../../docs/development-methodology.md)。

bench 固定硬件/driver、依赖、SHA、输入输出分布、并发与正确性，分离冷编译/预热/稳定测量，报告相对基线。未涉及性能要求的修复不凭日志中的绝对 tok/s 宣称加速。

## 抽取方法与模块别名

将类方法抽为普通函数时，执行构造路径检查拥有类、receiver 与 `super`/`__class__`；保留装饰器行为，class/static/property 描述符另声明绑定。常量重写须区分标识符、属性名、字符串和注释，重复执行仍应保持结果；本地同名变量应先审阅作用域。对源类拆分拒绝猜测。

验证 post-apply 时预先导入实际消费者，检查旧别名是否刷新；同时断言缺失属性和无关模块不被写入。只检查 patch 后重新 import 会漏掉旧引用。本轮修复前后执行证据见 [实施记录](../../../docs/hcu-sglang-development.md#2026-10-26-3-故障复现与工具修正)。


### 设备专用优化的输入准入

若宿主平台探测用于选择设备 kernel，还需检查实际输入设备。参照 [CP remap 案例](../../../docs/development-methodology.md#11-设备能力与张量准入)，用 HCU 主机上的 CPU 输入验证参考路径不导入设备依赖，再用真实 HCU 输入验证优化分支及相关 Graph 数值；不能只 mock 平台为非 HCU 绕过该边界。


### 缓存池类与通信器集成

当差异覆盖构造、布局、状态和多个协作方法时，应以一致的对象契约接入，保留原拥有类的闭包，核对派生类继承和预导入消费者。可复用的官方类/utility 先比对实现及实际依赖，再验证完整构造、缓冲和下一消费者；不能只替换缺失常量来使导入通过。参见 [CP 父池与预算案例](../../../docs/hcu-sglang-development.md#2026-10-26-72-cp-父池容量与握手接线)。

通信可用性按实际生命周期判断。当前 PyNCCL 的默认 disabled 与作用域内启用分别检查，并断言真实 collective 数值；native all-reduce、PyNCCL 和模型 PD 传输的证据分别记录。新增 GPU 单测沿用显式 `HCU_PLUGIN_RUN_GPU_TESTS=1` 准入，默认 CPU 检查不触发这些设备数值用例。


### 描述符协议与复制计划

涉及 PD/分片缓存时，从公开状态生产入口检查编解码、peer 发布与最终 byte effects。对照应覆盖源/目标不同排序、同层不同 family、未拥有项、组件类型的重复次序、元数据数量和 ABI 错配；错误注册在任何写入前拒绝，重新注册撤销旧 verdict。目标组件已映射时，对应指针、stride、索引和描述符都沿同一个目标索引。

复制计划的参考结果从身份与原始缓冲独立计算。检查每个批次沿用同一注册快照，验证异常时仍等待已经启动的写入完成。可用有界 CPU engine 做局部反例，再以真实 HTTP/ZMQ、可用原生 transport 和实际缓冲复验；按各 lane 的实际范围记录，见 [描述符案例](../../../docs/development-methodology.md#13-描述符身份与位置式校验冲突)。


### Attention 类图与原生构建边界

若迁移涉及基类新增状态，检查派生类的真实基类与 class closure；模块别名更新不能重建已有继承关系。CP 空查询回归保留 prepare/prefetch/wait 到早退的顺序，后端每条独立索引读取路径分别检查且不改写共享 metadata。相关实证见 [类图案例](../../../docs/development-methodology.md#14-attention-类图与空查询通信顺序)。

JIT 故障核对最终编译器 include 顺序及其进入缓存键的时点，不只检查 extra include 或资源哈希。native dtype 拒绝先区分数值格式与 packed-byte carrier，用实际 provider 对照 bytes/pointer/stride；数值转换与 view 不是同一种操作。记录 oracle 的边界，详见 [头文件案例](../../../docs/development-methodology.md#15-jit-实际头文件优先级) 和 [原生接口案例](../../../docs/development-methodology.md#16-原生算子-provider-与-fp8-carrier)。


### CP 缓存的读取与写入所有权

涉及紧凑 staging 时，分别核对读 getter 与持久写 getter；非 owner 可能仍须产生复制的 latent、索引或 pair state。调用点、Python carrier 与 native 无写入分支须一起验证，不能只把 tensor 改成 None。真实池测试保留共享 staging 的前后字节，设备测试覆盖有/无主 KV 的 eager 与变输入 Graph；物理 page-major payload/scale 按各自跨度核对。实证见 [写入所有权案例](../../../docs/development-methodology.md#18-缓存读取-staging-与持久写入目标)。


### 前向记录与后端依赖图

后端新增状态跨越构造器、metadata 类型、候选发布/消费与执行记录时，核对完整依赖图及类型身份；class REPLACE 后仍需检查原派生类的实际基类。对新增 ForwardBatch 字段检查声明、默认、实际构造、dataclass 复制及序列化，区分它与冻结 ServerArgs；见 [运行记录案例](../../../docs/development-methodology.md#19-前向批次-schema-与复制生命周期)。

保留真实 pool/allocator 的映射绑定及公开 metadata 初始化，数学探针不能用手填 core 替代这些边界。native logits 的 oracle 应明确编码、FP32/BF16 累积及候选可见范围；对齐适配放在实际 producer→consumer 边界，底层非法输入仍按合同拒绝。相关实证见 [候选评分案例](../../../docs/development-methodology.md#20-候选评分的行对齐与消费边界) 与 [容器分片案例](../../../docs/development-methodology.md#21-python-序列分片与-tensor-高级索引)。真实 CP>1 验证同时核对拓扑、空计数参与、独立通信器及销毁；分片节点方法通过不能代替模型引擎或流式生命周期验收。

环境变量迁移测试使用当前官方 `envs` 对象及实际 descriptor，通过环境变量驱动分支；不要用 `raising=False` 给旧模块补不存在的配置属性，否则会掩盖真实导入错误。见 [环境变量对象案例](../../../docs/development-methodology.md#22-环境变量对象与测试替身的接口一致性)。

CP 数值对照出现偏差时，先比较实际 Q/KV/索引与有效长度，再以同卡、相同原生批次形状隔离 provider 的形状依赖；独立 FP32 参考与原生完整批次对照保留各自结论。显式记录参考分块选项，不通过隐藏更换参考或放宽阈值覆盖原始失败。见 [批次形状案例](../../../docs/development-methodology.md#23-原生计算批次形状与数值参考)。

服务与产物来源验证同时固定工作目录和 Python 搜索路径；预检进程与实际启动方式不同，应检查派生进程是否会优先找到当前目录中的旧包。记录实际模块路径及启动目录，不能只由分发名或预检结果推断模型进程来源。参见 [产物与来源案例](../../../docs/development-methodology.md#8-交付产物与模块来源验证)。

## 归一化的 residual 与量化返回契约

迁移 RMSNorm/Gemma 等方法时，同时核对独立输入、residual、post-residual、非连续视图、weight 与预计算 `1 + weight`、batch-invariant 以及可选 AITER 分支。数学近似相同不代表状态契约相同：检查来源要求的融合分支是否实际命中、哪些张量原地更新、调用方仍持有哪些别名；不要用 native 回退替代来源已支持的 HCU fused 路径。

若输出供量化投影使用，按真实消费者保留 FP8 tensor/scale 的 carrier、是否额外返回 residual，以及 `update_input` 等生产约定；显式未启用的可选能力须按来源拒绝。CPU kernel stub 只验证数据流和分支，另以真实 vendor 数学、Graph replay 和整体服务分别提供证据。单个类的选中方法对照保持选择范围，不能据此关闭整个文件的迁移项。

归一化包含 collective 时，核对 `tp`、`attn_tp`、`moe_tp` 的真实成员、world size 与 fallback，不能用 MoE EP 代替 MoE TP，也不能在融合拒绝时换回全 TP 组。同时保留计算后端与 communicator 策略的独立开关。验证预先导入的 public alias 是否随 Hook 更新，以及 quantized/BF16 side output 的生产消费；CPU 组标签测试与真实多进程通信分别记录。复用未变化的框架 coordinator 方法不等于验证了其完整构造、传输后端或 Graph 生命周期。


## 线性层显式量化与调用布局

迁移线性投影时同时验证公开签名、构造/partition字段、真实post-load权重布局、q/scale消费者能力与返回容器；不要只在模型私有方法直接调用quant_method来绕过公开接口缺口。显式carrier应保持来源定义的优先级，不能因fused开关关闭而丢弃，也不能把INT8交给不支持的FP8消费者。

保留输入/残差是否原地更新及epsilon来源，检查继承构造器的super owner和Merged普通/融合分支的不同返回。Source新增位置参数可能与旧output-buffer位置冲突，两种实际调用分别回归。GPU夹具应执行真实权重后处理，按最终布局计算参考；CPU标记流通过不能代替真实GEMM或Graph结果，各artifact版本分别记录。


## 选择性 backend fallback 与失败缓存

迁移fallback时保留来源的probe、异常类型/文本筛选、cache key和数学/scale/bias处理，不能只替换成一个能跑的后端。分别测成功不缓存、已知失败只probe一次、形状/device/dtype隔离及非预期错误原样抛出；裁剪/reshape、输出dtype、旧返回容器均沿公开入口检查。真实GPU测试分清自然触发的backend失败与预置已知cache的夹具，后者只能证明选中kernel的数值，不冒充实际设备probe失败。Graph replay应更新输入或scale后重验。


Triton bias/scale分支要检查SSA合并类型与公开API的mixed dtype/多维输出，显式store等价cast应保持计算顺序并单独记录为来源适配。迁移时不仅原全局名不能局部绑定，新增的provider/owner限定名也不能被维度变量遮蔽；例如函数局部N会令N.helper指向整数。抽取预检应在写文件前拒绝此类冲突，采用不冲突的明确namespace。


量化carrier必须沿真实公开consumer入口验证，并检查token flatten/padding与backend输出buffer行数。保留Tensor view的storage即可，不把新view对象identity误当重复量化；用禁止requant的反例和实际q/scale指针分别证明。遇到producer shape失败时核对来源的条件平台分支，不能只修下游reshape而继续使用不匹配的生产端。不得在GPU上试跑已知q/output行数不一致的越界配置；先用严格CPU backend契约捕获。


Block-FP8 backend迁移必须把加载布局和计算接口一并核对：checkpoint与GEMM的[N,K]/[K,N]方向、block scale的[N-block,K-block]/[K-block,N-block]方向及contiguous状态用非方形shape测试；从原生post-load到原生apply验证，不能只直调helper。auto选择需保留高优先级backend；仅为对应consumer改布局。数值参考应解码真实per-group量化结果，Graph replay改变输入后复验。


Dense mxscale迁移应检查capability、gfx95门控、shape白名单、权重shuffle之后的scale准备和全部consumer readiness。E8M0 scale的逻辑shape不等于其物理column-major字节排列；分别验证原始字节、独立量化数值与Graph输入更新。可运行的融合Triton生产端不能证明另一架构的AITER GEMM；vendor导入失败与CPU条件模拟应单独记录，不能关闭优化开关后宣称该分支通过。


Compressed FP8应通过native LinearMethod→scheme验证建参、post-load与apply；propagated dtype和optional量化参数必须在真实上层wrapper中断言。AROUND的legacy反射签名可能丢失新增kwargs，需限定owner提供实际参数元信息，其他scheme保留signature/cache过滤。LightOp opt-in保持OCP[N,K]与bias输出dtype一致，普通HCU保持其转置和per-token scale；compressed block也必须遵守所选backend的weight/scale双向布局。KME的策略夹具与真正自然失败分开记录。


模块差异审查不能只枚举顶层函数：递归检查conditional function/fake registration、lazy import、global初值和动态绑定。记录当前设备不命中的差异为pending，不能据此删掉范围；反向AST归一需限定具体适配。保留目标基线新增的正确性保护，来源较旧并不授权删除它们。通用wrapper回归至少覆盖一个真实的其他native scheme；独立SlimQuant全模型结果不能冒充该wrapper覆盖。


FP4/FP8 MoE迁移从native constructor/create/static factory到完整post-load检查；新dtype必须与实际runner、量化信息结构、weight与scale的布局成对接入。保留staticmethod/classmethod descriptor；通过caller's真实类验证。ASM布局用真实provider及独立逆排列检查所有bytes/Parameter storage；FP4转换用独立E2M1解码比较值与scale。加载初始化的layout/conversion测试不能宣称完整routed MoE GEMM或forward Graph通过，局部分支provider可用也不等于全局vendor配置可import。


真实MoE forward测试应调用完整插件注册入口，不能只登记量化方法而漏掉已有alignment/activation等依赖；用真实runtime context发布必要配置，TP1通信group显式声明。量化不连续边界会放大上游BF16微小舍入，应分阶段独立核对GEMM、activation和down/route，保留原门限；记录实际中间值供下阶段参考不等于替换实际kernel。共享测试的模型/metadata路径使用显式环境变量。


W8A8 strategy迁移同时核对[N,K]/[K,N]、packed stride、scale dtype、cache m bucket与真正provider；入口名称cutlass可能在HCU委托另一backend，按最终调用记录。Bias必须从真实native wrapper经过helper到结果，不能只检查参数存在。新scheme增加optional输入后提供其参数元信息；旧签名兼容单元测试可限定注册范围，但真实GPUforward必须完整插件注册。Static/AZP元数据与实际非对称量化数值、gfx928布局fixture与真实gfx928执行分开记录。


比较已存在的owned helper时要核对完整Source函数/全局，而非只核对本次调用的一个分支，防止漏掉fast path。会在缺binary时exit的provider必须在调用前按来源gate排除；异常fallback测试不能证明能捕获进程exit。GPU记录实际ASM/Triton计数和Graph更新，不能只凭名称推断后端。量化config target测试覆盖不含Linear名称的自定义子类及MRO/路径/fused优先级。


JIT资源审查调用[read-only资源工具](../../../tools/audit_jit_resources.py)，同时核对来源OID/mode、owned bytes、wheel成员与实际编译器依赖闭包。历史deps记录与.so存在不证明当前symbol执行；不得自动把其状态改completed。用native BuildSpec及translation unit检查真实source/include顺序；再在独立冷cache调用公共Python入口实际编译、算数/边界/Graph。HIP宏中的模板逗号问题应作精确语法适配，保留线程/occupancy/算法，并反向恢复来源bytes。


移植radix/warp kernel时逐项核对histogram bin归属与实际warp width，不只修复缺失intrinsic使其能编译。BF16 ordered compare要明确signed-zero/NaN/subnormal语义；使用整型位序验证极小值，先排除CPU参考flush模式，不能看到GPU/reference不等就直接判定GPU有错。保持原计算kernel、launch尺寸和quota算法，边界覆盖来源声明的max width/k及Graph页表更新，CUDA未实测需单独记录。

逻辑warp/硬件wave边界：先读实际JIT include中的kWarpThreads与shuffle width，再确定top-k容量和histogram bins；不能从gfx硬件wave宽度推断接口上限。零token须走真实公共JIT入口，验证host检查后不发射零grid；hash与共享header的mask分别覆盖，Graph中的设备token count应在重放时重新读取。

路由producer与consumer成对验证：新接入的JIT若要求FP32，先按原配方检查gate真实输出dtype和Source gate分发，不在consumer随意cast。gate→hash公共入口的GPU/Graph测试应保留真实GEMM/路由数学，再用完整服务捕获模型级缺口。FP32容器不等于FP32累加精度：按Source是否先输出BF16再转换建立独立参考；显式FP32 kernel与该兼容路径的数值契约分别记录。

完整路由模块迁入时，先核对共享config/output/enum/protocol类型是否与native相同；相同类型保留同一对象身份，再验证原class与owned factory的构造和真实消费者。LightOP融合、int64/padding与shared-slot重映射各走真实GPU入口；能力探测/导入成功不等于内核执行。Source旧函数名与实际定义不一致时修正调用并保存可反向恢复的精确patch，测试真实EPLB谓词而非用常量关闭融合。合成EP均衡必须与模型精度分开验收。

计数/采样Triton入口需覆盖共享列切片产生的非连续routed视图与drop/shared逻辑ID；用测试自有的相邻哨兵验证bounds，不能让无效ID触碰无关分配。Graph重放同时改变真实IDs与有效行数设备指针。metadata共享类型先核对完整fields：Source新增seed字段时用兼容子类和原factory桥接，不能误称完全相同。Source工厂kwargs必须与实际solver构造器核对；load-aware的无per-layer-collective设计不要替换成LP的all-reduce。

均衡solver不能只用手造replica表验收：必须从native元数据工厂走到真实solver，检查num_valid_copies/num_replicated，确认上游没有nearest折叠导致算法退化。不同算法的候选保留规则与静态policy分开验证；nearest保持旧行为，locality_fair保留完整候选并优先本GPU/本节点。保存旧函数用于差分时放在容器内，HookRegistry会替换模块级原函数别名，避免“旧/新”实际都调用新Hook。

EPLB统计需验证实际dispatch计数与top-k选择计数的差别：auto模式跨normal/low-latency累加，padding权重0但ID可能有效，不能用原始ID假装实际工作量。归约前先本地汇总固定shape，并用int64覆盖长窗口；真实多rank测试包括窗口步数不等、空窗口和超过int32总数。GPU capture测试需实际进入CUDAGraph，确认Python collect/host append被跳过，不能只mock capture状态。

Graph layout状态必须成对验收：capture完成后保存实际buffer layout，replay前恢复该layout再设置dispatch mode，插入另一模型layout后仍应回到原值。检查真实当前类名和完整producer入口，不以注册日志替代实际调用。方法迁出后既测object-return也测public file-output分支，并核对free-global解析；只测file helper不能证明调用者的time/Path等命名空间完整。

数据集精度见[固定GSM8K子集验收](../../../docs/validation.md#固定-gsm8k-子集与原始响应验收)：冻结数据、prefix/评分索引、采样与模型身份；当前题oracle不可进prompt。先验证HTTP/模型/正常stop，再按固定协议评分；缺失/错误/截断不得丢出分母或选择性重试。无接受阈值时只报告测量，不把六项QA或128题子集冒称通用精度或DAS回归比较。

成对精度对照需从原始结果独立复算并校验controller状态、清理、完整serving参数及优化环境。同数据/同分不足以代表同配置；功能变体与基础配方应标incomparable，分别建立匹配控制组。逐题报告单端正确/双方正确/双方错误，保持oracle原样，不因错题歧义、调试或重试提高已报告分数。

Graph捕获批大小迁移应从真实DllmConfig/算法到公开caller验证请求上限，并覆盖pool上限、默认值、TP/CP alignment、captured_req_width、TBO及compile列表；上限先参与ceil alignment，不用截断最终列表替代Source。预导入decode/runner别名需实际一致。配置元数据fixture与CPU列表计算不能宣称实际扩散模型Graph或精度通过；普通模型的八卡服务只能覆盖其自身路径。

Graph启动阶段的预算collective要按实际参与者选择组：只在最后PP stage存在的draft不得进入要求所有stage参与的world collective。保留native预算setter和共享initialized状态，测试另一PP组同时进入不同collective；模拟显存查询时仍执行真实Gloo归约并明确范围。混合attention/MoE模型检查全局layer索引尾部与局部区间的每个handler；缓存释放测试同时检查capture/finalize顺序及实际Graph变输入replay，不能把小型allocator测试称为H16模型验收。

预填充runner迁移应配对schedule vote与forward准入的context/sequence上限、backend vote；capture与replay共用一个scope owner。构造顺序测试完整执行owned构造器，明确昂贵base/buffer/backend夹具，验证预热先于backend初始化及super不重复初始化。replay同时检查model prepare、prepared IDs、encoder折叠元数据、草稿输出字段、层加载等待和异常恢复；真实GPU primitive Graph经公开execute通过不等同完整模型BCG或DP replay命中。

当Source函数新增必填参数，必须从固定官方版本实际生产caller执行，而不只直调新签名；修复前反例保留原caller并使用实际Hook后的callee。同步记录加列需同时迁移生产、fallback、归约、读取和声明的batch字段，以真实多进程通信检查宽度、mixed布局及子组global root。模型捕获完成但ready前调度退出仍为服务失败，保存原失败与清理，修正后在相同feature配置重跑。

Encoder replay折叠需覆盖原调度行与新增replay行的双向映射、尾部hidden indices、padding剔除、Engram history及KV plan身份；公开worker返回给下游的plan应属于原scheduled batch。legacy replay合批后核对MLP/logprob执行计数，fold保留原batch视图；小型GPU Graph的映射结果不代表compatible模型cache-hit。模型类型与feature约束须经resolve_once验证，不能仅解析CLI seed。

Prefill compute预算与KV预算分开验收：replay行计入compute并先从首chunk扣除，KV只收page-ceiled extend/headroom；exact fill与DLLM也需一致扣账。从公开adder保留旧负预算反例，并区分“本请求已提交、OTHER表示不能再接下一个”与未提交拒绝。新增Req字段reader配对memory-budget参数及固定native staged plan ABI；容量失败defer须覆盖idle与已有工作两种状态。配置增量优先具名seam，验证previous顺序、早退与实际dispatcher调用。

配置优化开关应在实际resolve_once前后核对环境值，不能只比较启动输入env；上游默认处理可能覆盖显式值。新增目标先查已有Hook与canonical owner，去重合同不能代替去掉重复注册；合并旧实现时验证legacy导出和预导入consumer身份。原型/最终产物分别记字节与验证，算法未变也不把原型实测冒称最终版本重跑。

窗口repair接口按component独立检查tombstone与索引、真实树split、其他组件字节、LRU与protected/evictable账本；替换caller前先接通这些后端。可选native后端沿现有factory延迟适配，保留自定义factory与子类，不因注册接口提前触发扩展编译。Rust单窗口限制与Python多窗口能力分别验证；owned plan可复制/序列化不等于staged producer/H2D已完成。

窗口plan生产与消费成套迁移：按组件裁剪缺失host范围、完整bounce归属到ACK、resident repair与tail insert分开验证；失败分配context回滚，排队后发布失败不能释放pending目标。使用真实树作为prefix/match读出，不能在夹具硬编码长度掩盖所有权错配。新增str Enum需检查旧成员str/hash及pickle兼容。GPU H2D fixture必须明确控制器/allocator/pub facade替代范围，使用pinned payload与stream完成event验证物理字节，不能自称native controller或L3模型验证。

## FileIO 分片状态与淘汰归属

MLA KV 可跨 TP rank 共用文件，Mamba 等 TP 状态仍须按 rank 分片。变更文件 suffix 或淘汰接口时，成对检查 backend constructor、lazy factory、metadata warmup、读命中 touch、reserve/commit/abort 及启动扫描；不能只改写入文件名。用独立临时目录和真实字节读写保留旧实现反例，覆盖非零 rank 的有界写入、共享文件可读但不计入其 LRU、跨 rank 淘汰隔离，以及 pending 写入/失败回滚。公共 batch IO 使用真实 backend；若 host pool 为数据夹具须明确其边界，不把该测试记作控制器线程或完整模型 L3 命中验证。

## KV 事件 namespace 的生产与消费

LoRA 名称登记须发生在请求的最终 extra_key 确定之后，并与树使用同一张表。覆盖旧 IPC 仅有 adapter ID 的回退、LoRA 与 base 的 hash 隔离、store/remove/parent 链一致、GPU removal 后延迟 host store，以及名称不同的事件不能合并。测试须保留原生公共请求 handler 与树事件出口；昂贵 admission/allocator 若为夹具，单独声明。跨 AROUND 层传递临时 request 上下文时，验证嵌套、异常 finally 恢复及非 HCU 的有效 original。Python recorder 通过不代表 Rust tagged event、发布 socket 或 router 消费已验证。

## Rust provider、事件 ABI 与宽 token hash

新增 owned crate 时成对检查 backend 预检和实际 loader，不让源码已随插件交付的后端因官方 manifest 缺失而错误回退。保留有效 original 的能力/自定义限制、显式 workspace、cache/build mode 和 features；用独立模块名区分旧 binding。验证来自 wheel 的 Cargo.lock/header/源码完整性和真实编译加载，不能用模拟 tuple 代替 Rust 生产事件。store/remove 必须携带同一 namespace，测试后消费 eviction 返回的索引并核对数量，不能忽略 destructor 的 ownership 报错。

改变 token 的编码范围时，Rust 与 C++/Python 输入转换须一起更新。覆盖 u32 边界、int64 最大值、负数拒绝、普通/bigram、逐页混合宽窄编码及 prior 链，用独立 hashlib oracle 与两种真实编译后端对照。普通 token 的历史哈希需保持一致；CPU binding 通过不外推为 GPU、L3 或完整模型验证。

### 原生 RoPE 的 keyword-only 状态

包装器区分 keyword-only 状态未传与显式 None，完整转发给当前有效 original，
保留旧接口；回归覆盖两种平台、对象身份和原生异常。diffusion 的独立 registry、
平台与真实 worker 分别取证，不以 SRT 注册结果替代。涉及 TP 配置时复制实际
rank/并行度并验证真实 group；局部接口、数值/Graph 与实际图片语义分别验收。

同一 diffusion callback 的入口别名须避免重复包装同一个 target/type/callable，
同时保留其他 Hook；覆盖双入口加载、registry reset 后重注册和注册异常传播。
不得以永久模块标志跳过 reset 后的注册。入口和部署约定见安装指南。
平台专用目标按已初始化的实际平台注册，分别覆盖 ROCm 与 OOT；不能通过
补造 vendor 属性让无关后端的 import/apply 看似成功。
