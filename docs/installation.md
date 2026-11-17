# HCU 插件安装与启用指南

> **本文用途**：安装当前 `hcu-sglang-plugin`，核对 `hcu` 入口和实际加载来源，再按模型配方启动服务。本文命令均使用当前包；旧 `sglang_hcu_plugin` 的历史产物与验证保留在 [V3 实践记录](v3-runtime-validation.md) 和 [2026-10-26 交付记录](hcu-sglang-development.md#2026-10-26-v3-2026-10-26-实施与验证记录)。

## 使用 Skill 安装与验证插件

对应 Skill：[`hcu-plugin-install`](../skills/hcu-plugin-install/SKILL.md)。输入包括官方与插件版本、源码或 wheel 路径/哈希、Python 环境、安装/运行用户、可写安装目录和证据目录；要求模型服务时再提供模型及完整配方。

```text
读取 <仓库绝对路径>/skills/hcu-plugin-install/SKILL.md，按该 Skill 执行。
官方源码及版本：<路径和 SHA>。
插件产物：<源码路径及 SHA，或 wheel 路径及 SHA256>。
目标 Python 环境：<环境>；安装方式：<wheel 或 editable>。
安装用户 / 运行用户：<身份>；安装/缓存目录：<可写路径>。
证据目录：<路径>；服务验收范围：<不要求，或模型与完整配方>。
核验解释器、distribution、入口、实际来源和加载；分别记录已验证与未执行项。
```

预期交付为安装、入口、设备、加载检查及要求的服务结果。入口存在不等于加载成功，health 200 不等于答案正确。

## 当前仓库安装入口（2026-10-28 起）

| 对象 | 当前名称 |
|---|---|
| distribution / pip 安装与卸载名 | `hcu-sglang-plugin` |
| Python 包 / 模块启动名 | `hcu_sglang` |
| `sglang.srt.plugins` 入口 | `hcu = hcu_sglang:register` |
| `sglang.multimodal_gen.plugins` 入口（H3 候选） | `hcu = hcu_sglang.multimodal_gen:register` |
| 启用白名单 | `SGLANG_PLUGINS=hcu` |
| 服务入口 | `python3 -m hcu_sglang` 或 `hcu-sglang-serve` |

旧 `hcu` 不是当前入口或别名；仅修改环境变量不能将旧 wheel 变成当前包。当前模块启动入口负责 AITER 兼容初始化，不安装全局 `usercustomize.py`。

## 1. 运行环境与兼容基线

在已具备兼容官方 SGLang、HCU/DTK/PyTorch HIP 和 vendor 依赖的环境中执行。插件要求 Python 3.10+；官方 SGLang 及其构建依赖还有各自要求。`--no-deps` 只安装插件，不准备 Torch、AITER、LightOp、DeepEP、vLLM 扩展或 Rust 工具链，也不保证任意新机器可用。

当前 dev34 的固定官方基线为 v0.5.22、`baa9befa381511cd4d56f83980951e5a45fffc6d`；代码提交为 `86f3f70ac72e673627c816f546eaf647cb045cf9`。历史 wheel、加载合同和四卡模型结果见 [开发迁移记录](hcu-sglang-development.md)。模型结果分属不同插件版本，不能整体外推为当前版本的完整验收。

以下命令在**实际运行服务的同一环境内**执行。先按 [路径配置](configuration.md#1-当前工具的路径配置) 设置 `REPO`、`OFFICIAL`、`PLUGIN`、`ARTIFACTS`；本仓就是安装对象时，`PLUGIN="$REPO"`。

```bash
: "${REPO:?请设置本仓路径}"
: "${OFFICIAL:?请设置官方源码路径}"
export PLUGIN="$REPO"
: "${ARTIFACTS:?请设置本次证据目录}"
mkdir -p "$ARTIFACTS"
python3 --version
python3 -m pip --version
git -C "$OFFICIAL" rev-parse HEAD
git -C "$PLUGIN" rev-parse HEAD
```

使用节点允许的安装/服务用户，并确认设备访问权限。镜像中已安装的 SGLang 可能不是目标官方版本，后续须核对实际来源。

## 2. 安装方式与隔离配置

wheel 或 editable 二选一。下面使用独立 target，不修改共享 site-packages；同一个 SITE 用于后续检查与服务，重新安装使用新目录。

```bash
export SITE=$(mktemp -d)
```

默认缓存目录可写时无需额外配置。若运行用户没有可写缓存，按实际需要设置 `XDG_CACHE_HOME`、`AITER_JIT_DIR`、`TORCH_EXTENSIONS_DIR`、`TRITON_CACHE_DIR`、`TORCHINDUCTOR_CACHE_DIR`、`SGLANG_CACHE_DIR`。AITER 的目录不由 XDG 自动替代。编译扩展时另检查对应 Rust/DTK 工具链。

### 2.1 Wheel 安装

可使用已有 wheel，或从干净提交构建本仓。源码构建示例：

```bash
BUILD=$(mktemp -d)
git -C "$PLUGIN" archive HEAD | tar -x -C "$BUILD"
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir "$ARTIFACTS/wheels" "$BUILD"
```

`git archive HEAD` 不包含未提交修改。将 WHEEL 设置为实际构建或交付的 wheel 绝对路径，核对版本和 SHA256 后安装：

```bash
: "${WHEEL:?请设置本次 hcu_sglang_plugin wheel 路径}"
test -f "$WHEEL"
sha256sum "$WHEEL"
python3 -m pip install --no-deps --target "$SITE" "$WHEEL"
export PYTHONPATH="$OFFICIAL/python:$SITE"
export SGLANG_PLUGINS=hcu
```

wheel 验证不要把 `$PLUGIN` 源码加入 PYTHONPATH，避免遮蔽安装产物。只有将匹配的官方 SGLang 和 plugin 都正常安装进同一个 Python 环境时，才可不设该 PYTHONPATH；仅 `--target` 安装 wheel 仍需加入 SITE。

### 2.2 Editable 安装

在源码所属用户及已有构建依赖的环境执行：

```bash
python3 -m pip install --no-deps --no-build-isolation --target "$SITE" -e "$PLUGIN"
export PYTHONPATH="$OFFICIAL/python:$PLUGIN:$SITE"
export SGLANG_PLUGINS=hcu
```

target 中的 editable `.pth` 不会仅因加入 PYTHONPATH 自动执行，因此显式保留源码和入口 metadata。源码修改会影响后续新进程。安装用户与服务用户不同时，继续使用同一 SITE 和 Python 环境，并确认文件可读、缓存可写。

## 3. 安装、入口与加载验证

先检查实际来源和入口，不启动模型：

```bash
python3 - <<'PYCODE'
import sys
from importlib.metadata import distribution
import sglang
import hcu_sglang
pkg = distribution("hcu-sglang-plugin")
eps = [ep for ep in pkg.entry_points if ep.group == "sglang.srt.plugins"]
print("python:", sys.executable)
print("official:", sglang.__file__)
print("plugin:", hcu_sglang.__file__)
print("version:", pkg.version)
print("entry points:", [(ep.name, ep.value) for ep in eps])
assert [(ep.name, ep.value) for ep in eps] == [("hcu", "hcu_sglang:register")]
PYCODE
```

wheel 的包应来自 `$SITE/hcu_sglang`，editable 应来自 `$PLUGIN/hcu_sglang`；官方应来自指定 OFFICIAL。再检查实际服务用户的设备权限：

```bash
python3 - <<'PYCODE'
import torch
print("HIP:", torch.version.hip, "available:", torch.cuda.is_available(),
      "visible:", torch.cuda.device_count())
assert torch.version.hip and torch.cuda.is_available(), "运行用户尚不能初始化 HCU"
PYCODE
```

在依赖齐全的环境执行真实 register/apply 合同。必须选择 `hcu_sglang`，并在导入 native 依赖前初始化 AITER 兼容：

```bash
# wheel 使用 SITE；editable 改为 PLUGIN_ROOT="$PLUGIN"。
PLUGIN_ROOT="$SITE"
# 合同必须来自与 wheel 相同版本的源码；自行构建时可使用上面的 BUILD。
CONTRACT="$PLUGIN/tests/contracts/hcu_source_alignment_targets.json"
python3 - "$REPO" "$OFFICIAL" "$PLUGIN_ROOT" "$CONTRACT" "$ARTIFACTS/runtime.json" <<'PYCODE'
from pathlib import Path
import runpy
import sys
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
repo, official, plugin_root, contract, output = sys.argv[1:]
tool = str(Path(repo) / "tools/validate_plugin_runtime.py")
sys.argv = [tool, "--official-python", str(Path(official) / "python"),
            "--plugin-root", plugin_root, "--plugin-package", "hcu_sglang",
            "--expected-targets", contract, "--output", output]
runpy.run_path(tool, run_name="__main__")
PYCODE
```

以对应版本完整合同为准：目标非空、全部 applied、三个具名 seam 齐全、来源正确且 `errors=[]`。dev34 的历史结果为 385/385 targets 和 3 个 seams；旧版本的 208/363 等数字不作为当前合同。冷缓存可能触发 JIT，观察中断后先检查原进程，避免并发重启。

## 4. 模型服务启动与验收

保留安装时的环境，按模型使用完整配方。下面仅展示入口，不能代替模型的 TP、backend、量化、Graph、显存等参数：

```bash
: "${MODEL:?请设置本次模型目录}"
export SGLANG_PLUGINS=hcu
python3 -m hcu_sglang --model-path "$MODEL" --port 30001
```

先确认 GPU、容量与端口，保留完整启动参数。验证监听进程归属、`/health`、`/v1/models` 和真实请求；按 [响应校验](validation.md#7-确定性生成响应的离线验收) 检查答案、模型身份与正常 stop。服务后清理自己创建的进程。历史四卡配方与实际结果见 [开发迁移记录](hcu-sglang-development.md)。

### 4.1 gfx936 上的 page size 约束

HCU gfx936 平台的 paged FlashAttention kernel 要求 `page_size` 为 64 的倍数。用默认 page size 启动时，extend 路径会在 `vllm_flash_attn_varlen_func` 内直接失败：

```text
RuntimeError: Paged KV cache block size must be divisible by 64 at gfx936 platform
```

启动参数显式加 `--page-size 64`（或其它 64 的倍数）。该约束与模型规模无关：单卡 0.6B 冒烟同样触发，加上 `--page-size 64` 后正常 ready。

## 5. 停用、卸载与常见问题

`SGLANG_PLUGINS` 是入口白名单；在 pinned loader 中，未设置或空字符串会发现所有插件，不能用于停用。新进程中若需屏蔽 general plugins，可使用确认不存在的名称，例如 `SGLANG_PLUGINS=__no_general_plugins__`；保留部署所需的 adapter/platform 基线。

隔离 target 停用时从环境中移除该 SITE/源码路径。只有用户要求卸载个人环境中的正常安装包时，才执行 `python3 -m pip uninstall hcu-sglang-plugin`；不要在共享环境贸然卸载。已有进程不受安装或环境变量变化影响，需新进程验证。

| 失败 | 下一步 |
|---|---|
| 没有 wheel | 从本仓干净提交构建，或获取匹配版本产物 |
| import 成功但没有 `hcu` 入口 | 核对同一 Python 的 metadata，不只检查 PYTHONPATH |
| 入口存在但 register/apply 失败 | 查首条 soft error、实际来源和完整目标合同 |
| 模型启动失败、乱码、hang | 转 [故障诊断](troubleshooting.md)，不反复重装掩盖错误 |

新增独立插件见 [最小案例](first-plugin.md)，开发本仓能力见 [开发指南](plugins.md)。


## MiniMax-H3 Diffusion 候选入口（2026-11-09）

此段是进行中的 H3 升级候选，不构成设备数值、模型或性能验收。官方 0.5.22 的 Diffusion 使用独立入口组和独立 HookRegistry；SRT register/apply 合同不能替代 Diffusion 合同。Diffusion 入口仅初始化 lazy AITER 兼容，注册 attention、H3 与 QK-Norm/RoPE 目标，不导入全部 SRT 插件。SRT 入口仍为 `hcu`；Diffusion 同时接受 `hcu` 和 `hcu_diffusion`，均受官方 `SGLANG_PLUGINS` 白名单控制。两个 Diffusion 名称同时加载时，同一 target、Hook 类型和 callable 只注册一次；不同来源的 Hook 保留，registry reset 后可重新注册。两个 ROCm/AITER 后端选择 Hook 仅在已初始化的 ROCm 平台注册；HCU OOT 不使用它们，也不补造 vendor 能力。

安装修改后的 wheel 后核对 importlib.metadata 中的 sglang.multimodal_gen.plugins 入口；已有旧 wheel 不会随源码修改自动更新。使用官方 `sglang serve --model-type diffusion`，不要使用本仓面向 SRT 的模块启动器代替它。H3 当前配方显式设置 `SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rocm` 选择实际 HIP 后端，避免原有 amdsmi 探测在 HCU 上失败后落到 CPU；既有平台检测 Hook 在平台初始化之后应用，不能承担初始化前的选择。该配方尚待真实父子进程验证，不宣称自动平台发现已修复。

新旧正式比较均须设置 `FLASH_ATTENTION_FAST_SOFTMAX=1`，同时保留 Cookbook 中的量化、offload、SDPA 和 VAE streaming 选项。本文不以入口存在或 CPU 合同通过宣称模型验收完成。安装对应 Skill、输入与交付边界仍按本文开头执行。

## MiniMax-H3 的 diffusion 入口

安装本仓及 adapter 后显式设置 `SGLANG_PLUGINS=hcu_diffusion`。社区
`5792cf6e5e11` 的图片 worker 使用独立入口组 `sglang.multimodal_gen.plugins`；
分别核验主进程与 worker 的 entry point、diffusion registry 和已应用 targets。
原有 SRT `hcu` 入口保持不变，安装成功不等于模型验收通过。
已验证范围见 [开发记录](hcu-sglang-development.md#2026-11-14-minimax-h3-image-原生-hcu-验收)；
完整部署参数与回退统一见 adapter-das 的 `docs/image-dtk2610.md` §8。
