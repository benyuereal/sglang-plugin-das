#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Replay the 2026-11-09 startup hook targets against the real HookRegistry.

Registers exactly the 15 targets that failed during that PD run (plus one
resolvable control) and compares unguarded vs guarded ``apply_hooks()``:

* unguarded: one ERROR record (with traceback) per failing target;
* guarded:   zero ERROR records, one INFO summary naming every skipped target,
             and the control target still patched.

Usage:
    python tests/runtime/hook_guard_replay.py --sglang-python <path/to/sglang/python>
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path

# The 15 targets that logged "Failed to apply hooks to ..." on 2026-11-09,
# each once per process. 14 are multimodal/video-generation surfaces.
FAILED_TARGETS = (
    "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.LTX2PipelineConfig.supports_dynamic_batching",
    "sglang.multimodal_gen.configs.pipeline_configs.wan.Wan2_2_T2V_A14B_Config.__post_init__",
    "sglang.multimodal_gen.configs.pipeline_configs.wan.Wan2_2_TI2V_5B_Config.__post_init__",
    "sglang.multimodal_gen.configs.pipeline_configs.wan.WanI2V480PConfig.__post_init__",
    "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerImpl.forward_varlen",
    "sglang.multimodal_gen.runtime.layers.attention.turbo_layer.MinimalA2AAttnOp.__init__",
    "sglang.multimodal_gen.runtime.layers.attention.turbo_layer.MinimalA2AAttnOp.forward",
    "sglang.multimodal_gen.runtime.models.dits.ltx_2.LTX2Attention.forward",
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3.MiniMaxH3DiTModel.post_load_weights",
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_qk_norm",
    "sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_rope_qk",
    "sglang.multimodal_gen.runtime.models.dits.wanvideo.WanTransformer3DModel.__init__",
    "sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor.PipelineExecutor.profile_execution",
    "sglang.multimodal_gen.runtime.server_args.server_args.ServerArgs._adjust_attention_backend",
    "sglang.srt.speculative.eagle_utils._sync_eagle_verify_outputs",
)

CONTROL_TARGET = "json.dumps"


def load_real_registry(python_root: Path):
    source = python_root / "sglang" / "srt" / "plugins" / "hook_registry.py"
    if not source.is_file():
        raise SystemExit(f"hook_registry.py not found under {python_root}")
    for name in ("sglang", "sglang.srt", "sglang.srt.plugins"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules.setdefault(name, module)
    module = types.ModuleType("sglang.srt.plugins.hook_registry")
    module.__file__ = str(source)
    sys.modules["sglang.srt.plugins.hook_registry"] = module
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), module.__dict__)
    return module


class Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def run(module, use_guard: bool):
    HookRegistry = module.HookRegistry
    HookType = module.HookType

    def around(original, *a, **kw):
        return original(*a, **kw)

    if use_guard:
        from hcu_sglang import hook_guard

        hook_guard._installed = False
        hook_guard._skipped.clear()
        hook_guard.install()
    else:
        from hcu_sglang import hook_guard

        hook_guard._installed = False
        hook_guard._skipped.clear()

    HookRegistry.reset()
    HookRegistry._patched = set()
    for target in FAILED_TARGETS:
        HookRegistry.register(target, around, HookType.AROUND)
    HookRegistry.register(CONTROL_TARGET, around, HookType.AROUND)

    collector = Collect()
    root = logging.getLogger()
    root.addHandler(collector)
    try:
        HookRegistry.apply_hooks()
    finally:
        root.removeHandler(collector)

    errors = [r for r in collector.records if r.levelno >= logging.ERROR]
    summaries = [
        r
        for r in collector.records
        if r.levelno in (logging.INFO, logging.WARNING)
        and "hcu hook guard: skipped" in r.getMessage()
    ]
    patched = CONTROL_TARGET in HookRegistry._patched
    # Every failed target is also logged once as "Failed to apply hooks to".
    failed_lines = [
        r for r in collector.records if "Failed to apply hooks to" in r.getMessage()
    ]
    return {
        "errors": len(errors),
        "failed_lines": len(failed_lines),
        "summaries": len(summaries),
        "summary_text": summaries[0].getMessage() if summaries else "",
        "patched": patched,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-python", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.CRITICAL)  # quiet console
    module = load_real_registry(args.sglang_python.resolve())

    # Root level must let ERROR through: basicConfig above sets the root
    # threshold, which would otherwise filter the very records under test
    # before any handler sees them.
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, root.handlers[:]
    root.setLevel(logging.DEBUG)
    root.handlers = []
    try:
        before = run(module, use_guard=False)
        after = run(module, use_guard=True)
    finally:
        root.setLevel(saved_level)
        root.handlers = saved_handlers

    print("\n=== unguarded (what the 2026-11-09 logs show) ===")
    print("  'Failed to apply hooks' lines :", before["failed_lines"])
    print("  ERROR records with traceback  :", before["errors"])
    print("  control target patched        :", before["patched"])

    print("\n=== guarded ===")
    print("  'Failed to apply hooks' lines :", after["failed_lines"])
    print("  ERROR records with traceback  :", after["errors"])
    print("  INFO summary lines            :", after["summaries"])
    print("  control target patched        :", after["patched"])
    print("\n  summary:", after["summary_text"])

    ok = (
        before["errors"] == len(FAILED_TARGETS)
        and before["failed_lines"] == len(FAILED_TARGETS)
        and after["errors"] == 0
        and after["failed_lines"] == 0
        and after["summaries"] == 1
        and after["patched"]
        and all(t in after["summary_text"] for t in FAILED_TARGETS)
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
