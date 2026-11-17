#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check of hook_guard against the real SGLang HookRegistry.

Loads the shipping ``sglang/srt/plugins/hook_registry.py`` from an SGLang
checkout (stdlib-only module) into a stub package, then asserts:

* an unresolvable target no longer produces an ERROR record (the traceback),
  but is reported in exactly one INFO summary;
* a resolvable target is still patched, and both of its AROUND wrappers survive
  -- the guard must not dedupe them.

Usage:
    python tests/runtime/hook_guard_real_registry.py --sglang-python <path/to/sglang/python>
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path


def load_real_registry(python_root: Path):
    """Import hook_registry.py from a checkout without importing all of sglang."""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-python", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    registry_module = load_real_registry(args.sglang_python.resolve())
    HookRegistry = registry_module.HookRegistry
    HookType = registry_module.HookType

    import json

    good = "json.dumps"
    absent = (
        "sglang.multimodal_gen.configs.pipeline_configs.wan."
        "WanI2V480PConfig.__post_init__"
    )

    calls = []

    def around_a(original, *a, **kw):
        calls.append("a")
        return original(*a, **kw)

    def around_b(original, *a, **kw):
        calls.append("b")
        return original(*a, **kw)

    from hcu_sglang import hook_guard

    hook_guard._installed = False
    hook_guard._skipped.clear()
    hook_guard.install()

    HookRegistry.reset()
    HookRegistry.register(good, around_a, HookType.AROUND)
    HookRegistry.register(good, around_b, HookType.AROUND)
    # Two targets the build does not provide, both unresolvable.
    HookRegistry.register(absent, around_a, HookType.AROUND)
    HookRegistry.register(absent + "_x", around_b, HookType.AROUND)

    collector = Collect()
    root = logging.getLogger()
    root.addHandler(collector)
    try:
        HookRegistry.apply_hooks()
    finally:
        root.removeHandler(collector)

    exceptions = [r for r in collector.records if r.levelno >= logging.ERROR]
    summaries = [r for r in collector.records if "hcu hook guard" in r.getMessage()]

    # Exercise the patched function: both AROUND wrappers must run.
    json.dumps({"k": 1})

    print("\n--- observed ---")
    print("ERROR/traceback records :", len(exceptions))
    for record in exceptions:
        print("   ", record.getMessage()[:120])
    print("guard summary records   :", len(summaries))
    for record in summaries:
        print("   ", record.getMessage()[:200])
    print("good target patched     :", good in HookRegistry._patched)
    print("wrappers that ran       :", calls)

    ok = (
        len(exceptions) == 0
        and len(summaries) == 1
        and good in HookRegistry._patched
        and calls == ["b", "a"]
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
