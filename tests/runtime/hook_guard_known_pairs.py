#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Prove the known-pair filter hides only the audited duplicates.

Loads the real SGLang ``hook_registry.py`` into a stub package and checks that:

* the "Multiple AROUND hooks" notice is suppressed for a target in
  SAFE_AROUND_PAIRS;
* the same notice still appears for a target that is NOT on the list;
* other warnings from that logger are untouched.

Usage:
    python tests/runtime/hook_guard_known_pairs.py --sglang-python <path/to/sglang/python>
"""

from __future__ import annotations

import argparse
import logging
import sys
import types
from pathlib import Path


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
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def apply_two_around(module, target):
    HookRegistry = module.HookRegistry
    HookType = module.HookType

    def around(original, *a, **kw):
        return original(*a, **kw)

    HookRegistry.reset()
    HookRegistry._patched = set()
    # json.dumps stands in as the enclosing object so the target resolves without
    # needing the quantization module; only the target string matters here.
    HookRegistry.register(target, around, HookType.AROUND)
    HookRegistry.register(target, around, HookType.AROUND)
    collector = Collect()
    log = logging.getLogger(module.__name__)
    log.addHandler(collector)
    saved = log.level
    log.setLevel(logging.DEBUG)
    try:
        # Fire the exact notice the guard filters on.
        log.warning(
            "Multiple AROUND hooks on '%s' (2 hooks, from: a, b). "
            "If any AROUND hook skips calling original_fn, inner hooks will be "
            "bypassed.",
            target,
        )
        log.warning("Something else entirely about %s", target)
    finally:
        log.setLevel(saved)
        log.removeHandler(collector)
    return collector.messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-python", type=Path, required=True)
    args = parser.parse_args()

    module = load_real_registry(args.sglang_python.resolve())
    from hcu_sglang import hook_guard

    known = "sglang.srt.arg_groups.parallel_hook.handle_eplb_and_dispatch"
    unknown = "sglang.srt.some.future.Target.method"

    hook_guard._installed = False
    hook_guard._skipped.clear()
    hook_guard.install()

    known_msgs = apply_two_around(module, known)
    unknown_msgs = apply_two_around(module, unknown)

    known_multi = [m for m in known_msgs if m.startswith("Multiple AROUND hooks")]
    known_other = [m for m in known_msgs if m.startswith("Something else")]
    unknown_multi = [m for m in unknown_msgs if m.startswith("Multiple AROUND hooks")]

    print("\n--- known pair (should be silenced) ---")
    print("  Multiple-AROUND notices:", len(known_multi))
    print("  other warnings         :", len(known_other))
    print("\n--- unknown target (should still warn) ---")
    print("  Multiple-AROUND notices:", len(unknown_multi))

    ok = (
        len(known_multi) == 0
        and len(known_other) == 1
        and len(unknown_multi) == 1
    )
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
