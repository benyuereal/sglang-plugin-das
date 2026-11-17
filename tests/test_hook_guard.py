# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Offline tests for the hook-registration guard.

No torch, no SGLang runtime: a stand-in ``sglang.srt.plugins.hook_registry``
module reproduces the behaviour the guard must react to -- an unresolvable
target raising inside ``_apply_target``.
"""

from __future__ import annotations

import logging
import os
import sys
import types
import unittest
from unittest import mock

from hcu_sglang import hook_guard


class _TargetMissing(Exception):
    """Stands in for the AttributeError/ImportError a bad target raises."""


def _build_fake_registry():
    """Minimal HookRegistry whose _apply_target fails on names starting 'absent'."""

    class HookType:
        BEFORE = "BEFORE"
        AFTER = "AFTER"
        AROUND = "AROUND"
        REPLACE = "REPLACE"

    class HookRegistry:
        _hooks: dict = {}
        _patched: set = set()
        applied: list = []

        @classmethod
        def _apply_target(cls, target, hooks):
            if target.startswith("absent"):
                raise _TargetMissing(f"no such target: {target}")
            cls.applied.append((target, len(hooks)))

        @classmethod
        def apply_hooks(cls):
            for target, hooks in sorted(cls._hooks.items()):
                if target in cls._patched:
                    continue
                try:
                    cls._apply_target(target, hooks)
                    cls._patched.add(target)
                except Exception:
                    pass

    return HookType, HookRegistry


def _install_fake_modules():
    """Put a fake sglang.srt.plugins.hook_registry package in sys.modules."""
    hook_type, registry = _build_fake_registry()
    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.plugins": types.ModuleType("sglang.srt.plugins"),
        "sglang.srt.plugins.hook_registry": types.ModuleType(
            "sglang.srt.plugins.hook_registry"
        ),
    }
    modules["sglang.srt.plugins.hook_registry"].HookType = hook_type
    modules["sglang.srt.plugins.hook_registry"].HookRegistry = registry
    modules["sglang.srt.plugins"].hook_registry = modules[
        "sglang.srt.plugins.hook_registry"
    ]
    return modules, hook_type, registry


class HookGuardTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(sys.modules)
        self.modules, self.hook_type, self.registry = _install_fake_modules()
        sys.modules.update(self.modules)
        hook_guard._installed = False
        hook_guard._skipped.clear()

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self.saved)
        hook_guard._installed = False
        hook_guard._skipped.clear()

    def test_absent_targets_are_skipped_and_present_ones_untouched(self):
        hook_guard.install()
        self.registry._hooks = {
            "present.a": [(self.hook_type.AROUND, lambda o, *a: o(*a), None)],
            "absent.b": [(self.hook_type.AROUND, lambda o, *a: o(*a), None)],
        }
        with self.assertLogs("hcu_sglang.hook_guard", level="INFO") as logs:
            self.registry.apply_hooks()

        # The present target went through the original path, unchanged.
        self.assertEqual(self.registry.applied, [("present.a", 1)])
        # The absent one was recorded, not raised, and reported in one summary.
        summary = [line for line in logs.output if "skipped" in line]
        self.assertEqual(len(summary), 1)
        self.assertIn("absent.b", summary[0])
        self.assertIn("_TargetMissing", summary[0])
        self.assertEqual(hook_guard._skipped, {})

    def test_several_absent_targets_share_one_summary_line(self):
        hook_guard.install()
        self.registry._hooks = {
            "absent.b": [(self.hook_type.AROUND, lambda o, *a: o(*a), None)],
            "absent.c": [(self.hook_type.AROUND, lambda o, *a: o(*a), None)],
        }
        with self.assertLogs("hcu_sglang.hook_guard", level="INFO") as logs:
            self.registry.apply_hooks()
        summary = [line for line in logs.output if "skipped 2 hook target" in line]
        self.assertEqual(len(summary), 1)

    def test_guard_is_idempotent(self):
        hook_guard.install()
        first = self.registry.__dict__["_apply_target"].__func__
        hook_guard._installed = False
        hook_guard.install()
        self.assertIs(self.registry.__dict__["_apply_target"].__func__, first)

    def test_disabled_by_environment(self):
        with mock.patch.dict(os.environ, {"HCU_HOOK_GUARD": "0"}):
            hook_guard.install()
        self.assertFalse(hook_guard._installed)
        self.assertEqual(
            self.registry.__dict__["_apply_target"].__func__.__name__,
            "_apply_target",
        )

    def test_multiple_around_hooks_are_preserved(self):
        """Both AROUND wrappers must survive: they self-gate per runner backend.

        Dropping either would disable a quantization path outright, so the guard
        must never dedupe them even though SGLang logs a notice.
        """
        hook_guard.install()
        self.registry._hooks = {
            "present.a": [
                (self.hook_type.AROUND, lambda o, *a: o(*a), None),
                (self.hook_type.AROUND, lambda o, *a: o(*a), None),
            ]
        }
        self.registry.apply_hooks()
        self.assertEqual(self.registry.applied, [("present.a", 2)])

    def test_known_pair_filter_silences_only_audited_targets(self):
        known = "sglang.srt.arg_groups.parallel_hook.handle_eplb_and_dispatch"
        unknown = "sglang.srt.future.Some.method"
        filt = hook_guard._KnownPairFilter()

        def record(msg):
            return logging.LogRecord(
                "sglang.srt.plugins.hook_registry", logging.WARNING, __file__, 1, msg, None, None
            )

        notice = "Multiple AROUND hooks on '%s' (2 hooks, from: a, b). If any AROUND hook skips calling original_fn, inner hooks will be bypassed."
        self.assertFalse(filt.filter(record(notice % known)))
        self.assertTrue(filt.filter(record(notice % unknown)))
        # Unrelated messages from the same logger are never touched.
        self.assertTrue(filt.filter(record("Failed to apply hooks to %s" % known)))

    def test_known_pair_filter_respects_environment(self):
        hook_logger = logging.getLogger("sglang.srt.plugins.hook_registry")
        saved = hook_logger.filters[:]
        hook_logger.filters = []
        try:
            with mock.patch.dict(os.environ, {"HCU_HOOK_KNOWN_PAIRS": "0"}):
                hook_guard._install_known_pair_filter(
                    self.modules["sglang.srt.plugins.hook_registry"]
                )
            self.assertFalse(
                any(isinstance(f, hook_guard._KnownPairFilter) for f in hook_logger.filters)
            )
        finally:
            hook_logger.filters = saved

    def test_apply_hooks_still_propagates_real_errors(self):
        """A non-target failure inside apply_hooks must not be swallowed."""
        hook_guard.install()

        def boom(cls):
            raise RuntimeError("unrelated failure")

        self.registry.apply_hooks = classmethod(boom)
        hook_guard._installed = False
        hook_guard.install()
        # The guard rebinds apply_hooks, but failures raised by the *body* of
        # raw_apply_hooks must propagate: only _apply_target is tolerant.
        with self.assertRaises(RuntimeError):
            self.registry.apply_hooks()


if __name__ == "__main__":
    unittest.main()
