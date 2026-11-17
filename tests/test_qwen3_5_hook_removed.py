# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGET = "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention"


class Qwen35HookRemovedTests(unittest.TestCase):
    def test_plugin_entry_does_not_register_the_passthrough_hook(self):
        text = (ROOT / "hcu_sglang/__init__.py").read_text()
        # The MTP quant head shares the qwen3_5 module prefix; the passthrough
        # hook is identified by its target, not by the prefix.
        self.assertNotIn(TARGET, text)

    def test_no_module_registers_the_passthrough_self_attention_target(self):
        offenders = []
        for path in (ROOT / "hcu_sglang").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                value = node.args[0]
                if isinstance(value, ast.Constant) and value.value == TARGET:
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_declared_target_contract_no_longer_lists_it(self):
        import json

        declared = json.loads((ROOT / "tests/contracts/hcu_source_alignment_targets.json").read_text())[
            "expected_hook_targets"
        ]
        self.assertNotIn(TARGET, declared)


if __name__ == "__main__":
    unittest.main()
