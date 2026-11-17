# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pin the fork point of the vendored GDN backend.

``gdn_source.forward_decode`` is sglang's target-decode loop plus the HCU
deltas; ``forward_extend``/``_verify_kernel_forward``/``_convolve_prefill``
are the plugin's overlay. An upstream sync changes the copied body, and these
hashes make that change show up as a failing test instead of silent drift.

Update after a reviewed sync: run this file directly, paste the new digests,
and say which upstream commit the copy now matches.

    python3 tests/test_gdn_source_fork_point.py
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import unittest

SOURCE = (Path(__file__).resolve().parents[1]
          / "hcu_sglang/layers/attention/linear/gdn_source.py")

PINNED = {
    "forward_decode": "e991f711b27f57102cd81e32a9a6fa945243e73937c5387f1541ff48996d0121",
    "_convolve_prefill": "872dd8b3633f91a1299d7743537b2423734fa1707f57ce9c06d3924ed5b9a872",
    "forward_extend": "1acc74f7e3822ea039ce0e2662252691cfaa0351ed498f704e8dc34bd84eaa11",
    "_verify_kernel_forward": "9fc00333fd2eb0fd0bfa4aab7a0fd1be7dada247956cade690e55c3879fc75e9",
}

# The deltas that make the copy ours; losing one of these is the regression.
MARKERS = {
    "forward_decode": "conv_update(",
    "_convolve_prefill": "initial_states=conv_states",
    "forward_extend": "_UPSTREAM_FORWARD_EXTEND[0](",
    "_verify_kernel_forward": "gdn_commit.verify_step(",
}


def _functions():
    tree = ast.parse(SOURCE.read_text())
    return {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _digest(node) -> str:
    return hashlib.sha256(
        ast.dump(node, include_attributes=False).encode()
    ).hexdigest()


class ForkPointTest(unittest.TestCase):
    def setUp(self):
        self.functions = _functions()

    def test_the_copied_and_overlay_definitions_are_pinned(self):
        for name, expected in PINNED.items():
            with self.subTest(definition=name):
                self.assertIn(name, self.functions)
                self.assertEqual(_digest(self.functions[name]), expected)

    def test_each_definition_keeps_its_hcu_delta(self):
        for name, marker in MARKERS.items():
            with self.subTest(definition=name):
                self.assertIn(marker, ast.unparse(self.functions[name]))


if __name__ == "__main__":
    nodes = _functions()
    for name in sorted(PINNED):
        print(f'    "{name}": "{_digest(nodes[name])}",')
