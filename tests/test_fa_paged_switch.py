# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
INTERFACE = ROOT / "hcu_sglang/layers/attention/flashattention_interface.py"


def _env_defaults(path: Path):
    """(name -> default) for the dict literal installed by register_envs()."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) else None
        )
        if getattr(target, "id", None) != "_ENV_SPECS":
            continue
        defaults = {}
        for key, value in zip(node.value.keys, node.value.values):
            name = ast.literal_eval(key)
            if not name.startswith("SGLANG_HCU_FA_"):
                continue
            defaults[name] = ast.literal_eval(value.elts[1])
        return defaults
    raise AssertionError("_ENV_SPECS not found")


def _load_interface():
    """Load the interface module without importing the hcu_sglang package."""
    stubs = {}
    for name in ("hcu_sglang", "hcu_sglang.utils"):
        module = types.ModuleType(name)
        module.__path__ = []
        stubs[name] = module
    detection = types.ModuleType("hcu_sglang.utils.detection")
    detection.is_hcu = lambda: True
    detection.is_hcu_native_fp8_supported = lambda: True
    stubs["hcu_sglang.utils.detection"] = detection
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("fa_iface_under_test", INTERFACE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


class DirectPagedEligibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fi = _load_interface()

    def setUp(self):
        self._enabled = self.fi._paged_bshd_enabled
        self._layout = self.fi._layout_enabled
        self.fi._paged_bshd_enabled = lambda: True
        self.fi._layout_enabled = lambda: False
        self.addCleanup(self._restore)
        self.k = torch.zeros(8, 64, 4, 256)
        self.v = torch.zeros(8, 64, 4, 256)
        self.pt = torch.zeros(40, 2, dtype=torch.int32)
        self.lens = torch.full((40,), 100, dtype=torch.int32)

    def _restore(self):
        self.fi._paged_bshd_enabled = self._enabled
        self.fi._layout_enabled = self._layout

    def eligible(self, q_rows, mq, **overrides):
        kwargs = dict(
            q=torch.zeros(q_rows, 24, 256),
            k_cache=self.k,
            v_cache=self.v,
            cu_seqlens_q=torch.arange(0, q_rows + 1, mq, dtype=torch.int32),
            max_seqlen_q=mq,
            page_table=self.pt,
            cache_seqlens=self.lens,
            window_size=(-1, -1),
            softcap=0.0,
        )
        kwargs.update(overrides)
        return self.fi._direct_paged_eligible(**kwargs)

    def test_switch_is_off_by_default(self):
        defaults = _env_defaults(ROOT / "hcu_sglang/environ.py")
        self.assertEqual(defaults["SGLANG_HCU_FA_PAGED_BSHD"], False)
        self.fi._paged_bshd_enabled = lambda: False
        self.assertFalse(self.eligible(40, 1))

    def test_decode_and_mtp_verify_shapes_are_eligible(self):
        self.assertTrue(self.eligible(40, 1))
        self.assertTrue(self.eligible(200, 5))

    def test_prefill_row_mismatch_is_not_eligible(self):
        # 40 packed q rows but a 1-token chunk for each of 41 requests.
        cu = torch.arange(0, 42, dtype=torch.int32)
        self.assertFalse(self.eligible(40, 1, cu_seqlens_q=cu))
        self.assertFalse(self.eligible(200, 5, cu_seqlens_q=torch.arange(0, 42, 5, dtype=torch.int32)))

    def test_page_size_and_head_dim_must_match(self):
        self.assertFalse(self.eligible(40, 1, k_cache=torch.zeros(8, 32, 4, 256)))
        self.assertFalse(self.eligible(40, 1, v_cache=torch.zeros(8, 64, 4, 128)))

    def test_unsupported_window_and_softcap_fall_back(self):
        self.assertFalse(self.eligible(40, 1, window_size=(4095, 0)))
        self.assertFalse(self.eligible(40, 1, softcap=0.5))

    def test_legacy_layout_disables_the_bshd_route(self):
        self.fi._layout_enabled = lambda: True
        self.assertFalse(self.eligible(40, 1))

    def test_missing_tensors_fall_back(self):
        self.assertFalse(self.eligible(40, 1, page_table=None))
        self.assertFalse(self.eligible(40, 1, cache_seqlens=100))


if __name__ == "__main__":
    unittest.main()
