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


class HndPageViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fi = _load_interface()

    def test_contiguous_nhd_pool_is_not_hnd(self):
        nhd = torch.zeros(8, 64, 4, 256)
        self.assertFalse(self.fi._is_hnd(nhd))

    def test_single_kv_head_nhd_is_not_hnd(self):
        nhd = torch.zeros(8, 64, 1, 256)
        self.assertFalse(self.fi._is_hnd(nhd))

    def test_permuted_hnd_pool_is_detected_and_restored(self):
        base = torch.arange(8 * 4 * 64 * 256, dtype=torch.float32).view(8, 4, 64, 256)
        view = base.permute(0, 2, 1, 3)
        self.assertTrue(self.fi._is_hnd(view))
        k, v = self.fi._to_hnd_buffers(view, view)
        self.assertTrue(k.is_contiguous() and v.is_contiguous())
        self.assertEqual(tuple(k.shape), (8, 4, 64, 256))
        self.assertTrue(torch.equal(k, base))


if __name__ == "__main__":
    unittest.main()
