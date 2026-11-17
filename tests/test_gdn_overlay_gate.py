# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The GDN overlay gate: SGLANG_GDN_HIP=1 plus an importable lightop."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from hcu_sglang.layers.attention.linear import gdn_commit, gdn_recurrent


class OverlayGateTest(unittest.TestCase):
    def gate(self, value, *, lightop=True, set_var=True):
        env = {gdn_recurrent._SWITCH_ENV: value} if set_var else {}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(gdn_recurrent, "_gdn",
                                  object() if lightop else None):
            return gdn_recurrent.overlay_enabled()

    def test_unset_and_off_values_disable(self):
        self.assertFalse(self.gate("", set_var=False))
        for value in ("", "0", "off", "none", "false", "no", " OFF "):
            with self.subTest(value=value):
                self.assertFalse(self.gate(value))

    def test_on_values_enable_when_lightop_is_importable(self):
        for value in ("1", "all", "on", "true", "yes", " 1 "):
            with self.subTest(value=value):
                self.assertTrue(self.gate(value))

    def test_an_unknown_value_is_on(self):
        self.assertTrue(self.gate("conv,prenorm"))

    def test_without_lightop_the_overlay_stays_off(self):
        self.assertFalse(self.gate("1", lightop=False))

    def test_the_commit_chain_follows_the_gate(self):
        with mock.patch.dict(os.environ, {gdn_recurrent._SWITCH_ENV: "1"}), \
                mock.patch.object(gdn_recurrent, "_gdn", object()):
            self.assertTrue(gdn_commit.enabled())
        with mock.patch.dict(os.environ, {gdn_recurrent._SWITCH_ENV: "1"}), \
                mock.patch.object(gdn_recurrent, "_gdn", None):
            self.assertFalse(gdn_commit.enabled())


if __name__ == "__main__":
    unittest.main()
