# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SGLANG_GDN_HIP is one binary switch, resolved to an immutable mode."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from hcu_sglang.layers.attention.linear import gdn_commit
from hcu_sglang.layers.attention.linear import gdn_switches as switches


class ParseTest(unittest.TestCase):
    def setUp(self):
        switches._warned_legacy = False
        quiet = mock.patch.object(switches._LOG, "warning")
        quiet.start()
        self.addCleanup(quiet.stop)

    def mode(self, value, *, set_var=True):
        env = {switches.ENV: value} if set_var else {}
        with mock.patch.dict(os.environ, env, clear=True):
            return switches.mode()

    def test_unset_and_off_values_disable_the_stack(self):
        self.assertFalse(self.mode("", set_var=False).enabled)
        for value in ("", "0", "off", "none", "false", "no", " OFF "):
            with self.subTest(value=value):
                self.assertFalse(self.mode(value).enabled)

    def test_on_values_enable_the_stack(self):
        for value in ("1", "all", "on", "true", "yes", " 1 "):
            with self.subTest(value=value):
                self.assertTrue(self.mode(value).enabled)
                switches._LOG.warning.assert_not_called()

    def test_an_unknown_value_is_on_with_one_warning(self):
        self.assertTrue(self.mode("conv,prenorm").enabled)
        switches._LOG.warning.assert_called_once()
        self.assertIn("treating it as on", switches._LOG.warning.call_args[0][0])

    def test_the_parsed_mode_is_cached_by_raw_value(self):
        first, second = self.mode("1"), self.mode("1")
        self.assertIs(first, second)

    def test_legacy_switches_are_ignored_and_reported(self):
        with mock.patch.dict(os.environ, {
            switches.ENV: "1",
            "SGLANG_GDN_HIP_BPLUS": "1",
        }, clear=True), mock.patch.object(switches._LOG, "warning") as warning:
            switches._warned_legacy = False
            self.assertTrue(switches.enabled())
        warning.assert_called_once()
        self.assertIn("SGLANG_GDN_HIP_BPLUS", warning.call_args[0][1])

    def test_the_commit_chain_follows_the_switch(self):
        for value, expected in (("1", True), ("0", False)):
            with self.subTest(value=value), \
                    mock.patch.dict(os.environ, {switches.ENV: value}):
                self.assertEqual(gdn_commit.enabled(), expected)


if __name__ == "__main__":
    unittest.main()
