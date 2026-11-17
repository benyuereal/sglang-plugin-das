# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SGLANG_GDN_HIP parsing and the capabilities each module derives from it."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from hcu_sglang.layers.attention.linear import gdn_commit, gdn_recurrent
from hcu_sglang.layers.attention.linear import gdn_switches as switches

DELIVERED = frozenset({
    switches.CONV, switches.PRENORM, switches.KERNEL,
    switches.BPLUS, switches.HANDOFF,
})


class ParseTest(unittest.TestCase):
    def setUp(self):
        switches._warned = False
        quiet = mock.patch.object(switches._LOG, "warning")
        quiet.start()
        self.addCleanup(quiet.stop)

    def parse(self, value, *, set_var=True):
        env = {switches.ENV: value} if set_var else {}
        env.update({
            name: "" for name in os.environ
            if name.startswith("SGLANG_GDN_HIP_")
        })
        with mock.patch.dict(os.environ, env, clear=True):
            return switches.active()

    def test_unset_and_off_values_disable_everything(self):
        self.assertEqual(self.parse("", set_var=False), frozenset())
        for value in ("", "0", "off", "none", "false", "no"):
            with self.subTest(value=value):
                self.assertEqual(self.parse(value), frozenset())

    def test_on_values_enable_the_delivered_stack(self):
        for value in ("1", "all", "on", "true", "yes", "ALL"):
            with self.subTest(value=value):
                self.assertEqual(self.parse(value), DELIVERED)

    def test_a_subset_is_an_isolation_arm(self):
        self.assertEqual(self.parse("conv,prenorm"),
                         frozenset({switches.CONV, switches.PRENORM}))
        self.assertEqual(self.parse("conv + prenorm"),
                         frozenset({switches.CONV, switches.PRENORM}))
        self.assertEqual(self.parse("kernel"), frozenset({switches.KERNEL}))

    def test_handoff_implies_bplus_but_not_the_reverse(self):
        self.assertEqual(self.parse("handoff"),
                         frozenset({switches.HANDOFF, switches.BPLUS}))
        self.assertEqual(self.parse("bplus"), frozenset({switches.BPLUS}))

    def test_replay_all_is_additive(self):
        self.assertEqual(self.parse("all,replay-all"),
                         DELIVERED | {switches.REPLAY_ALL})

    def test_unknown_tokens_are_ignored_not_fatal(self):
        self.assertEqual(self.parse("conv,typo"), frozenset({switches.CONV}))
        switches._LOG.warning.assert_called_with(
            "[gdn] ignoring unknown %s token(s): %s", switches.ENV, ["typo"]
        )

    def test_a_parsed_value_is_cached_by_the_raw_string(self):
        first = self.parse("all")
        second = self.parse("all")
        self.assertIs(first, second)


class CapabilityTest(unittest.TestCase):
    def setUp(self):
        switches._warned = False

    def _env(self, value):
        patch = mock.patch.dict(os.environ, {switches.ENV: value}, clear=False)
        patch.start()
        self.addCleanup(patch.stop)

    def test_kernel_predicates_need_the_operator_library(self):
        self._env("all")
        with mock.patch.object(gdn_recurrent, "_gdn", None):
            self.assertFalse(gdn_recurrent.enabled_kernel())
            self.assertFalse(gdn_recurrent.prenorm_enabled())
            self.assertFalse(gdn_recurrent.conv_enabled())
        with mock.patch.object(gdn_recurrent, "_gdn", object()):
            self.assertTrue(gdn_recurrent.enabled_kernel())
            self.assertTrue(gdn_recurrent.prenorm_enabled())
            self.assertTrue(gdn_recurrent.prenorm_gate_enabled())
            self.assertTrue(gdn_recurrent.conv_enabled())

    def test_the_prenorm_gate_ships_with_prenorm(self):
        self._env("kernel")
        with mock.patch.object(gdn_recurrent, "_gdn", object()):
            self.assertFalse(gdn_recurrent.prenorm_enabled())
            self.assertFalse(gdn_recurrent.prenorm_gate_enabled())
        self._env("prenorm")
        with mock.patch.object(gdn_recurrent, "_gdn", object()):
            self.assertTrue(gdn_recurrent.prenorm_gate_enabled())

    def test_commit_capabilities(self):
        for value, bplus, handoff in (
            ("all", True, True),
            ("bplus", True, False),
            ("handoff", True, True),
            ("conv", False, False),
            ("0", False, False),
        ):
            with self.subTest(value=value):
                self._env(value)
                self.assertEqual(gdn_commit.enabled(), bplus)
                self.assertEqual(gdn_commit.handoff_enabled(), handoff)

    def test_legacy_switches_are_ignored_and_reported(self):
        with mock.patch.dict(os.environ, {
            switches.ENV: "all",
            "SGLANG_GDN_HIP_BPLUS": "1",
        }, clear=True), mock.patch.object(switches._LOG, "warning") as warning:
            switches._warned = False
            self.assertTrue(switches.has(switches.CONV))
        warning.assert_called_once()
        self.assertIn("SGLANG_GDN_HIP_BPLUS", warning.call_args[0][1])


if __name__ == "__main__":
    unittest.main()
