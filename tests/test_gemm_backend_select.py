# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The GEMM seam's selection rules: env precedence, roles, and fallbacks."""

from __future__ import annotations

import os
import types
import unittest
from unittest import mock

from hcu_sglang.layers.gemm import backend


class _SeamBase(unittest.TestCase):
    def setUp(self):
        providers = mock.patch.object(backend, "_providers", {})
        providers.start()
        self.addCleanup(providers.stop)

    def _env(self, **values):
        patch = mock.patch.dict(os.environ, values, clear=True)
        patch.start()
        self.addCleanup(patch.stop)


class EnvTest(_SeamBase):
    def test_selected_name_precedence_and_legacy_switch(self):
        self._env()
        self.assertEqual(backend.selected_name(), "none")
        self._env(HBIRD_GEMM_SGLANG="1")
        self.assertEqual(backend.selected_name(), "hbird")
        self._env(HBIRD_GEMM_SGLANG="0")
        self.assertEqual(backend.selected_name(), "none")
        self._env(HBIRD_GEMM_SGLANG="1", SGLANG_GEMM_BACKEND="torch")
        self.assertEqual(backend.selected_name(), "torch")
        self._env(SGLANG_GEMM_BACKEND=" HBird ")
        self.assertEqual(backend.selected_name(), "hbird")

    def test_quant_head_name_defaults_to_auto(self):
        self._env()
        self.assertEqual(backend.quant_head_name(), "auto")
        self._env(SGLANG_QUANT_HEAD_BACKEND=" LightOp ")
        self.assertEqual(backend.quant_head_name(), "lightop")

    def test_quant_head_provider_maps_auto_to_the_gemm_backend(self):
        self._env(SGLANG_GEMM_BACKEND="hbird")
        seen = []
        with mock.patch.object(backend, "_provider_for", seen.append):
            backend.quant_head_provider()
            self._env(SGLANG_QUANT_HEAD_BACKEND="lightop")
            backend.quant_head_provider()
        self.assertEqual(seen, ["hbird", "lightop"])

    def test_role_covers_target_and_draft_workers(self):
        self._env()
        self.assertEqual(backend.role(), "all")
        self.assertTrue(backend.selected(None))
        self._env(SGLANG_GEMM_ROLE="draft")
        self.assertTrue(backend.selected(
            types.SimpleNamespace(is_draft_worker=True)))
        self.assertFalse(backend.selected(
            types.SimpleNamespace(is_draft_worker=False)))
        self._env(HBIRD_GEMM_ROLE="target")
        self.assertTrue(backend.selected(
            types.SimpleNamespace(is_draft_worker=False)))
        self.assertFalse(backend.selected(
            types.SimpleNamespace(is_draft_worker=True)))
        self._env(SGLANG_GEMM_ROLE="none")
        self.assertFalse(backend.selected(None))
        self._env(SGLANG_GEMM_ROLE="")
        self.assertEqual(backend.role(), "all")

    def test_a_backend_is_loaded_once(self):
        calls = []

        def loader():
            calls.append(1)
            return object()

        self._env(SGLANG_GEMM_BACKEND="hbird")
        with mock.patch.object(backend, "_LOADERS", {"hbird": loader}):
            first = backend.provider()
            second = backend.provider()
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)

    def test_unknown_backend_is_none(self):
        self._env(SGLANG_GEMM_BACKEND="nope")
        with mock.patch.object(backend._LOG, "warning"):
            self.assertIsNone(backend.provider())


class LmHeadTest(_SeamBase):
    def test_no_backend_keeps_the_callers_path(self):
        self._env()
        self.assertIsNone(
            backend.lm_head_linear(object(), object(), None, False))

    def test_provider_result_is_returned(self):
        fake = types.SimpleNamespace(
            linear=lambda x, weight, handle: "logits",
            packed_handle=lambda module: None,
        )
        with mock.patch.object(backend, "provider", lambda: fake), \
                mock.patch.object(backend, "_lm_head_weight",
                                  lambda *args: object()):
            self.assertEqual(
                backend.lm_head_linear(object(), object(), None, False),
                "logits")

    def test_provider_failure_keeps_the_callers_path(self):
        def boom(*args):
            raise RuntimeError("no workspace")

        fake = types.SimpleNamespace(linear=boom,
                                     packed_handle=lambda module: None)
        with mock.patch.object(backend, "provider", lambda: fake), \
                mock.patch.object(backend, "_lm_head_weight",
                                  lambda *args: object()):
            self.assertIsNone(
                backend.lm_head_linear(object(), object(), None, False))

    def test_install_shim_without_a_backend_is_false(self):
        self._env()
        self.assertFalse(backend.install_shim())


if __name__ == "__main__":
    unittest.main()
