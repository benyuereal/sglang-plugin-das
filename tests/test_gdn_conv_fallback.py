# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Which causal-conv1d update runs when lightop or the HCU twin is unavailable.

The dispatch seam (``gdn_conv``) imports neither sglang nor lightop, so these
control-flow tests run without a device. ``_convolve_prefill`` lives in the
vendored module and is skipped where sglang's import chain cannot load.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

from hcu_sglang.layers.attention.linear import gdn_conv, gdn_recurrent


class ConvUpdateFallbackTests(unittest.TestCase):
    def setUp(self):
        gdn_conv._state["logged"] = False
        gdn_conv._state["last"] = ""
        self.calls = []
        self.logs = []
        self.x = torch.zeros(1, 8)
        self.kwargs = {"conv_state_indices": torch.zeros(1, dtype=torch.int32)}

    def _call(self, *, use_hcu=False, enabled=False, hip=None,
              verify=False):
        """Run update() with lightop and both upstream updates stubbed."""

        def hcu(*args, **kwargs):
            self.calls.append("hcu")
            return "hcu"

        def triton(*args, **kwargs):
            self.calls.append("triton")
            return "triton"

        def lightop(*args, **kwargs):
            self.calls.append("lightop")
            return hip() if callable(hip) else hip

        gdn_conv.configure(
            triton=triton,
            hcu=hcu if use_hcu else None,
            log=self.logs.append,
        )
        patches = [
            mock.patch.object(gdn_recurrent, "overlay_enabled", lambda: enabled),
            mock.patch.object(gdn_recurrent, "conv_update", lightop),
            mock.patch.object(gdn_recurrent, "conv_last_reason",
                              lambda: "outside the kernel"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        run = gdn_conv.update_verify if verify else gdn_conv.update
        return run(self.x, None, None, None, "silu", **self.kwargs)

    def test_lightop_off_runs_triton(self):
        self.assertEqual(self._call(), "triton")
        self.assertEqual(self.calls, ["triton"])

    def test_lightop_off_keeps_the_hcu_twin(self):
        # Regression: the fallback must not drop main's HCU update when
        # SGLANG_USE_CAUSAL_CONV1D=1 and lightop is absent.
        self.assertEqual(self._call(use_hcu=True), "hcu")
        self.assertEqual(self.calls, ["hcu"])

    def test_lightop_result_is_returned_untouched(self):
        self.assertEqual(self._call(enabled=True, hip=lambda: "hip"), "hip")
        self.assertEqual(self.calls, ["lightop"])

    def test_lightop_decline_falls_back_to_the_hcu_twin(self):
        self.assertEqual(self._call(enabled=True, use_hcu=True, hip=None), "hcu")
        self.assertEqual(self.calls, ["lightop", "hcu"])
        self.assertEqual(len(self.logs), 1)
        self.assertIn("outside the kernel", self.logs[0])

    def test_decline_is_logged_once(self):
        for _ in range(3):
            self._call(enabled=True, use_hcu=True, hip=None)
        self.assertEqual(len(self.logs), 1)

    def test_lightop_failure_falls_back(self):
        def boom():
            raise RuntimeError("kernel load failed")

        self.assertEqual(self._call(enabled=True, use_hcu=True, hip=boom), "hcu")
        self.assertEqual(self.calls, ["lightop", "hcu"])
        self.assertIn("kernel load failed", self.logs[0])

    def test_a_decline_in_the_verify_path_uses_triton(self):
        self.assertEqual(
            self._call(enabled=True, use_hcu=True, hip=None, verify=True),
            "triton")
        self.assertEqual(self.calls, ["lightop", "triton"])
        self.assertIn("fell back to Triton", self.logs[0])

    def test_verify_without_the_overlay_uses_triton(self):
        self.assertEqual(self._call(enabled=False, verify=True), "triton")
        self.assertEqual(self.calls, ["triton"])

    def test_verify_without_any_updater_raises(self):
        gdn_conv.configure(triton=None, hcu=None, log=self.logs.append)
        with mock.patch.object(gdn_recurrent, "overlay_enabled", lambda: False):
            with self.assertRaisesRegex(RuntimeError, "no conv updater"):
                gdn_conv.update_verify(self.x, None, None)

    def test_update_before_configure_is_a_programming_error(self):
        gdn_conv.configure(triton=None, hcu=None,
                           log=self.logs.append)
        with self.assertRaises(RuntimeError):
            gdn_conv.update(self.x, None, None)


def _import_gdn_source():
    try:
        from hcu_sglang.layers.attention.linear import gdn_source
    except Exception:  # noqa: BLE001 - sglang's import chain needs a device
        return None
    return gdn_source


gdn_source = _import_gdn_source()


@unittest.skipIf(gdn_source is None, "sglang's import chain needs a device")
class ConvolvePrefillTests(unittest.TestCase):
    def setUp(self):
        self.layer = types.SimpleNamespace(
            conv_weights="w", bias="b", activation="silu"
        )
        self.conv_states = torch.zeros(4, 8)
        self.mixed_qkv = torch.zeros(4, 8)
        self.cache_indices = torch.zeros(4, dtype=torch.int32)
        self.forward_batch = types.SimpleNamespace(
            extend_prefix_lens=torch.zeros(1),
            extend_seq_lens_cpu=[4],
        )
        self.backend = types.SimpleNamespace(
            forward_metadata=types.SimpleNamespace(
                conv_block_table=None,
                query_start_loc=torch.tensor([0, 4]),
            )
        )

    def _wrapped(self, *args):
        self.wrapped_called = True
        return "upstream"

    def test_delegates_when_the_hcu_conv_is_off(self):
        with mock.patch.object(gdn_source, "_use_hcu_causal_conv1d", False):
            out = gdn_source._convolve_prefill(
                self._wrapped, self.backend, self.layer, self.forward_batch,
                self.mixed_qkv, self.conv_states, self.cache_indices,
            )
        self.assertEqual(out, "upstream")
        self.assertTrue(self.wrapped_called)

    def test_delegates_for_a_block_table(self):
        self.backend.forward_metadata.conv_block_table = object()
        with mock.patch.object(gdn_source, "_use_hcu_causal_conv1d", True), \
                mock.patch.object(gdn_source, "causal_conv1d_fn_hcu",
                                  lambda *a, **k: self.fail("kernel ran"),
                                  create=True):
            out = gdn_source._convolve_prefill(
                self._wrapped, self.backend, self.layer, self.forward_batch,
                self.mixed_qkv, self.conv_states, self.cache_indices,
            )
        self.assertEqual(out, "upstream")

    def test_hcu_kernel_gets_initial_states(self):
        seen = {}

        def kernel(mixed_qkv, weight, bias, **kwargs):
            seen.update(kwargs)
            return torch.zeros(8, 4)

        with mock.patch.object(gdn_source, "_use_hcu_causal_conv1d", True), \
                mock.patch.object(gdn_source, "causal_conv1d_fn_hcu", kernel,
                                  create=True):
            out = gdn_source._convolve_prefill(
                self._wrapped, self.backend, self.layer, self.forward_batch,
                self.mixed_qkv, self.conv_states, self.cache_indices,
            )
        self.assertEqual(tuple(out.shape), (4, 8))
        self.assertIs(seen["initial_states"], self.conv_states)
        self.assertNotIn("conv_states", seen)
        self.assertNotIn("block_table", seen)


if __name__ == "__main__":
    unittest.main()
