# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression: the non-handoff (plain) B+ compensate path must replay its rows
when SGLANG_GDN_HIP_BPLUS_SKIP_FULL=0.

Before the fix, ``neg`` was ``torch.full_like(idx32, -1) if skip_full else None``
and the plain path built ``h0_plain = torch.where(trunc, idx32, neg)`` with no
guard. With SKIP_FULL=0 that is ``where(..., None)``, which raises TypeError; the
raise was swallowed by the per-layer ``except Exception`` so every layer's row
was dropped, no state was replayed, and the final answer came out wrong. ``neg``
is now built unconditionally, so the plain path always has a valid int32 index.
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

from hcu_sglang.layers.attention.linear import gdn_bplus, gdn_hip_bplus



def _lin_utils():
    """L0 runs without sglang on the path; the GPU lane imports it."""
    try:
        import sglang.srt.layers.attention.linear.utils as module
    except ImportError:
        raise unittest.SkipTest("L0 lane has no sglang on the path")
    return module


# batch=2, draft width (capture_T)=3, one GDN layer, one stash entry.
N = 2
T = 3


def _stash_entry():
    layer_cache = types.SimpleNamespace(
        intermediate_ssm=torch.zeros(4, 5, 2, 3),  # shape[1] = stride
        temporal=torch.zeros(4),
    )
    qkv = torch.zeros(N * T)
    return {
        "layer_cache": layer_cache,
        "q": qkv, "k": qkv, "v": qkv, "a": qkv, "b": qkv,
        "A_log": torch.zeros(1), "dt_bias": torch.zeros(1), "out": torch.zeros(1),
        "qh": 1, "qk": 1, "kh": 1, "kk": 1, "vh": 1, "vv": 1,
    }


def _backend():
    backend = types.SimpleNamespace()
    st = gdn_bplus.state(backend)
    st.stash = {0: _stash_entry()}
    st.snap_idx = torch.zeros(N, dtype=torch.int32)
    st.capture_T = T
    st.live_T = None
    st.prev = None
    st.step_cache = None
    return backend


class PlainCompensateTest(unittest.TestCase):
    def setUp(self):
        self._handoff_ok = gdn_bplus._HANDOFF_OK
        self._linutils = _lin_utils()
        self._pp = getattr(self._linutils, "pp_spec_stable_rows_enabled", None)
        self._had_pp = hasattr(self._linutils, "pp_spec_stable_rows_enabled")

    def tearDown(self):
        gdn_bplus._HANDOFF_OK = self._handoff_ok
        if self._had_pp:
            setattr(self._linutils, "pp_spec_stable_rows_enabled", self._pp)
        else:
            delattr(self._linutils, "pp_spec_stable_rows_enabled")

    def _run_plain(self, env):
        # Force the non-handoff (plain) branch: B+ on, handoff off.
        env.update(
            {
                "SGLANG_GDN_HIP_BPLUS": "1",
                "SGLANG_GDN_HIP_HANDOFF": "0",
                "SGLANG_GDN_HIP_BPLUS_MULTI": "0",  # skip the multi launcher
            }
        )
        gdn_bplus._HANDOFF_OK = None  # re-evaluate with the env above
        self._linutils.pp_spec_stable_rows_enabled = lambda: False
        backend = _backend()
        state_indices = torch.tensor([1, 2], dtype=torch.int64)
        last_correct = torch.tensor([1, -1], dtype=torch.int64)
        recorded = {}

        def fake_compensate(*args, **kwargs):
            recorded["h0"] = args[8]
            recorded["called"] = True

        with mock.patch.dict("os.environ", env), mock.patch.object(
            gdn_hip_bplus, "compensate", fake_compensate
        ):
            ok = gdn_bplus.compensate_from_stash(backend, state_indices, last_correct)
        self.assertTrue(ok)
        self.assertTrue(recorded.get("called"))
        return recorded["h0"]

    def test_skip_full_off_plain_path_replays_every_slot(self):
        h0 = self._run_plain({"SGLANG_GDN_HIP_BPLUS_SKIP_FULL": "0"})
        self.assertEqual(h0.dtype, torch.int32)
        # trunc = [True, False]: with the switch off the full-accept slot keeps
        # its own index instead of the -1 skip marker.
        self.assertEqual(h0.tolist(), [1, 2])

    def test_skip_full_on_plain_path_marks_the_full_accept(self):
        h0 = self._run_plain({"SGLANG_GDN_HIP_BPLUS_SKIP_FULL": "1"})
        assert h0.tolist() == [1, -1]


if __name__ == "__main__":
    unittest.main()
