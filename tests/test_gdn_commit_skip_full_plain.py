# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression: the plain B+ compensate path replays only truncated slots.

The plain path is the only B+ commit mode: verify writes the full snapshot at
the batch-position rows, and ``compensate_from_stash`` replays the slots the
accept truncated into the slot-indexed temporal state. A fully accepted slot
must keep the ``-1`` skip marker, because verify already wrote its snapshot.

``neg`` is built unconditionally: ``torch.where(trunc, idx32, None)`` raises
TypeError, and the per-layer ``except Exception`` would swallow it, drop every
layer's row, and produce a wrong answer.
"""

from __future__ import annotations

import os
import types
import unittest
from unittest import mock

import torch

from hcu_sglang.layers.attention.linear import gdn_commit, gdn_recurrent



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
        "q": qkv, "k": qkv, "v": qkv,
        # a/b are per-token 2-D (rows, gate_dim), like the real stash entries.
        "a": torch.zeros(N * T, 2), "b": torch.zeros(N * T, 2),
        "A_log": torch.zeros(1), "dt_bias": torch.zeros(1), "out": torch.zeros(1),
        "dims": gdn_commit.HeadDims(qh=1, qk=1, kh=1, kk=1, vh=1, vv=1),
    }


def _backend():
    backend = types.SimpleNamespace()
    st = gdn_commit.state(backend)
    st.stash = {0: _stash_entry()}
    st.snap_idx = torch.zeros(N, dtype=torch.int32)
    st.capture_T = T
    st.live_T = None
    return backend


class PlainCompensateTest(unittest.TestCase):
    def setUp(self):
        self._linutils = _lin_utils()
        self._pp = getattr(self._linutils, "pp_spec_stable_rows_enabled", None)
        self._had_pp = hasattr(self._linutils, "pp_spec_stable_rows_enabled")
        self._pp_on = False
        self._linutils.pp_spec_stable_rows_enabled = lambda: self._pp_on

    def tearDown(self):
        if self._had_pp:
            setattr(self._linutils, "pp_spec_stable_rows_enabled", self._pp)
        else:
            delattr(self._linutils, "pp_spec_stable_rows_enabled")

    def _run_plain(self, state_indices, last_correct, backend=None):
        backend = backend or _backend()
        recorded = {}

        def fake_compensate(*args, **kwargs):
            recorded["ssm"] = args[7]
            recorded["h0"] = args[8]
            recorded["snap"] = args[10]
            recorded["called"] = True

        with mock.patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"}), \
                mock.patch.object(gdn_recurrent, "has_multi", lambda: False), \
                mock.patch.object(gdn_recurrent, "compensate", fake_compensate):
            ok = gdn_commit.compensate_from_stash(
                backend, state_indices, last_correct
            )
        return ok, recorded, backend

    def test_plain_path_replays_only_the_truncated_slots(self):
        ok, recorded, _ = self._run_plain(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1, -1], dtype=torch.int64),
        )
        self.assertTrue(ok)
        self.assertTrue(recorded.get("called"))
        self.assertEqual(recorded["h0"].dtype, torch.int32)
        # trunc = [True, False]: only the accepted-prefix slot is replayed.
        self.assertEqual(recorded["h0"].tolist(), [1, -1])

    def test_plain_path_replays_a_zero_step_accept(self):
        ok, recorded, _ = self._run_plain(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([0, 1], dtype=torch.int64),
        )
        self.assertTrue(ok)
        self.assertEqual(recorded["h0"].tolist(), [1, 2])

    def test_plain_path_marks_a_full_accept(self):
        ok, recorded, _ = self._run_plain(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([T - 1, T - 1], dtype=torch.int64),
        )
        self.assertTrue(ok)
        # Verify already wrote these snapshot rows; nothing to replay.
        self.assertEqual(recorded["h0"].tolist(), [-1, -1])

    def test_plain_path_replays_into_the_temporal_state(self):
        ok, recorded, backend = self._run_plain(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1, -1], dtype=torch.int64),
        )
        self.assertTrue(ok)
        temporal = gdn_commit.state(backend).stash[0]["layer_cache"].temporal
        self.assertIs(recorded["ssm"], temporal)

    def test_plain_path_rebuilds_the_batch_position_rows(self):
        ok, recorded, _ = self._run_plain(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1, -1], dtype=torch.int64),
        )
        self.assertTrue(ok)
        self.assertEqual(recorded["snap"].tolist(), [0, 1])

    def test_plain_path_uses_the_verify_rows_for_pp_stable_rows(self):
        self._pp_on = True
        backend = _backend()
        gdn_commit.state(backend).snap_idx = torch.tensor(
            [5, 6], dtype=torch.int32
        )
        ok, recorded, _ = self._run_plain(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1, -1], dtype=torch.int64),
            backend,
        )
        self.assertTrue(ok)
        self.assertEqual(recorded["snap"].tolist(), [5, 6])

    def test_plain_path_declines_when_pp_rows_do_not_match(self):
        self._pp_on = True
        backend = _backend()
        gdn_commit.state(backend).snap_idx = torch.tensor(
            [5], dtype=torch.int32
        )
        ok, recorded, _ = self._run_plain(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1, -1], dtype=torch.int64),
            backend,
        )
        self.assertFalse(ok)
        self.assertNotIn("called", recorded)


if __name__ == "__main__":
    unittest.main()
