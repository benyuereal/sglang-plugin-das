# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Regression: stash failure eviction and the extra_buffer track replay.

Two properties the radix-on plain B+ commit depends on:

* ``stash`` must forget a layer whose row was not fully written (partial copy,
  rejected shape, or an old step's row). The commit replays every entry it can
  see with the *current* step's indices, so a stale entry silently commits the
  wrong token window.
* The upstream extra_buffer commit scatters the interval-crossing state from
  ``intermediate_ssm[snap_row, track_step]``, but verify only writes the full-run
  row. Every crossing step that is neither ``T - 1`` nor the truncated-slot
  replay's ``last_correct`` must be replayed once into that row.
"""

from __future__ import annotations

import os
import types
import unittest
from unittest import mock

import torch

from hcu_sglang.layers.attention.linear import gdn_commit, gdn_recurrent


def _lin_utils():
    try:
        import sglang.srt.layers.attention.linear.utils as module
    except ImportError:
        raise unittest.SkipTest("L0 lane has no sglang on the path")
    return module


N = 2
T = 4


def _backend(entries=None, with_pool=False):
    backend = types.SimpleNamespace()
    if with_pool:
        backend.req_to_token_pool = types.SimpleNamespace(size=N)
    st = gdn_commit.state(backend)
    st.stash = dict(entries or {})
    st.snap_idx = torch.arange(N, dtype=torch.int32)
    st.capture_T = T
    st.live_T = None
    return backend


def _stash_entry():
    """A replay-shaped entry: one flat row per (req, draft step)."""
    layer_cache = types.SimpleNamespace(
        intermediate_ssm=torch.zeros(4, 6, 2, 3),  # shape[1] = cache steps
        temporal=torch.zeros(4, 2, 3),
    )
    qkv = torch.zeros(N * T)
    entry = gdn_commit.StashEntry(
        rows=N * T,
        dims=gdn_commit.HeadDims(qh=1, qk=1, kh=1, kk=1, vh=1, vv=1),
        qkv=qkv, q=qkv, k=qkv, v=qkv,
        a=torch.zeros(N * T, 2), b=torch.zeros(N * T, 2),
        out=torch.zeros(1),
    )
    entry.layer_cache = layer_cache
    entry.A_log = torch.zeros(1)
    entry.dt_bias = torch.zeros(1)
    return entry


class _PpOff(unittest.TestCase):
    def setUp(self):
        self._linutils = _lin_utils()
        self._pp = getattr(self._linutils, "pp_spec_stable_rows_enabled", None)
        self._had_pp = hasattr(self._linutils, "pp_spec_stable_rows_enabled")
        self._linutils.pp_spec_stable_rows_enabled = lambda: False

    def tearDown(self):
        if self._had_pp:
            setattr(self._linutils, "pp_spec_stable_rows_enabled", self._pp)
        else:
            delattr(self._linutils, "pp_spec_stable_rows_enabled")


class StashCleanupTest(_PpOff):
    def _layer(self):
        return types.SimpleNamespace(layer_id=0, A_log=torch.zeros(1),
                                     dt_bias=torch.zeros(1))

    def test_partial_copy_drops_the_layer_entry(self):
        backend = _backend({0: _stash_entry()}, with_pool=True)
        q = torch.zeros(1, N * T, 1, 1)
        a = torch.zeros(N * T, 2)
        with mock.patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"}), \
                mock.patch.object(gdn_recurrent, "_gdn", object()), \
                mock.patch.object(gdn_commit, "_fused_qkv_view",
                                  side_effect=RuntimeError("copy broke")):
            ok = gdn_commit.stash(backend, self._layer(), q, q, q, a, a, T)
        self.assertFalse(ok)
        self.assertNotIn(0, gdn_commit.state(backend).stash)

    def test_empty_rows_drop_the_previous_step_entry(self):
        backend = _backend({0: _stash_entry()})
        empty = torch.zeros(1, 0, 1, 1)
        with mock.patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"}), \
                mock.patch.object(gdn_recurrent, "_gdn", object()):
            ok = gdn_commit.stash(backend, self._layer(), empty, empty, empty,
                                  torch.zeros(0, 2), torch.zeros(0, 2), T)
        self.assertFalse(ok)
        self.assertNotIn(0, gdn_commit.state(backend).stash)

    def test_success_keeps_the_entry(self):
        backend = _backend(with_pool=True)
        one = torch.ones(1, N * T, 1, 1)
        ab = torch.ones(N * T, 2)
        with mock.patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"}), \
                mock.patch.object(gdn_recurrent, "_gdn", object()):
            ok = gdn_commit.stash(backend, self._layer(), one, one, one,
                                  ab, ab, T)
        self.assertTrue(ok)
        self.assertIn(0, gdn_commit.state(backend).stash)


class TrackReplayTest(_PpOff):
    def _run(self, state_indices, last_correct, track, steps, backend=None):
        backend = backend or _backend({0: _stash_entry()})
        calls = []

        def fake_compensate(*args):
            calls.append({"h0": args[8].tolist(), "store": args[12].tolist()})

        with mock.patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"}), \
                mock.patch.object(gdn_recurrent, "_gdn", object()), \
                mock.patch.object(gdn_recurrent, "has_multi", lambda: False), \
                mock.patch.object(gdn_recurrent, "compensate", fake_compensate):
            ok = gdn_commit.compensate_from_stash(
                backend, state_indices, last_correct,
                torch.tensor(track, dtype=torch.int64),
                torch.tensor(steps, dtype=torch.int64),
            )
        return ok, calls

    def test_crossing_before_the_accepted_step_gets_its_own_row(self):
        # req0: full accept (last=3) but the interval crossed at step 2.
        # req1: truncated at step 1, crossing landed on the same step.
        ok, calls = self._run(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([T - 1, 1], dtype=torch.int64),
            track=[7, 8],
            steps=[2, 1],
        )
        self.assertTrue(ok)
        self.assertEqual(len(calls), 2)
        # req0 was fully accepted, so verify already wrote its accepted row.
        self.assertEqual(calls[0]["h0"], [-1, 2])
        self.assertEqual(calls[0]["store"], [-1, 1])
        # Only req0 needs the extra replay; req1 already got row 1 above.
        self.assertEqual(calls[1]["h0"], [1, -1])
        self.assertEqual(calls[1]["store"], [2, -1])

    def test_full_run_row_needs_no_replay(self):
        ok, calls = self._run(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([T - 1, T - 1], dtype=torch.int64),
            track=[7, 8],
            steps=[T - 1, -1],
        )
        self.assertTrue(ok)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["h0"], [-1, -1])

    def test_freed_track_slots_are_skipped(self):
        ok, calls = self._run(
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
            track=[-1, -1],
            steps=[0, 0],
        )
        self.assertTrue(ok)
        self.assertEqual(calls[1]["h0"], [-1, -1])

    def test_padding_state_slots_are_skipped(self):
        ok, calls = self._run(
            torch.tensor([-1, 2], dtype=torch.int64),
            torch.tensor([1, 1], dtype=torch.int64),
            track=[7, 8],
            steps=[0, 0],
        )
        self.assertTrue(ok)
        # The padding state slot (-1) is skipped; the live row keeps its replay.
        self.assertEqual(calls[1]["h0"], [-1, 2])

    def test_track_absent_keeps_the_single_plain_launch(self):
        backend = _backend({0: _stash_entry()})
        calls = []

        def fake_compensate(*args):
            calls.append(args[8].tolist())

        with mock.patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"}), \
                mock.patch.object(gdn_recurrent, "_gdn", object()), \
                mock.patch.object(gdn_recurrent, "has_multi", lambda: False), \
                mock.patch.object(gdn_recurrent, "compensate", fake_compensate):
            ok = gdn_commit.compensate_from_stash(
                backend,
                torch.tensor([1, 2], dtype=torch.int64),
                torch.tensor([1, -1], dtype=torch.int64),
            )
        self.assertTrue(ok)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
