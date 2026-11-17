# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Bitwise parity between the shared verify-step values and the per-layer
arithmetic they replaced (verify wrapper + h0_rows + ensure_prev)."""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

import torch

from hcu_sglang.layers.attention.linear import gdn_commit, gdn_recurrent

CACHE_STEPS = 12
OTHER_STEPS = 16
POOL_ROWS = 24
BATCH = 16
DRAFT = 5


def _params():
    return types.SimpleNamespace(temporal=torch.zeros(POOL_ROWS, 2, 4))


def _old_per_layer(prev_buf, slots, cache_steps, draft):
    """Pre-patch arithmetic, run once per layer."""
    s64 = slots.to(torch.int64).clamp_min(0)
    prev = prev_buf.index_select(0, s64).to(torch.int32)
    store = torch.full((int(slots.numel()),), draft - 1, dtype=torch.int32)
    safe = slots.to(torch.int64).clamp(0, int(prev_buf.numel()) - 1)
    prev_v = prev_buf.index_select(0, safe).to(torch.int32)
    store_row = torch.remainder(prev_v + 1, int(cache_steps)).to(torch.int32)
    h0 = (
        s64 * int(cache_steps) + prev.to(torch.int64).clamp_min(0)
    ).to(torch.int32)
    return s64, prev, store, store_row, h0


class SharedStepTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        gdn_commit._HANDOFF_OK = None
        self.backend = types.SimpleNamespace()
        self.params = _params()
        self.prev_buf = torch.randint(
            -1, CACHE_STEPS, (POOL_ROWS + 3,), dtype=torch.int32
        )
        gdn_commit.state(self.backend).prev = self.prev_buf
        self.slots = torch.randint(0, POOL_ROWS, (BATCH,), dtype=torch.int32)
        self.slots[3] = -1
        self.slots[9] = -1

    def tearDown(self):
        gdn_commit._HANDOFF_OK = None

    def assertBitEqual(self, got, want):
        self.assertEqual(got.dtype, want.dtype)
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        self.assertTrue(torch.equal(got, want))

    def shared(self, slots=None, cache_steps=CACHE_STEPS, draft=DRAFT):
        return gdn_commit.shared_step_values(
            self.backend,
            self.slots if slots is None else slots,
            self.params,
            cache_steps,
            draft,
        )

    def test_every_layer_matches_the_per_layer_arithmetic(self):
        sh = None
        for lid in range(6):
            gdn_commit.begin_verify_step(self.backend, lid)
            sh = self.shared()
            s64, prev, store, store_row, h0 = _old_per_layer(
                self.prev_buf, self.slots, CACHE_STEPS, DRAFT
            )
            self.assertBitEqual(sh["s64"], s64)
            self.assertBitEqual(sh["prev"], prev)
            self.assertBitEqual(sh["store"], store)
            self.assertBitEqual(sh["rows"][CACHE_STEPS]["store_row"], store_row)
            self.assertBitEqual(sh["rows"][CACHE_STEPS]["h0"], h0)

    def test_shared_tensors_are_reused_within_a_forward(self):
        gdn_commit.begin_verify_step(self.backend, 0)
        first = self.shared()
        gdn_commit.begin_verify_step(self.backend, 1)
        second = self.shared()
        self.assertIs(first, second)
        self.assertIs(first["s64"], second["s64"])
        self.assertIs(first["store"], second["store"])

    def test_a_new_forward_recomputes_from_the_committed_buffer(self):
        gdn_commit.begin_verify_step(self.backend, 0)
        self.shared()
        self.prev_buf.random_(-1, CACHE_STEPS)
        gdn_commit.begin_verify_step(self.backend, 0)
        sh = self.shared()
        _, prev, _, store_row, h0 = _old_per_layer(
            self.prev_buf, self.slots, CACHE_STEPS, DRAFT
        )
        self.assertBitEqual(sh["prev"], prev)
        self.assertBitEqual(sh["rows"][CACHE_STEPS]["store_row"], store_row)
        self.assertBitEqual(sh["rows"][CACHE_STEPS]["h0"], h0)

    def test_each_stride_keeps_its_own_rows(self):
        gdn_commit.begin_verify_step(self.backend, 0)
        sh = self.shared()
        gdn_commit.begin_verify_step(self.backend, 1)
        self.shared(cache_steps=OTHER_STEPS)
        for steps in (CACHE_STEPS, OTHER_STEPS):
            _, _, _, store_row, h0 = _old_per_layer(
                self.prev_buf, self.slots, steps, DRAFT
            )
            self.assertBitEqual(sh["rows"][steps]["store_row"], store_row)
            self.assertBitEqual(sh["rows"][steps]["h0"], h0)

    def test_draft_width_changes_the_store_row_buffer(self):
        # Adaptive speculation varies the width between forwards, never inside
        # one: the new forward is what re-fills the store row.
        gdn_commit.begin_verify_step(self.backend, 0)
        self.shared(draft=3)
        gdn_commit.begin_verify_step(self.backend, 0)
        sh = self.shared(draft=DRAFT)
        store = torch.full((BATCH,), DRAFT - 1, dtype=torch.int32)
        self.assertBitEqual(sh["store"], store)

    def test_step_h0_only_matches_the_batch_it_was_computed_for(self):
        gdn_commit.begin_verify_step(self.backend, 0)
        self.shared()
        self.assertIsNotNone(gdn_commit.step_h0(self.backend, self.slots, CACHE_STEPS))
        shorter = self.slots[:8]
        self.assertIsNone(gdn_commit.step_h0(self.backend, shorter, CACHE_STEPS))
        other = self.slots.clone()
        self.assertIsNone(gdn_commit.step_h0(self.backend, other, CACHE_STEPS))
        self.assertIsNone(gdn_commit.step_h0(self.backend, self.slots, OTHER_STEPS))

    def test_h0_rows_fallback_matches_the_cached_rows(self):
        gdn_commit.begin_verify_step(self.backend, 0)
        sh = self.shared()
        s64, prev, _, _, h0 = _old_per_layer(
            self.prev_buf, self.slots, CACHE_STEPS, DRAFT
        )
        self.assertBitEqual(gdn_commit.h0_rows(self.slots, prev, CACHE_STEPS), h0)
        self.assertBitEqual(sh["rows"][CACHE_STEPS]["h0"], h0)

    def test_verify_forwards_the_precomputed_rows_untouched(self):
        captured = {}

        def fake_call(*args, **kwargs):
            captured["store"] = kwargs["store_step"]
            captured["store_row"] = kwargs["store_row"]

        class FakeLib:
            def __getattr__(self, name):
                return fake_call

        q = torch.zeros(1, BATCH * DRAFT, 2, 8)
        v = torch.zeros(1, BATCH * DRAFT, 4, 6)
        snap_idx = torch.zeros(BATCH, dtype=torch.int32)
        store = torch.full((BATCH,), DRAFT - 1, dtype=torch.int32)
        store_row = torch.zeros(BATCH, dtype=torch.int32)
        with patch.object(gdn_recurrent, "lib", FakeLib):
            gdn_recurrent.verify(
                q, q, v, q, q, q, q, None, v, snap_idx, v, snap_idx,
                CACHE_STEPS, DRAFT, v, prev_step=self.prev_buf,
                store=store, store_row=store_row,
            )
        self.assertIs(captured["store"], store)
        self.assertIs(captured["store_row"], store_row)

    def test_verify_fallback_recomputes_the_rows(self):
        captured = {}

        def fake_call(*args, **kwargs):
            captured["store"] = kwargs["store_step"]
            captured["store_row"] = kwargs["store_row"]

        class FakeLib:
            def __getattr__(self, name):
                return fake_call

        q = torch.zeros(1, BATCH * DRAFT, 2, 8)
        v = torch.zeros(1, BATCH * DRAFT, 4, 6)
        snap_idx = self.slots.to(torch.int32)
        with patch.object(gdn_recurrent, "lib", FakeLib):
            gdn_recurrent.verify(
                q, q, v, q, q, q, q, None, v, snap_idx, v, snap_idx,
                CACHE_STEPS, DRAFT, v, prev_step=self.prev_buf,
            )
        _, _, store, store_row, _ = _old_per_layer(
            self.prev_buf, snap_idx, CACHE_STEPS, DRAFT
        )
        self.assertBitEqual(captured["store"], store)
        self.assertBitEqual(captured["store_row"], store_row)

    def test_commit_prev_rotates_the_same_rows_as_the_reference(self):
        idx = torch.tensor(
            [0, 5, -1, 7, 7, -1, 2, 3, 4, 5, -1, 9, 10, 11, 12, 13],
            dtype=torch.int64,
        )
        last = torch.tensor(
            [4, 0, 1, 2, 3, 4, 0, -1, 3, 2, 1, 0, 4, 3, 2, 1],
            dtype=torch.int64,
        )
        T = 5
        want = self.prev_buf.clone()
        prev = want.index_select(0, idx.clamp_min(0)).to(torch.int64)
        store_row = torch.remainder(prev + 1, CACHE_STEPS)
        keep = torch.where(last < T - 1, last, store_row).to(torch.int32)
        discard = torch.full_like(idx, int(want.numel()) - 1)
        put_idx = torch.where(idx >= 0, idx, discard)
        want.index_copy_(0, put_idx, keep)

        gdn_commit.state(self.backend).prev = self.prev_buf.clone()
        env = {
            "SGLANG_GDN_HIP_BPLUS": "1",
            "SGLANG_GDN_HIP_HANDOFF": "1",
            "SGLANG_GDN_HIP_HANDOFF_SKIP_TRACK": "1",
        }
        with patch.dict(os.environ, env):
            ok = gdn_commit.commit_prev(self.backend, idx, last, T, CACHE_STEPS)
        self.assertTrue(ok)
        self.assertBitEqual(gdn_commit.state(self.backend).prev, want)


if __name__ == "__main__":
    unittest.main()
