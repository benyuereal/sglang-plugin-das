# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Handoff row space: request scratch rows, not persistent mamba slots.

The speculative ``intermediate_ssm`` has one row per live request
(``max_num_reqs + 1``); mamba slots under a radix cache span every checkpoint
slot. These tests pin the request-row mapping, the seed/commit arithmetic and
the out-of-range refusals that keep the kernels in bounds.
"""

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
SCRATCH_ROWS = POOL_ROWS + 1
BATCH = 16
DRAFT = 5
STATE = (2, 4)


def _params():
    return types.SimpleNamespace(
        temporal=torch.zeros(POOL_ROWS, *STATE),
        intermediate_ssm=torch.zeros(SCRATCH_ROWS, DRAFT, *STATE),
    )


def _reference_rows(prev_buf, rows, cache_steps, draft):
    """The per-layer arithmetic, on request rows."""
    s64 = rows.to(torch.int64)
    gather = s64.clamp_min(0)
    prev = prev_buf.index_select(0, gather).to(torch.int32)
    valid = (s64 >= 0) & (prev >= 0)
    store = torch.full((int(rows.numel()),), draft - 1, dtype=torch.int32)
    store_row = torch.remainder(prev.to(torch.int64) + 1, cache_steps)
    store_row = torch.where(valid, store_row, torch.full_like(store_row, -1))
    h0 = s64 * int(cache_steps) + prev.to(torch.int64).clamp_min(0)
    h0 = torch.where(valid, h0, torch.full_like(h0, -1))
    return s64, prev, store, store_row.to(torch.int32), h0.to(torch.int32)


class SharedStepTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.backend = types.SimpleNamespace()
        self.params = _params()
        # Slot id space (persistent mamba slots); unrelated to scratch rows.
        self.prev_buf = torch.randint(
            -1, CACHE_STEPS, (SCRATCH_ROWS,), dtype=torch.int32
        )
        gdn_commit.state(self.backend).prev = self.prev_buf
        self.rows = torch.randint(0, SCRATCH_ROWS, (BATCH,), dtype=torch.int32)
        self.rows[3] = -1
        self.rows[9] = -1

    def assertBitEqual(self, got, want):
        self.assertEqual(got.dtype, want.dtype)
        self.assertEqual(tuple(got.shape), tuple(want.shape))
        self.assertTrue(torch.equal(got, want))

    def shared(self, rows=None, cache_steps=CACHE_STEPS, draft=DRAFT):
        return gdn_commit.shared_step_values(
            self.backend,
            self.rows if rows is None else rows,
            self.params,
            cache_steps,
            draft,
        )

    def test_every_layer_matches_the_per_layer_arithmetic(self):
        sh = None
        for lid in range(6):
            gdn_commit.begin_verify_step(self.backend, lid)
            sh = self.shared()
            s64, prev, store, store_row, h0 = _reference_rows(
                self.prev_buf, self.rows, CACHE_STEPS, DRAFT
            )
            self.assertBitEqual(sh["s64"], s64)
            self.assertBitEqual(sh["prev"], prev)
            self.assertBitEqual(sh["store"], store)
            self.assertBitEqual(sh["rows"][CACHE_STEPS]["store_row"], store_row)
            self.assertBitEqual(sh["rows"][CACHE_STEPS]["h0"], h0)

    def test_request_rows_are_the_snapshot_rows(self):
        rows = torch.arange(BATCH, dtype=torch.int32)
        self.prev_buf.zero_()
        gdn_commit.begin_verify_step(self.backend, 0)
        sh = self.shared(rows=rows)
        _, _, _, _, h0 = _reference_rows(self.prev_buf, rows, CACHE_STEPS, DRAFT)
        # Row i owns flat rows [i*D, (i+1)*D); nothing multiplies the slot id.
        self.assertBitEqual(sh["rows"][CACHE_STEPS]["h0"], h0)
        self.assertBitEqual(
            sh["rows"][CACHE_STEPS]["h0"],
            (rows.to(torch.int64) * CACHE_STEPS).to(torch.int32),
        )

    def test_unseeded_rows_carry_no_h0_and_no_store_row(self):
        self.prev_buf.fill_(-1)
        gdn_commit.begin_verify_step(self.backend, 0)
        sh = self.shared()
        self.assertTrue(bool((sh["rows"][CACHE_STEPS]["h0"] == -1).all()))
        self.assertTrue(bool((sh["rows"][CACHE_STEPS]["store_row"] == -1).all()))

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
        _, prev, _, store_row, h0 = _reference_rows(
            self.prev_buf, self.rows, CACHE_STEPS, DRAFT
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
            _, _, _, store_row, h0 = _reference_rows(
                self.prev_buf, self.rows, steps, DRAFT
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
        self.assertIsNotNone(
            gdn_commit.step_h0(self.backend, self.rows, CACHE_STEPS))
        shorter = self.rows[:8]
        self.assertIsNone(gdn_commit.step_h0(self.backend, shorter, CACHE_STEPS))
        other = self.rows.clone()
        self.assertIsNone(gdn_commit.step_h0(self.backend, other, CACHE_STEPS))
        self.assertIsNone(
            gdn_commit.step_h0(self.backend, self.rows, OTHER_STEPS))

    def test_h0_rows_fallback_matches_the_cached_rows(self):
        gdn_commit.begin_verify_step(self.backend, 0)
        sh = self.shared()
        _, prev, _, _, h0 = _reference_rows(
            self.prev_buf, self.rows, CACHE_STEPS, DRAFT
        )
        rows = gdn_commit.h0_rows(self.rows, prev, CACHE_STEPS)
        rows = torch.where(self.rows >= 0, rows,
                           torch.full_like(rows, -1))
        self.assertBitEqual(rows, h0)
        self.assertBitEqual(sh["rows"][CACHE_STEPS]["h0"], h0)

    def test_verify_forwards_the_precomputed_rows_untouched(self):
        captured = {}

        def fake_call(*args, **kwargs):
            captured["store"] = kwargs["store"]
            captured["store_row"] = kwargs["store_row"]

        class FakeLib:
            def __getattr__(self, name):
                return fake_call

        q = torch.zeros(1, BATCH * DRAFT, 2, 8)
        v = torch.zeros(1, BATCH * DRAFT, 4, 6)
        snap_idx = torch.zeros(BATCH, dtype=torch.int32)
        store = torch.full((BATCH,), DRAFT - 1, dtype=torch.int32)
        store_row = torch.zeros(BATCH, dtype=torch.int32)
        with patch.object(gdn_recurrent, "_gdn", FakeLib()):
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
            captured["store"] = kwargs["store"]
            captured["store_row"] = kwargs["store_row"]

        class FakeLib:
            def __getattr__(self, name):
                return fake_call

        q = torch.zeros(1, BATCH * DRAFT, 2, 8)
        v = torch.zeros(1, BATCH * DRAFT, 4, 6)
        snap_idx = self.rows
        # The library-side default row arithmetic has no unseeded-row mask.
        self.prev_buf.clamp_min_(0)
        with patch.object(gdn_recurrent, "_gdn", FakeLib()):
            gdn_recurrent.verify(
                q, q, v, q, q, q, q, None, v, snap_idx, v, snap_idx,
                CACHE_STEPS, DRAFT, v, prev_step=self.prev_buf,
            )
        # The library default clamps the row and ignores the seed mask.
        safe = snap_idx.to(torch.int64).clamp(0, int(self.prev_buf.numel()) - 1)
        prev = self.prev_buf.index_select(0, safe).to(torch.int32)
        store = torch.full((BATCH,), DRAFT - 1, dtype=torch.int32)
        store_row = torch.remainder(prev + 1, CACHE_STEPS).to(torch.int32)
        self.assertBitEqual(captured["store"], store)
        self.assertBitEqual(captured["store_row"], store_row)


class RequestRowsTest(unittest.TestCase):
    def test_rows_above_the_scratch_are_masked(self):
        rows = torch.tensor([0, 1, SCRATCH_ROWS - 1, SCRATCH_ROWS, -1],
                            dtype=torch.int64)
        got = gdn_commit.request_rows(rows, SCRATCH_ROWS, 5)
        want = torch.tensor([0, 1, SCRATCH_ROWS - 1, -1, -1],
                            dtype=torch.int32)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(got.dtype, torch.int32)

    def test_a_short_row_tensor_is_refused(self):
        rows = torch.tensor([1, 2], dtype=torch.int32)
        self.assertIsNone(gdn_commit.request_rows(rows, SCRATCH_ROWS, 3))
        self.assertIsNone(gdn_commit.request_rows(None, SCRATCH_ROWS, 3))

    def test_batch_rows_reads_the_static_buffer(self):
        bufs = [torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 7], dtype=torch.int32)]
        backend = types.SimpleNamespace(handoff_rows_list=bufs)
        batch = types.SimpleNamespace(
            batch_size=2,
            req_pool_indices=torch.tensor([7, 5], dtype=torch.int64))
        self.assertTrue(torch.equal(
            gdn_commit.batch_rows(backend, batch), bufs[1][:2]))


class SeedPrevTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # One scratch row per request-pool row, plus the reserved row 0.
        self.backend = types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(
                size=POOL_ROWS, _alloc_size=POOL_ROWS + 1))
        env = patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.params = _params()
        st = gdn_commit.state(self.backend)
        st.prev = None
        st.disabled = False
        st.geometry = None
        st.seed_scratch = None

    def _seed(self, slots, rows):
        return gdn_commit.seed_prev(
            self.backend, torch.tensor(slots, dtype=torch.int32),
            self.params, DRAFT, torch.tensor(rows, dtype=torch.int32))

    def test_seed_writes_the_request_row_not_the_slot_row(self):
        temporal = self.params.temporal
        temporal.copy_(torch.arange(POOL_ROWS).view(-1, 1, 1).expand(
            POOL_ROWS, *STATE).to(temporal.dtype))
        slots, rows = [5, 7], [2, 3]
        self.assertTrue(self._seed(slots, rows))
        flat = self.params.intermediate_ssm.reshape(
            -1, *self.params.intermediate_ssm.shape[2:])
        # Step 0 of the request row holds the persistent slot's state.
        self.assertTrue(torch.equal(flat[2 * DRAFT], temporal[5]))
        self.assertTrue(torch.equal(flat[3 * DRAFT], temporal[7]))
        # The slot-indexed rows the old arithmetic used stay untouched.
        self.assertFalse(bool(flat[5 * DRAFT].any()))
        prev = gdn_commit.state(self.backend).prev
        self.assertEqual(int(prev[2]), 0)
        self.assertEqual(int(prev[3]), 0)
        self.assertEqual(int(prev[5]), -1)

    def test_seed_refuses_an_out_of_range_row(self):
        self.params.temporal.fill_(1.0)
        self.assertFalse(self._seed([5], [SCRATCH_ROWS]))
        flat = self.params.intermediate_ssm.reshape(
            -1, *self.params.intermediate_ssm.shape[2:])
        self.assertFalse(bool(flat.any()))

    def test_seed_refuses_a_negative_slot(self):
        self.assertFalse(self._seed([-1], [2]))


class CommitPrevTest(unittest.TestCase):
    def setUp(self):
        self.backend = types.SimpleNamespace()
        env = patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"})
        env.start()
        self.addCleanup(env.stop)
        st = gdn_commit.state(self.backend)
        st.prev = torch.full((SCRATCH_ROWS,), -1, dtype=torch.int32)
        st.commit_failed = False
        st.disabled = False
        st.step_ok = True
        st.logged = True

    def test_commit_advances_request_rows_only(self):
        st = gdn_commit.state(self.backend)
        rows = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        st.prev[rows.long()] = 0
        st.prev[6] = 2
        last = torch.tensor([DRAFT - 1, 1, DRAFT - 1, 0, 1],
                            dtype=torch.int32)
        self.assertTrue(
            gdn_commit.commit_prev(self.backend, rows, last, DRAFT, DRAFT))
        # Full accept rotates; truncated rows take the accepted step.
        self.assertEqual(int(st.prev[1]), 1)
        self.assertEqual(int(st.prev[2]), 1)
        self.assertEqual(int(st.prev[3]), 1)
        self.assertEqual(int(st.prev[4]), 0)
        self.assertEqual(int(st.prev[5]), 1)
        # Unseeded rows never gain a pointer.
        self.assertEqual(int(st.prev[6]), 2)
        self.assertEqual(int(st.prev[0]), -1)

    def test_commit_leaves_unseeded_rows_unseeded(self):
        st = gdn_commit.state(self.backend)
        rows = torch.tensor([1, 2], dtype=torch.int32)
        st.prev[1] = -1
        st.prev[2] = 3
        last = torch.tensor([DRAFT - 1, DRAFT - 1], dtype=torch.int32)
        self.assertTrue(
            gdn_commit.commit_prev(self.backend, rows, last, DRAFT, DRAFT))
        self.assertEqual(int(st.prev[1]), -1)
        self.assertEqual(int(st.prev[2]), (3 + 1) % DRAFT)

    def test_commit_refuses_after_a_declined_verify(self):
        st = gdn_commit.state(self.backend)
        st.step_ok = False
        rows = torch.tensor([1], dtype=torch.int32)
        st.prev[1] = 0
        self.assertFalse(
            gdn_commit.commit_prev(self.backend, rows,
                                   torch.tensor([0], dtype=torch.int32),
                                   DRAFT, DRAFT))
        self.assertEqual(int(st.prev[1]), 0)

    def test_out_of_range_rows_are_clamped_not_written_past(self):
        st = gdn_commit.state(self.backend)
        rows = torch.tensor([SCRATCH_ROWS, -1], dtype=torch.int32)
        last = torch.tensor([DRAFT - 1, DRAFT - 1], dtype=torch.int32)
        self.assertTrue(
            gdn_commit.commit_prev(self.backend, rows, last, DRAFT, DRAFT))
        self.assertEqual(int(st.prev.numel()), SCRATCH_ROWS)
        self.assertEqual(int(st.prev[0]), -1)


class CompensateMaskTest(unittest.TestCase):
    """Only truncated rows replay: a live row with no accepted step would run
    the whole recurrence for a snapshot write it never makes."""

    def setUp(self):
        env = patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.rows = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        self.prev = torch.tensor([-1, 0, 2, 0, 3][:SCRATCH_ROWS]
                                 + [-1] * (SCRATCH_ROWS - 5),
                                 dtype=torch.int32)
        inter = torch.zeros(SCRATCH_ROWS, DRAFT, 2)
        ent = gdn_commit.StashEntry(
            rows=BATCH * DRAFT,
            dims=gdn_commit.HeadDims(1, 2, 1, 2, 1, 2),
            qkv=torch.zeros(BATCH * DRAFT, 6),
            q=torch.zeros(BATCH * DRAFT, 2),
            k=torch.zeros(BATCH * DRAFT, 2),
            v=torch.zeros(BATCH * DRAFT, 2),
            a=torch.zeros(BATCH * DRAFT, 1),
            b=torch.zeros(BATCH * DRAFT, 1),
            out=torch.zeros(1, 1, 1, 2),
        )
        ent.A_log, ent.dt_bias = torch.zeros(1), torch.zeros(1)
        ent.layer_cache = types.SimpleNamespace(
            intermediate_ssm=inter, temporal=torch.zeros(SCRATCH_ROWS, 1, 2))
        self.backend = types.SimpleNamespace()
        self.st = gdn_commit.state(self.backend)
        self.st.prev = self.prev
        self.st.stash = {0: ent}
        self.st.step_ok = True
        self.st.commit_failed = False
        self.st.capture_T = DRAFT
        self.st.live_T = DRAFT

    def test_only_truncated_rows_carry_an_h0(self):
        captured = {}

        def fake_compensate(*args, **kwargs):
            captured["h0"] = args[8]
            captured["store_step"] = args[12]

        with patch.object(gdn_recurrent, "has_multi", lambda: False), patch.object(
                gdn_recurrent, "compensate", fake_compensate):
            ok = gdn_commit.compensate_from_stash(
                self.backend, self.rows,
                torch.tensor([DRAFT - 1, 1, DRAFT - 1, 0], dtype=torch.int32))
        self.assertTrue(ok)
        h0 = captured["h0"]
        # Rows 1 and 3 were truncated; 0 and 2 fully accepted.
        self.assertEqual([int(x) for x in h0], [-1, 1 * DRAFT + 0, -1, 3 * DRAFT + 0])
        self.assertEqual([int(x) for x in captured["store_step"]], [-1, 1, -1, 0])


class VerifyRowCacheTest(unittest.TestCase):
    """The batch-row map is rebuilt once per verify forward, not per layer."""

    def setUp(self):
        env = patch.dict(os.environ, {"SGLANG_GDN_HIP": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.backend = types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(
                size=POOL_ROWS, _alloc_size=POOL_ROWS + 1))
        self.st = gdn_commit.state(self.backend)
        self.st.prev = torch.zeros(SCRATCH_ROWS, dtype=torch.int32)
        self.st.disabled = False
        self.st.geometry = None
        self.st.step_lid = -1
        self.st.step_cache = None
        self.req_rows = torch.tensor([3, 5], dtype=torch.int32)
        self.mamba_cache = _params()
        self.seen = []

    def _verify(self, layer_id):
        captured = {}

        def fake_verify(*args, **kwargs):
            captured["snap_idx"] = args[11]

        layer = types.SimpleNamespace(layer_id=layer_id, A_log=None, dt_bias=None)
        q = torch.zeros(1, 2 * DRAFT, 1, 2)
        v = torch.zeros(1, 2 * DRAFT, 1, 2)
        with patch.object(gdn_commit, "stash", lambda *a, **k: True), patch.object(
                gdn_recurrent, "verify", fake_verify):
            out = gdn_commit.verify_step(
                backend=self.backend, layer=layer, q=q, k=q, v=v, a=q, b=q,
                draft_token_num=DRAFT, n=2,
                snap_cache=self.mamba_cache.intermediate_ssm,
                cache_indices=torch.zeros(2, dtype=torch.int32),
                mamba_cache=self.mamba_cache,
                query_start_loc=torch.arange(0, 2 * DRAFT + 1, DRAFT,
                                             dtype=torch.int32),
                req_rows=self.req_rows)
        self.assertIsNotNone(out)
        return captured["snap_idx"]

    def test_rows_are_shared_inside_one_forward(self):
        first = self._verify(1)
        second = self._verify(2)
        self.assertIs(first, second)
        self.assertEqual([int(x) for x in first], [3, 5])

    def test_a_new_forward_rebuilds_the_rows(self):
        first = self._verify(1)
        # Same layer id again: begin_verify_step rotates the step cache.
        second = self._verify(1)
        self.assertIsNot(first, second)
        self.assertEqual([int(x) for x in second], [3, 5])


if __name__ == "__main__":
    unittest.main()
