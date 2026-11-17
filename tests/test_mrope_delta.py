# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import collections
from contextlib import ExitStack
from pathlib import Path
import gc
import io
import pickle
import pickletools
import types
import unittest
import weakref
from unittest import mock

import torch

from hcu_sglang import mrope_delta as subject
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor import forward_batch_info as native
from sglang.srt.plugins.hook_registry import HookRegistry


class MropeDeltaTests(unittest.TestCase):
    def setUp(self):
        subject._delta_cache.clear()
        self.runner = types.SimpleNamespace(device="cpu")
        self.execution = types.SimpleNamespace(
            deterministic=types.SimpleNamespace(rl_on_policy_target=None)
        )
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(subject._delta_cache.clear)
        self.stack.enter_context(mock.patch.object(subject, "is_hcu", return_value=True))
        self.stack.enter_context(mock.patch.object(subject, "is_pin_memory_available", return_value=False))
        self.stack.enter_context(mock.patch.object(subject, "get_exec", return_value=self.execution))
        self.stack.enter_context(mock.patch.object(native, "get_exec", return_value=self.execution))
        self.original_spec = native.ForwardBatch.compute_spec_mrope_positions
        self.original_decode = native.ForwardBatch._compute_mrope_positions_decode
        self.owner = type("MropeTestForwardBatch", (native.ForwardBatch,), {})
        self.stack.enter_context(mock.patch.object(native, "ForwardBatch", self.owner))
        self.stack.enter_context(mock.patch.object(HookRegistry, "_hooks", collections.defaultdict(list)))
        self.stack.enter_context(mock.patch.object(HookRegistry, "_patched", set()))
        subject.register_hooks()
        HookRegistry.apply_hooks()

    def mm(self, delta=-7, positions=None):
        return MultimodalInputs(
            mm_items=[], mrope_position_delta=torch.tensor([[delta]], dtype=torch.int64),
            mrope_positions=positions,
        )

    def forward(self, lengths):
        seq = torch.tensor(lengths, dtype=torch.int64)
        return self.owner(
            forward_mode=native.ForwardMode.DECODE, batch_size=len(lengths),
            input_ids=torch.zeros(len(lengths), dtype=torch.int64),
            req_pool_indices=torch.arange(len(lengths)), seq_lens=seq,
            out_cache_loc=torch.arange(len(lengths)), seq_lens_sum=sum(lengths),
            seq_lens_cpu=seq.clone(),
        )

    def batch(self, mm, positions=None):
        return types.SimpleNamespace(
            multimodal_inputs=mm, spec_info=types.SimpleNamespace(positions=positions)
        )

    def test_actual_registry_targets_applied(self):
        prefix = "sglang.srt.model_executor.forward_batch_info.ForwardBatch."
        self.assertEqual(HookRegistry._patched, {
            prefix + "compute_spec_mrope_positions", prefix + "_compute_mrope_positions_decode"
        })
        self.assertIn("mrope_delta.register_hooks()", Path(subject.__file__).with_name("__init__.py").read_text())

    def test_spec_positions_mixed_batch_and_explicit_draft_extend(self):
        for width in (1, 3, 4):
            with self.subTest(width=width):
                positions = torch.arange(3 * width).reshape(3, width) + 20
                batch = self.batch([self.mm(-7), None, self.mm(5)], positions)
                ref, out = self.forward([21, 22, 23]), self.forward([21, 22, 23])
                self.original_spec(ref, self.runner, batch)
                out.compute_spec_mrope_positions(self.runner, batch)
                torch.testing.assert_close(out.mrope_positions, ref.mrope_positions)
                out.compute_spec_mrope_positions(self.runner, batch, seq_positions=positions + 10)
                self.original_spec(ref, self.runner, batch, seq_positions=positions + 10)
                torch.testing.assert_close(out.mrope_positions, ref.mrope_positions)
                self.assertEqual(out.mrope_positions.dtype, torch.int64)

    def test_decode_positions_equal_native_in_mixed_batch(self):
        batch = self.batch([self.mm(-7), None, self.mm(5)])
        ref, out = self.forward([21, 22, 23]), self.forward([21, 22, 23])
        self.original_decode(ref, self.runner, batch)
        out._compute_mrope_positions_decode(self.runner, batch)
        torch.testing.assert_close(out.mrope_positions, ref.mrope_positions)
        self.assertEqual(out.mrope_positions[0].tolist(), [13, 21, 27])

    def test_decode_and_spec_share_one_request_cache(self):
        mm = self.mm()
        batch = self.batch([mm], torch.tensor([20, 21, 22]))
        out = self.forward([21])
        out.compute_spec_mrope_positions(self.runner, batch)
        cached = subject._mrope_delta_device(mm, "cpu")
        for _ in range(4):
            out.compute_spec_mrope_positions(self.runner, batch)
            out._compute_mrope_positions_decode(self.runner, batch)
            self.assertIs(subject._mrope_delta_device(mm, "cpu"), cached)
        self.assertEqual(len(subject._delta_cache), 1)

    def test_cache_is_not_added_to_inputs_or_serialized(self):
        mm = self.mm()
        fields = set(vars(mm))
        cached = subject._mrope_delta_device(mm, "cpu")
        self.assertEqual(set(vars(mm)), fields)
        self.assertIsNot(cached, mm.mrope_position_delta)

        class RejectDeviceCachePickler(pickle.Pickler):
            def reducer_override(self, obj):
                if obj is cached:
                    raise AssertionError("device cache entered serialization")
                return NotImplemented

        for protocol in {pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL}:
            with self.subTest(protocol=protocol):
                payload = io.BytesIO()
                RejectDeviceCachePickler(payload, protocol=protocol).dump(mm)
                strings = {
                    arg for opcode, arg, _ in pickletools.genops(payload.getvalue())
                    if opcode.name in {"UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8"}
                }
                self.assertTrue(fields.issubset(strings))
                with self.assertRaisesRegex(AssertionError, "device cache entered serialization"):
                    RejectDeviceCachePickler(io.BytesIO(), protocol=protocol).dump(cached)

    def test_request_collection_releases_cache(self):
        mm = self.mm()
        key, ref = id(mm), weakref.ref(mm)
        subject._mrope_delta_device(mm, "cpu")
        del mm
        gc.collect()
        self.assertIsNone(ref())
        self.assertNotIn(key, subject._delta_cache)

    def test_source_replacement_and_mutation_invalidate(self):
        mm = self.mm()
        first = subject._mrope_delta_device(mm, "cpu")
        mm.mrope_position_delta = torch.tensor([[8]])
        second = subject._mrope_delta_device(mm, "cpu")
        self.assertIsNot(first, second)
        self.assertEqual(second.tolist(), [8])
        mm.mrope_position_delta.add_(3)
        third = subject._mrope_delta_device(mm, "cpu")
        self.assertIsNot(second, third)
        self.assertEqual(third.tolist(), [11])

    def test_inference_tensor_source_supported(self):
        with torch.inference_mode():
            mm = self.mm()
            value = subject._mrope_delta_device(mm, "cpu")
            self.assertIs(subject._mrope_delta_device(mm, "cpu"), value)

    def test_text_rl_and_precomputed_paths_preserve_effective_original(self):
        precomputed = self.mm(0, torch.arange(90).reshape(3, 30))
        cases = [([None, None], None), ([self.mm(), None], "policy"), ([precomputed, None], None)]
        for mm, policy in cases:
            self.execution.deterministic.rl_on_policy_target = policy
            batch = self.batch(mm)
            out, ref = self.forward([21, 22]), self.forward([21, 22])
            with mock.patch.object(subject, "_mrope_delta_device", side_effect=AssertionError("cache must be bypassed")):
                out._compute_mrope_positions_decode(self.runner, batch)
            self.original_decode(ref, self.runner, batch)
            torch.testing.assert_close(out.mrope_positions, ref.mrope_positions)
        fallback = mock.Mock(return_value="effective baseline")
        self.assertEqual(subject.around_compute_mrope_positions_decode(fallback, out, self.runner, batch), "effective baseline")
        fallback.assert_called_once_with(out, self.runner, batch)

    def test_non_hcu_keeps_baseline_and_registers_nothing(self):
        original = mock.Mock(return_value="baseline")
        out, batch = self.forward([21]), self.batch([self.mm()], torch.tensor([20]))
        with mock.patch.object(subject, "is_hcu", return_value=False), mock.patch.object(HookRegistry, "register") as register:
            self.assertEqual(subject.around_compute_spec_mrope_positions(original, out, self.runner, batch), "baseline")
            self.assertEqual(subject.around_compute_mrope_positions_decode(original, out, self.runner, batch), "baseline")
            subject.register_hooks()
            register.assert_not_called()


if __name__ == "__main__":
    unittest.main()
