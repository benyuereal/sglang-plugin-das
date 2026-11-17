import sys
import types
import unittest
from unittest import mock

import torch

from hcu_sglang import speculative_sampling as subject


class TestSpeculativeSampling(unittest.TestCase):
    def setUp(self):
        self.info = types.SimpleNamespace(
            sampling_seed=None, need_min_p_sampling=False,
            temperatures=torch.tensor([[1.0], [2.0]]),
            top_ks=torch.tensor([2, 3]), top_ps=torch.tensor([0.8, 0.9]),
        )
        self.spec = types.SimpleNamespace(
            speculative_algorithm="EAGLE", speculative_eagle_topk=1,
            speculative_use_rejection_sampling=False,
        )
        self.context = types.ModuleType("sglang.srt.runtime_context")
        self.context.get_spec = lambda: self.spec

    def test_registers_sampling_hooks_once_through_speculative_entry(self):
        from hcu_sglang import speculative_compat
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        with mock.patch.object(HookRegistry, "register") as register, mock.patch.object(
            speculative_compat, "patch_broken_torchaudio_availability"
        ):
            speculative_compat.register_hooks()
        expected = {
            "sglang.srt.speculative.eagle_utils._can_use_sparse_uno_tree_target_sampling": subject.around_can_use_sparse_target_sampling,
            "sglang.srt.speculative.uno_utils.sample_uno_tree_target_tokens": subject.around_sample_tree_target_tokens,
        }
        for target, hook in expected.items():
            self.assertEqual(
                [call for call in register.call_args_list if call.args[0] == target],
                [mock.call(target, hook, HookType.AROUND)],
            )

    def test_policy_and_fallback(self):
        with mock.patch.object(subject, "is_hcu", return_value=True), mock.patch.dict(
            sys.modules, {"sglang.srt.runtime_context": self.context}
        ):
            original = mock.Mock(return_value="upstream")
            self.assertTrue(subject.around_can_use_sparse_target_sampling(original, None, self.info))
            self.assertEqual(subject.around_can_use_sparse_target_sampling(original, 20, self.info), "upstream")
            for field, value in (("speculative_algorithm", "NGRAM"), ("speculative_eagle_topk", 2), ("speculative_use_rejection_sampling", True)):
                before = getattr(self.spec, field)
                setattr(self.spec, field, value)
                self.assertEqual(subject.around_can_use_sparse_target_sampling(original, None, self.info), "upstream")
                setattr(self.spec, field, before)
            self.info.sampling_seed = torch.tensor([7])
            self.assertEqual(subject.around_can_use_sparse_target_sampling(original, None, self.info), "upstream")
            self.info.sampling_seed = None
            self.info.need_min_p_sampling = True
            self.assertEqual(subject.around_can_use_sparse_target_sampling(original, None, self.info), "upstream")
        with mock.patch.object(subject, "is_hcu", return_value=False):
            self.assertEqual(subject.around_can_use_sparse_target_sampling(original, None, self.info), "upstream")

    def test_sampler_expands_parameters_and_uses_joint_filter(self):
        logits = torch.tensor([[0., 1., 2.], [2., 1., 0.], [0., 2., 1.], [1., 0., 2.]])
        sampler = mock.Mock(return_value=torch.tensor([2, 0, 1, 2], dtype=torch.int32))
        module = types.ModuleType("lightop.sampling")
        module.top_k_top_p_sampling_from_probs = sampler
        with mock.patch.object(subject, "_uses_hcu_mtp", return_value=True), mock.patch.dict(
            sys.modules, {"lightop.sampling": module}
        ):
            out = subject.around_sample_tree_target_tokens(
                mock.Mock(), next_token_logits=logits, sampling_info=self.info,
                batch_size=2, verify_width=2, max_top_k=None,
            )
        self.assertEqual(out.tolist(), [[2, 0], [1, 2]])
        self.assertEqual(out.dtype, torch.int64)
        args, kwargs = sampler.call_args
        torch.testing.assert_close(args[0], torch.softmax(logits / torch.tensor([[1.], [1.], [2.], [2.]]), -1))
        self.assertEqual(args[1].tolist(), [2, 2, 3, 3])
        torch.testing.assert_close(args[2], torch.tensor([0.8, 0.8, 0.9, 0.9]))
        self.assertEqual(kwargs, {"filter_apply_order": "joint", "deterministic": True})

    def test_sampler_preserves_original(self):
        original = mock.Mock(return_value="upstream")
        with mock.patch.object(subject, "_uses_hcu_mtp", return_value=False):
            result = subject.around_sample_tree_target_tokens(
                original, next_token_logits="logits", sampling_info=self.info,
                batch_size=2, verify_width=2, max_top_k=20,
            )
        self.assertEqual(result, "upstream")
        original.assert_called_once_with(next_token_logits="logits", sampling_info=self.info, batch_size=2, verify_width=2, max_top_k=20)


if __name__ == "__main__":
    unittest.main()
