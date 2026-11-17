from types import SimpleNamespace
import unittest

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from hcu_sglang.environ import register_envs

register_envs()

from hcu_sglang.speculative_dspark_worker import build_verify_generation_result, build_verify_epilogue
from sglang.srt.speculative.dspark_components.dspark_verify import DsparkVerifyEpilogue, TargetVerifyResult


class EpilogueContractTests(unittest.TestCase):
    def test_decode_result_preserves_native_fields_and_optional_metrics(self):
        token_ids = torch.tensor([19])
        verify = TargetVerifyResult(logits_output=object(), can_run_cuda_graph=True)
        result = build_verify_generation_result(verify, next_token_ids=token_ids)
        self.assertIs(result.next_token_ids, token_ids)
        self.assertIsNone(result.expert_distribution_metrics)
        marker = object()
        result = build_verify_generation_result(SimpleNamespace(expert_distribution_metrics=marker))
        self.assertIs(result.expert_distribution_metrics, marker)

    def options(self):
        return dict(max_bs=4, verify_num_draft_tokens=6, device='cpu', tp_sync=object())

    def test_fixed_official_constructor_and_static_step_buffers(self):
        options=self.options()
        with self.assertRaisesRegex(TypeError,'simulate_fold'):
            DsparkVerifyEpilogue(simulate_fold=False,**options)
        epilogue=build_verify_epilogue(DsparkVerifyEpilogue,simulate_fold=False,**options)
        self.assertIs(type(epilogue),DsparkVerifyEpilogue)
        self.assertIs(epilogue._tp_sync,options['tp_sync'])
        epilogue.begin_static_step(3,True)
        torch.testing.assert_close(epilogue.verify_lens_buf,torch.tensor([6,6,6,0]))
        self.assertEqual(epilogue.inject_gate_buf.item(),1)
        epilogue.begin_static_step(1,False)
        torch.testing.assert_close(epilogue.verify_lens_buf,torch.tensor([6,0,0,0]))
        self.assertEqual(epilogue.inject_gate_buf.item(),0)

    def test_unsupported_simulation_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError,'simulate_fold'):
            build_verify_epilogue(DsparkVerifyEpilogue,simulate_fold=True,**self.options())

    def test_provider_with_simulation_preserves_flag_and_errors(self):
        class Provider:
            def __init__(self,*,simulate_fold,marker):
                self.simulate_fold=simulate_fold
                self.marker=marker
        marker=object()
        result=build_verify_epilogue(Provider,simulate_fold=True,marker=marker)
        self.assertTrue(result.simulate_fold)
        self.assertIs(result.marker,marker)
        with self.assertRaises(TypeError):
            build_verify_epilogue(Provider,simulate_fold=False,unexpected=marker)


if __name__=='__main__':
    unittest.main()
