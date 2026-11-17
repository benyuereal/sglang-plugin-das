from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs

register_envs()
from hcu_sglang import kernel_compat as K
from sglang.kernels.ops.moe import moe_fused_gate, moe_route_quant_fused, moe_route_radix
from sglang.srt.plugins.hook_registry import HookRegistry

TARGET = 'sglang.kernels.ops.moe.moe_route_radix.covered'


def k3_router_inputs():
    scores = torch.zeros(4, 896, dtype=torch.bfloat16)
    bias = torch.zeros(896, dtype=torch.float32)
    return scores, bias, 16


class MoeRouteRadixCoveredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset()
        K.register_hooks()
        HookRegistry.apply_hooks()

    def test_real_registration_patches_the_shared_module_both_consumers_read(self):
        self.assertIn(TARGET, HookRegistry._patched)
        self.assertIs(moe_fused_gate.moe_route_radix, moe_route_radix)
        self.assertIs(moe_route_quant_fused.moe_route_radix, moe_route_radix)

    def test_upstream_predicate_accepts_the_k3_decode_shape(self):
        scores, bias, topk = k3_router_inputs()
        with patch.object(K, 'is_hcu', return_value=False):
            self.assertTrue(moe_route_radix.covered(scores, bias, topk))
            self.assertFalse(moe_route_radix.covered(scores, bias, 8))

    def test_hcu_reports_uncovered_so_the_triton_router_keeps_the_dispatch(self):
        scores, bias, topk = k3_router_inputs()
        frozen = scores.clone()
        self.assertFalse(moe_route_radix.covered(scores, bias, topk))
        self.assertFalse(moe_route_quant_fused.covered(
            scores, bias, topk, torch.zeros(4, 3584, dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(scores, frozen))

    def test_non_hcu_around_forwards_arguments_to_the_effective_prior_wrapper(self):
        scores, bias, topk = k3_router_inputs()
        marker = object()
        original = Mock(return_value=marker)
        with patch.object(K, 'is_hcu', return_value=False):
            self.assertIs(K.around_moe_route_radix_covered(original, scores, bias, topk), marker)
        original.assert_called_once_with(scores, bias, topk)


if __name__ == '__main__':
    unittest.main()
