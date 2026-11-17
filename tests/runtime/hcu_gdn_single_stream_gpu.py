import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from sglang.srt.models import qwen3_5 as M
from hcu_sglang.models.qwen import qwen3_5 as P
from sglang.srt.plugins.hook_registry import HookRegistry


class GDNStreamTests(unittest.TestCase):
    def make_layer(self, enabled=True):
        calls = []
        stream = torch.cuda.Stream()
        def initialize(layer, config, layer_id, quant_config=None, alt_stream=None, prefix=""):
            torch.nn.Module.__init__(layer)
            calls.append((config, layer_id, quant_config, alt_stream, prefix))
            layer.alt_stream = alt_stream
            layer._fused_in_proj_weight = None
            layer._fused_input_proj_cpu_enabled = SimpleNamespace(value=False)
        HookRegistry.reset()
        with patch.object(M.Qwen3_5GatedDeltaNet, '__init__', initialize):
            P.register()
            HookRegistry.apply_hooks()
            with patch.object(P, 'is_hcu', return_value=enabled):
                layer = M.Qwen3_5GatedDeltaNet('config', 7, None, stream, prefix='gdn')
        self.assertEqual(calls, [('config', 7, None, stream, 'gdn')])
        return layer, stream

    def test_hcu_constructor_disables_alt(self):
        layer, stream = self.make_layer()
        self.assertIsNone(layer.alt_stream)

    def test_non_hcu_retains_effective_original(self):
        layer, stream = self.make_layer(False)
        self.assertIs(layer.alt_stream, stream)

    def test_constructor_error_preserves_original_failure_and_state(self):
        stream = object()
        layer = SimpleNamespace(alt_stream=stream)
        error = ValueError("invalid constructor configuration")
        initialize = Mock(side_effect=error)
        with patch.object(P, "is_hcu", return_value=True):
            with self.assertRaises(ValueError) as raised:
                P.around_gdn_init(initialize, layer, "config", 7, prefix="gdn")
        self.assertIs(raised.exception, error)
        self.assertIs(layer.alt_stream, stream)
        initialize.assert_called_once_with(layer, "config", 7, prefix="gdn")

    def test_real_projection_and_dynamic_graph_replay(self):
        layer, stream = self.make_layer()
        self.assertIsNone(layer.alt_stream)
        torch.manual_seed(17)
        x = torch.randn(1, 16, device='cuda', dtype=torch.bfloat16)
        q = torch.randn(16, 32, device='cuda', dtype=torch.bfloat16)
        b = torch.randn(16, 8, device='cuda', dtype=torch.bfloat16)
        observed = []
        def projection(weight):
            def call(value):
                observed.append(torch.cuda.current_stream().cuda_stream)
                return value @ weight, None
            return call
        layer.in_proj_qkvz = projection(q)
        layer.in_proj_ba = projection(b)
        with patch.object(M, '_use_aiter', False), patch.object(M, '_is_cpu', False), patch.object(M, '_is_npu', False), patch.object(M, '_gdn_use_alt_stream', True), patch.object(M, 'check_cuda_graph_backend', return_value=False):
            for rows in (0, 1, 8, 1024):
                value = torch.randn(rows, 16, device='cuda', dtype=torch.bfloat16)
                with patch.object(M, 'get_is_capture_mode', return_value=False):
                    outputs = layer._forward_input_proj(value)
                for result, weight in zip(outputs, (q, b)):
                    torch.testing.assert_close(result, value @ weight, rtol=0, atol=0)
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                with patch.object(M, 'get_is_capture_mode', return_value=False):
                    for _ in range(3): layer._forward_input_proj(x)
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            observed.clear()
            with patch.object(M, 'get_is_capture_mode', return_value=True), patch.object(torch.cuda, 'stream', wraps=torch.cuda.stream) as switch:
                with torch.cuda.graph(graph, stream=capture_stream):
                    outputs = layer._forward_input_proj(x)
                self.assertEqual(observed, [capture_stream.cuda_stream] * 2)
                self.assertTrue(all(call.args[0] is not stream for call in switch.call_args_list))
            for shift in (0.25, -0.5):
                x.add_(shift)
                graph.replay()
                torch.cuda.synchronize()
                for result, weight in zip(outputs, (q, b)):
                    torch.testing.assert_close(result, x @ weight, rtol=0, atol=0)
        print('GDN_REAL_PROJECTION_DYNAMIC_GRAPH_OK', flush=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
