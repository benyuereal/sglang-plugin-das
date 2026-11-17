from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()
from hcu_sglang.mem_cache.hcu_mha_pool import HCUMHATokenToKVPool


class HcuMhaSingleStreamTests(unittest.TestCase):
    def pool(self, **overrides):
        options = dict(size=128, page_size=64, dtype=torch.bfloat16,
                       head_num=2, head_dim=16, v_head_dim=8, layer_num=2,
                       device="cuda", enable_memory_saver=False,
                       start_layer=3, end_layer=5, kv_cache_layout="nhd")
        options.update(overrides)
        return HCUMHATokenToKVPool(**options)

    def test_constructor_ignores_upstream_alt_stream_request(self):
        for requested in (True, False):
            with self.subTest(requested=requested):
                pool = self.pool(enable_alt_stream=requested)
                self.assertIsNone(pool.alt_stream)
                self.assertEqual(tuple(pool.k_buffer[0].shape), (3, 2, 64, 16))
                self.assertEqual(tuple(pool.v_buffer[0].shape), (3, 2, 8, 64))

    def test_positional_constructor_and_duplicate_argument_contract(self):
        args = (128, 64, torch.bfloat16, 2, 16, 2, "cuda", False,
                8, None, None, None, 3, 5, True, False, "nhd")
        pool = HCUMHATokenToKVPool(*args)
        self.assertIsNone(pool.alt_stream)
        self.assertEqual(pool.v_head_dim, 8)
        self.assertEqual(pool.start_layer, 3)
        with self.assertRaises(TypeError):
            HCUMHATokenToKVPool(*args, enable_alt_stream=True)

    def check_state(self, pool, expected_k, expected_v):
        for actual, expected in zip(pool.k_buffer, expected_k):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        for actual, expected in zip(pool.v_buffer, expected_v):
            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    def test_public_write_and_changed_input_graph_replay(self):
        pool = self.pool(enable_alt_stream=True)
        self.assertIsNone(pool.alt_stream)
        layer = SimpleNamespace(layer_id=3)
        loc = torch.tensor([0, 63, 64, 127, 128], device="cuda", dtype=torch.int64)
        key = torch.arange(160, device="cuda", dtype=torch.float32).reshape(5, 2, 16).to(torch.bfloat16)
        value = torch.arange(80, device="cuda", dtype=torch.float32).reshape(5, 2, 8).to(torch.bfloat16)
        pool.set_kv_buffer(layer, loc, key, value)
        expected_k = [torch.zeros_like(t, device="cpu") for t in pool.k_buffer]
        expected_v = [torch.zeros_like(t, device="cpu") for t in pool.v_buffer]
        def expected_write(positions, keys, values):
            for i, position in enumerate(positions):
                page, offset = divmod(position, pool.page_size)
                expected_k[0][page, :, offset, :] = keys[i]
                expected_v[0][page, :, :, offset] = values[i]
        expected_write(loc.cpu().tolist(), key.cpu(), value.cpu())
        self.check_state(pool, expected_k, expected_v)
        empty = loc[:0]
        pool.set_kv_buffer(layer, empty, key[:0], value[:0])
        self.check_state(pool, expected_k, expected_v)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            pool.set_kv_buffer(layer, loc, key, value)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with patch("sglang.srt.model_executor.runner.get_is_capture_mode", return_value=True):
            with torch.cuda.graph(graph):
                pool.set_kv_buffer(layer, loc, key, value)
        for delta, positions in [(100, [1, 62, 65, 126, 129]), (200, [2, 61, 66, 125, 130])]:
            loc.copy_(torch.tensor(positions, device="cuda", dtype=torch.int64))
            key.copy_((torch.arange(160, device="cuda").reshape(5, 2, 16) + delta).to(torch.bfloat16))
            value.copy_((torch.arange(80, device="cuda").reshape(5, 2, 8) + delta).to(torch.bfloat16))
            graph.replay()
            torch.cuda.synchronize()
            expected_write(positions, key.cpu(), value.cpu())
            self.check_state(pool, expected_k, expected_v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
