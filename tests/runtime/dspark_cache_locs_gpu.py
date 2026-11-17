import unittest

import torch

from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func as early_assign
from hcu_sglang import kernel_compat
from sglang.srt.plugins.hook_registry import HookRegistry


class DSparkCacheLocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset()
        kernel_compat.register_hooks()
        HookRegistry.apply_hooks()

    def test_window_gathers_from_request_table_without_writing_it(self):
        table = (17 + torch.arange(32, device="cuda", dtype=torch.int32)).view(2, 16)
        rows = torch.tensor([1, 0], device="cuda", dtype=torch.int64)
        starts = torch.tensor([2, 6], device="cuda", dtype=torch.int32)
        ends = starts + 6
        expected = torch.cat((table[1, 2:8], table[0, 6:12])).long()
        snapshot = table.clone()
        slots = early_assign(rows, table, starts, ends, 2, 6, "cuda")
        torch.testing.assert_close(table, snapshot)
        torch.testing.assert_close(slots, expected)

    def test_replay_updates_live_rows_and_offsets(self):
        table = torch.arange(32, device="cuda", dtype=torch.int32).view(2, 16)
        row = torch.tensor([0], device="cuda", dtype=torch.int64)
        start = torch.tensor([1], device="cuda", dtype=torch.int32)
        end = start + 6
        early_assign(row, table, start, end, 1, 6, "cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            slots = early_assign(row, table, start, end, 1, 6, "cuda")
        for idx, offset in ((1, 3), (0, 7)):
            row.fill_(idx)
            start.fill_(offset)
            end.fill_(offset + 6)
            graph.replay()
            torch.testing.assert_close(slots, table[idx, offset:offset+6].long())


if __name__ == "__main__":
    unittest.main()
