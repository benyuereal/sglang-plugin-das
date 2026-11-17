import unittest
from hcu_sglang.models.qwen.prefill_policy import compact_prefill_counts, pure_prefill_modes


class CompactPolicyTest(unittest.TestCase):
    def test_pure_modes_exclude_verify_and_mixed(self):
        self.assertTrue(pure_prefill_modes([1, 4, 4, 1], extend=1, idle=4))
        for modes in ([], [4, 4], [1, 2], [1, 3], [1, 5]):
            self.assertFalse(pure_prefill_modes(modes, extend=1, idle=4))

    def test_idle_ranks_keep_one_transport_row(self):
        self.assertEqual(compact_prefill_counts([0, 32768, 642, 0], pure_prefill=True, eligible=True), [1, 32768, 642, 1])
        for pure, eligible in ((False, True), (True, False)):
            self.assertIsNone(compact_prefill_counts([0, 17], pure_prefill=pure, eligible=eligible))
        self.assertIsNone(compact_prefill_counts([0, 0], pure_prefill=True, eligible=True))
        with self.assertRaises(ValueError):
            compact_prefill_counts([3, -1], pure_prefill=True, eligible=True)
