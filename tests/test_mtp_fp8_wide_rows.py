"""Wide rows must not call ParallelLMHead.forward (it raises)."""

import unittest

import torch

from hcu_sglang.models.mtp_lmhead_quant import DraftHeadQuantMethod, SCHEME


class _Head:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x):  # what ParallelLMHead.forward does
        raise RuntimeError("LMHead's weights should be used in the sampler.")


class WideRowsTest(unittest.TestCase):
    def test_wide_rows_use_the_weight_not_the_module(self):
        vocab, hidden, rows = 8, 4, 5
        weight = torch.randn(vocab, hidden, dtype=torch.bfloat16)
        method = DraftHeadQuantMethod(
            _Head(weight),
            torch.empty((vocab, hidden), dtype=torch.float8_e4m3fn),
            torch.ones((vocab, 1)),
            gemm=None,
            max_m=rows - 1,
            scheme=SCHEME,
        )
        x = torch.randn(rows, hidden, dtype=torch.bfloat16)
        out = method.apply(None, x)
        torch.testing.assert_close(out, torch.matmul(x, weight.t()))


if __name__ == "__main__":
    unittest.main()
