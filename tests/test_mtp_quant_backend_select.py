"""The fp8 draft head must not depend on one GEMM backend."""

import os
import unittest
from unittest import mock

import torch

from hcu_sglang.layers.gemm import quant_lm_head


class QuantHeadBackendSelectTests(unittest.TestCase):
    def setUp(self):
        self.weight = torch.zeros(2, 4)
        self.gemm = object()

    def _prepare(self, backends, name):
        with mock.patch.dict(os.environ, {
            "SGLANG_QUANT_HEAD_BACKEND": name,
        }), mock.patch.object(quant_lm_head, "_BACKENDS", backends):
            return quant_lm_head.prepare(self.weight)

    def test_explicit_backend_is_used(self):
        backends = {"lightop": lambda w: (self.gemm, 192)}
        self.assertEqual(self._prepare(backends, "lightop"), (self.gemm, 192))

    def test_auto_falls_through_to_the_next_backend(self):
        def broken(_weight):
            raise RuntimeError("no workspace")

        backends = {
            "hbird": broken,
            "lightop": lambda w: (self.gemm, 192),
            "torch": lambda w: (object(), 192),
        }
        self.assertEqual(self._prepare(backends, "auto"), (self.gemm, 192))

    def test_declining_backend_falls_through(self):
        backends = {
            "hbird": lambda w: (None, 0),
            "lightop": lambda w: (self.gemm, 192),
        }
        self.assertEqual(self._prepare(backends, "auto"), (self.gemm, 192))

    def test_unknown_backend_declines(self):
        self.assertEqual(self._prepare({}, "nope"), (None, 0))

    def test_with_out_copies_into_the_graph_buffer(self):
        out = torch.zeros(2, 3)
        result = quant_lm_head._with_out(torch.ones(2, 3), out)
        self.assertIs(result, out)
        self.assertTrue(torch.equal(out, torch.ones(2, 3)))


if __name__ == "__main__":
    unittest.main()
