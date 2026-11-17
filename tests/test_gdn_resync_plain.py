# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The handoff -> plain resync must keep the layer axis intact."""

from __future__ import annotations

import types
import unittest

import torch

from hcu_sglang.layers.attention.linear import gdn_commit


class ResyncPlainTest(unittest.TestCase):
    def test_each_slot_is_copied_from_its_own_layer(self):
        layers, spec_rows, draft, hv, v, k = 2, 2, 3, 2, 2, 2
        inter = torch.arange(
            layers * spec_rows * draft * hv * v * k, dtype=torch.float32
        ).reshape(layers, spec_rows, draft, hv, v, k)
        temporal = torch.full((layers, 4, hv, v, k), -1.0)
        backend = types.SimpleNamespace()
        st = gdn_commit.state(backend)
        st.prev = torch.tensor([1, -1], dtype=torch.int32)
        mamba_cache = types.SimpleNamespace(
            intermediate_ssm=inter, temporal=temporal
        )
        rows = torch.tensor([0, -1], dtype=torch.int32)
        slots = torch.tensor([2, 3], dtype=torch.int32)

        gdn_commit.resync_plain(
            backend, rows, slots, mamba_cache, draft, 2
        )

        for layer in range(layers):
            self.assertTrue(
                torch.equal(temporal[layer, 2], inter[layer, 0, 1])
            )
            self.assertTrue(torch.all(temporal[layer, 3] == -1))
            self.assertTrue(torch.all(temporal[layer, 0] == -1))
        self.assertEqual(st.prev.tolist(), [-1, -1])


if __name__ == "__main__":
    unittest.main()
