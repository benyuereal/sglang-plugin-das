# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch

from hcu_sglang.multimodal_gen import minimax_h3


class MiniMaxH3RotaryDispatchTests(unittest.TestCase):
    def setUp(self):
        self.query = torch.zeros(3, 2, 8)
        self.key = torch.zeros_like(self.query)
        self.cache = torch.zeros(16, 8)
        self.positions = torch.arange(3)

    def test_native_keyword_preserves_identity_on_both_platforms(self):
        rotary = object()
        seen = []

        def native(query, key, cache, positions, *, rotary_emb):
            seen.append((query, key, cache, positions, rotary_emb))
            return query, key

        for hcu in (False, True):
            with self.subTest(hcu=hcu), patch.object(minimax_h3, 'is_hcu', return_value=hcu):
                result = minimax_h3.around_apply_rope_qk(
                    native, self.query, self.key, self.cache, self.positions,
                    rotary_emb=rotary,
                )
                self.assertIs(result[0], self.query)
                self.assertIs(result[1], self.key)
                self.assertEqual(len(seen), 1)
                for actual, expected in zip(seen.pop(), (
                    self.query, self.key, self.cache, self.positions, rotary,
                )):
                    self.assertIs(actual, expected)

    def test_explicit_none_is_forwarded(self):
        seen = []

        def native(query, key, cache, positions, *, rotary_emb):
            seen.append(rotary_emb)
            return query, key

        with patch.object(minimax_h3, 'is_hcu', return_value=True):
            minimax_h3.around_apply_rope_qk(
                native, self.query, self.key, self.cache, self.positions,
                rotary_emb=None,
            )
        self.assertEqual(seen, [None])

    def test_native_error_is_not_swallowed(self):
        def native(query, key, cache, positions, *, rotary_emb):
            raise RuntimeError('native rotary contract failed')

        with patch.object(minimax_h3, 'is_hcu', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'native rotary contract failed'):
                minimax_h3.around_apply_rope_qk(
                    native, self.query, self.key, self.cache, self.positions,
                    rotary_emb=object(),
                )

    def test_legacy_cpu_call_does_not_gain_a_keyword(self):
        def legacy(query, key, cache, positions):
            return query, key

        for hcu in (False, True):
            with self.subTest(hcu=hcu), patch.object(minimax_h3, 'is_hcu', return_value=hcu):
                result = minimax_h3.around_apply_rope_qk(
                    legacy, self.query, self.key, self.cache, self.positions,
                )
                self.assertIs(result[0], self.query)
                self.assertIs(result[1], self.key)


if __name__ == '__main__':
    unittest.main()
