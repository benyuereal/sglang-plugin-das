from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from hcu_sglang.layers.attention import flashattention_interface as F


class FlashAttentionInterfaceTests(unittest.TestCase):
    def test_flattened_prefill_uses_varlen_before_fixed_length_reshape(self):
        for batch_size, token_count in ((1, 1083), (4, 1138), (8, 1200)):
            with self.subTest(batch_size=batch_size, token_count=token_count):
                q = torch.empty(token_count, 12, 256)
                lengths = [1083] + [token_count - 1083] * (batch_size - 1)
                for index in range(1, batch_size):
                    lengths[index] = 1
                lengths[-1] = token_count - sum(lengths[:-1])
                cu_seqlens_q = torch.tensor(
                    [0] + list(torch.cumsum(torch.tensor(lengths), dim=0)),
                    dtype=torch.int32,
                )
                k_cache = torch.empty(2, 64, 12, 256)
                v_cache = torch.empty(2, 64, 12, 256)
                page_table = torch.zeros(batch_size, 2, dtype=torch.int32)
                result = object()
                with (
                    patch.object(F, "_layout_enabled", return_value=True),
                    patch(
                        "flash_attn.vllm_flash_attn_varlen_func",
                        return_value=result,
                    ) as varlen,
                    patch("flash_attn.vllm_flash_attn_with_kvcache") as fixed,
                    patch("flash_attn.flash_attn_with_kvcache") as fallback,
                ):
                    actual = F.flash_attn_with_kvcache(
                        q=q,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        page_table=page_table,
                        cu_seqlens_q=cu_seqlens_q,
                        cache_seqlens=torch.tensor(lengths, dtype=torch.int32),
                        max_seqlen_q=1083,
                    )
                self.assertIs(actual, result)
                self.assertIs(varlen.call_args.kwargs["q"], q)
                fixed.assert_not_called()
                fallback.assert_not_called()

    def test_flattened_prefill_uses_varlen_with_official_layout(self):
        q = torch.empty(1138, 12, 256)
        cu_seqlens_q = torch.tensor([0, 1083, 1138], dtype=torch.int32)
        k_cache = torch.empty(2, 64, 12, 256)
        v_cache = torch.empty(2, 64, 12, 256)
        page_table = torch.zeros(2, 2, dtype=torch.int32)
        result = object()
        with (
            patch.object(F, "_layout_enabled", return_value=False),
            patch(
                "flash_attn.vllm_flash_attn_varlen_func",
                return_value=result,
            ) as varlen,
            patch("flash_attn.vllm_flash_attn_with_kvcache") as fixed,
        ):
            actual = F.flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cu_seqlens_q=cu_seqlens_q,
                cache_seqlens=torch.tensor([1083, 55], dtype=torch.int32),
                max_seqlen_q=1083,
            )
        self.assertIs(actual, result)
        self.assertIs(varlen.call_args.kwargs["q"], q)
        self.assertEqual(varlen.call_args.kwargs["max_seqlen_k"], 2 * 64)
        # The official pool is [NP, PS, H, D]: the varlen entry must
        # be told bshd, not the HCU-only bhsd page layout.
        self.assertEqual(varlen.call_args.kwargs["layout"], "bshd")
        fixed.assert_not_called()

    def test_flattened_prefill_declares_bhsd_with_hcu_layout(self):
        q = torch.empty(1138, 12, 256)
        cu_seqlens_q = torch.tensor([0, 1083, 1138], dtype=torch.int32)
        k_cache = torch.empty(2, 12, 64, 256)
        v_cache = torch.empty(2, 12, 64, 256)
        page_table = torch.zeros(2, 2, dtype=torch.int32)
        result = object()
        with (
            patch.object(F, "_layout_enabled", return_value=True),
            patch(
                "flash_attn.vllm_flash_attn_varlen_func",
                return_value=result,
            ) as varlen,
            patch("flash_attn.vllm_flash_attn_with_kvcache") as fixed,
        ):
            actual = F.flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                page_table=page_table,
                cu_seqlens_q=cu_seqlens_q,
                cache_seqlens=torch.tensor([1083, 55], dtype=torch.int32),
                max_seqlen_q=1083,
            )
        self.assertIs(actual, result)
        # [NP, H, PS, D] pages carry bhsd; page_size is axis 2 here.
        self.assertEqual(varlen.call_args.kwargs["layout"], "bhsd")
        fixed.assert_not_called()

    def test_decode_retains_single_token_reshape(self):
        q = torch.empty(4, 12, 256)
        result = object()
        with (
            patch.object(F, "_layout_enabled", return_value=True),
            patch(
                "flash_attn.vllm_flash_attn_with_kvcache", return_value=result
            ) as fixed,
        ):
            actual = F.flash_attn_with_kvcache(
                q=q,
                k_cache=torch.empty(2, 64, 12, 256),
                v_cache=torch.empty(2, 64, 12, 256),
                page_table=torch.zeros(4, 2, dtype=torch.int32),
                cache_seqlens=torch.ones(4, dtype=torch.int32),
                max_seqlen_q=1,
            )
        self.assertIs(actual, result)
        self.assertEqual(fixed.call_args.kwargs["q"].shape, (4, 1, 12, 256))


if __name__ == "__main__":
    unittest.main()
