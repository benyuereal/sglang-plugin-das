# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import torch
from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func_interface
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn_2 import (
    FlashAttention2Backend, FlashAttention2Impl,
)


class HcuFlashAttention2Backend(FlashAttention2Backend):
    @staticmethod
    def get_impl_cls():
        return HcuFlashAttention2Impl


class HcuFlashAttention2Impl(FlashAttention2Impl):
    def __init__(self, *args, **kwargs):
        self.dropout_p = float(kwargs.get("dropout_p", 0.0))
        super().__init__(*args, **kwargs)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        del cu_seqlens_host
        cu_seqlens = cu_seqlens.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
        output = flash_attn_varlen_func_interface(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return output[0] if isinstance(output, tuple) else output
