from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


def _uses_hcu_mtp(max_top_k: Any, sampling_info: Any) -> bool:
    if not is_hcu() or max_top_k is not None:
        return False
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    return bool(
        spec.speculative_algorithm in ("EAGLE", "NEXTN")
        and spec.speculative_eagle_topk == 1
        and not spec.speculative_use_rejection_sampling
        and sampling_info.sampling_seed is None
        and not sampling_info.need_min_p_sampling
    )


def around_can_use_sparse_target_sampling(
    original: Callable[..., bool], max_top_k: Any, sampling_info: Any
) -> bool:
    if _uses_hcu_mtp(max_top_k, sampling_info):
        return True
    return original(max_top_k, sampling_info)


def around_sample_tree_target_tokens(
    original: Callable[..., torch.Tensor], *, next_token_logits: torch.Tensor,
    sampling_info: Any, batch_size: int, verify_width: int, max_top_k: Any,
) -> torch.Tensor:
    if not _uses_hcu_mtp(max_top_k, sampling_info):
        return original(
            next_token_logits=next_token_logits, sampling_info=sampling_info,
            batch_size=batch_size, verify_width=verify_width, max_top_k=max_top_k,
        )
    from lightop.sampling import top_k_top_p_sampling_from_probs

    temperatures = torch.repeat_interleave(sampling_info.temperatures, verify_width, dim=0)
    top_ks = torch.repeat_interleave(sampling_info.top_ks, verify_width, dim=0)
    top_ps = torch.repeat_interleave(sampling_info.top_ps, verify_width, dim=0)
    probs = torch.softmax(next_token_logits / temperatures, dim=-1)
    sampled = top_k_top_p_sampling_from_probs(
        probs.contiguous(), top_ks, top_ps,
        filter_apply_order="joint", deterministic=True,
    )
    return sampled.to(torch.long).reshape(batch_size, verify_width)
