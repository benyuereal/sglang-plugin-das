"""Route-preserving adapter for the gfx936 Qwen W4A8 EP moe_c path."""

import torch
import triton


def moe_c_native_sort(topk_ids, block_size, num_experts, expert_map):
    """Return moe_c flat route IDs, including every remote/dummy route.

    This opt-in specialization expects unique real experts per token and one
    final EP drop expert. Native sorting coalesces repeated token/expert pairs;
    distinct per-route dummy experts prevent it from dropping remote routes.
    The downstream moe_c GEMMs still consume the original topk weights/IDs.
    """
    import aiter

    tokens, topk = topk_ids.shape
    drop_id = num_experts - 1
    native_experts = drop_id + topk
    route = torch.arange(topk, dtype=torch.int32, device=topk_ids.device)[None, :]
    expanded = torch.where(topk_ids == drop_id, drop_id + route, topk_ids)
    # Match the native ABI's documented upper bound, including all padding.
    capacity = topk_ids.numel() + native_experts * block_size - topk
    # The converter visits the entire capacity, although sorting only promises
    # valid data through count. Encode padding with token_id == tokens so the
    # complete tail converts to the same flat padding ID as the old path.
    ck_ids = torch.full(
        (capacity,), tokens, dtype=torch.int32, device=topk_ids.device
    )
    flat_ids = torch.empty_like(ck_ids)
    sort_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    sorted_weights = torch.empty(capacity, dtype=torch.float32, device=topk_ids.device)
    experts = torch.zeros(
        triton.cdiv(capacity, block_size), dtype=torch.int32, device=topk_ids.device
    )
    positions = torch.empty(
        2 * native_experts, dtype=torch.int32, device=topk_ids.device
    )
    count = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    aiter.moe_sorting_fwd(
        expanded, sort_weights, ck_ids, sorted_weights, experts, positions,
        count, None, native_experts, block_size, None,
    )
    aiter.moe_sorting_ck_ids_to_moe_c(ck_ids, flat_ids, tokens, topk)
    # Padding/tail experts are initialized to zero. All expanded dummy experts
    # map through the original drop entry, preserving its -1 GEMM behavior.
    experts = expert_map[experts.clamp_max(drop_id)]
    return flat_ids, experts, count
