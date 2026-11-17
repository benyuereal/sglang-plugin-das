import pytest
import torch

from hcu_sglang.layers.moe.moe_c_native_sort import moe_c_native_sort


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="DTK required")
@pytest.mark.parametrize("tokens", [1, 4, 17, 129, 1025])
@pytest.mark.parametrize("block", [16, 32, 64, 128])
def test_all_routes_including_repeated_remote_routes_are_preserved(tokens, block):
    torch.manual_seed(tokens)
    ids = torch.stack([torch.randperm(128, device="cuda")[:10] for _ in range(tokens)]).to(torch.int32)
    ids[:, 3:] = 128
    expert_map = torch.arange(129, dtype=torch.int32, device="cuda")
    expert_map[-1] = -1
    flat, experts, count = moe_c_native_sort(ids, block, 129, expert_map)
    n = count.item()
    routes = flat[:n].cpu()
    mask = routes < tokens * 10
    assert torch.equal(routes[mask].sort().values, torch.arange(tokens * 10))
    expected = ids.flatten().cpu()[routes[mask]].long()
    expected[expected == 128] = -1
    assert torch.equal(experts.repeat_interleave(block)[:n].cpu()[mask], expected)
