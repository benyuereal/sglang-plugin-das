import os
import pytest
import torch

from hcu_sglang.layers.attention.fla import linear_attention as H
from hcu_sglang.layers.attention.fla import gdn_verify_source as S
from sglang.srt.layers.attention.linear.kernels import gdn_triton as G
from sglang.srt.plugins.hook_registry import HookRegistry


@pytest.fixture
def consumer(monkeypatch):
    from hcu_sglang.utils import detection

    monkeypatch.setattr(detection, "is_hcu", lambda: True)
    monkeypatch.setattr(S, "_use_decode_aiter_linear_attn", True)
    original = G.TritonGDNKernel.target_verify
    records = []
    monkeypatch.setattr(HookRegistry, "register", lambda *args: records.append(args))
    H.register()
    target = "sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel.target_verify"
    for name, hook, kind in records:
        if name == target:
            HookRegistry._apply_target(name, [(kind, hook, None)])
    try:
        yield G.TritonGDNKernel.__new__(G.TritonGDNKernel).target_verify
    finally:
        G.TritonGDNKernel.target_verify = original


def test_registered_verify_uses_vendor_triton_and_preserves_state_contract(monkeypatch, consumer):
    import aiter
    from aiter.ops.triton.fla import fused_sigmoid_gating_recurrent as vendor

    calls = []
    marker = object()
    monkeypatch.setattr(vendor, "fused_sigmoid_gating_delta_rule_update",
        lambda **kwargs: calls.append(kwargs) or marker)
    monkeypatch.setattr(aiter, "sglang_fused_sigmoid_gating_delta_rule_update",
        lambda **kwargs: pytest.fail("obsolete C++ verify route"))
    names = ("A_log", "dt_bias", "q", "k", "v", "a", "b", "ssm_states", "cache_indices",
        "query_start_loc", "intermediate_states_buffer", "intermediate_state_indices", "retrieve_parent_token")
    args = {name: object() for name in names}
    assert consumer(**args, cache_steps=4) is marker
    call, = calls
    for src, dst in (("ssm_states", "initial_state_source"), ("cache_indices", "initial_state_indices"),
            ("query_start_loc", "cu_seqlens")):
        assert call[dst] is args[src]
    for name in ("q", "k", "v", "a", "b", "intermediate_states_buffer", "intermediate_state_indices", "retrieve_parent_token"):
        assert call[name] is args[name]
    assert call["disable_state_update"] is True
    assert call["use_qk_l2norm_in_kernel"] is True
    assert call["cache_steps"] == 4


def math_reference(args):
    q, k, v = (args[name].float() for name in ("q", "k", "v"))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q, k = (x.repeat_interleave(v.shape[2] // x.shape[2], dim=2) for x in (q, k))
    pool = args["ssm_states"]
    cached = args["intermediate_states_buffer"].clone()
    output = torch.empty_like(v)
    offsets = args["query_start_loc"].tolist()
    parents = args["retrieve_parent_token"]
    for seq, (start, end) in enumerate(zip(offsets, offsets[1:])):
        slot = int(args["cache_indices"][seq])
        dst = int(args["intermediate_state_indices"][seq])
        state = pool[slot].float().clone() if slot >= 0 else torch.zeros_like(pool[0]).float()
        for step, token in enumerate(range(start, end)):
            if parents is not None and step:
                state = cached[dst, int(parents[seq, step])].float()
            decay = torch.exp(-args["A_log"].float().exp() * torch.nn.functional.softplus(
                args["a"][0, token].float() + args["dt_bias"].float()))
            state = state * decay[:, None, None]
            residual = v[0, token] - (state * k[0, token, :, None, :]).sum(-1)
            residual *= args["b"][0, token].float().sigmoid()[:, None]
            state = state + residual[:, :, None] * k[0, token, :, None, :]
            output[0, token] = (state * q[0, token, :, None, :]).sum(-1) * q.shape[-1]**-0.5
            if dst >= 0:
                cached[dst, step] = state.to(cached.dtype)
    return output.to(args["v"].dtype), cached


@pytest.mark.skipif(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(), reason="HCU GPU lane required")
@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [4, 6])
def test_registered_verify_math_state_and_live_graph(consumer, tree, state_dtype, width):
    torch.manual_seed(566)
    device = "cuda"
    heads, value_heads, dim, tokens, slots = 2, 4, 128, 2 * width, 5
    args = {name: torch.randn(1, tokens, heads, dim, device=device, dtype=torch.bfloat16)
        for name in ("q", "k")}
    args["v"] = torch.randn(1, tokens, value_heads, dim, device=device, dtype=torch.bfloat16)
    for name in ("a", "b"):
        args[name] = torch.randn(1, tokens, value_heads, device=device, dtype=torch.bfloat16)
    args.update(A_log=torch.randn(value_heads, device=device), dt_bias=torch.randn(value_heads, device=device),
        ssm_states=torch.randn(slots, value_heads, dim, dim, device=device, dtype=state_dtype),
        cache_indices=torch.tensor([1, 3], device=device, dtype=torch.int32),
        query_start_loc=torch.tensor([0, width, tokens], device=device, dtype=torch.int32),
        intermediate_states_buffer=torch.full((slots, width, value_heads, dim, dim), 7., device=device, dtype=state_dtype),
        intermediate_state_indices=torch.tensor([2, 4], device=device, dtype=torch.int32),
        cache_steps=width, retrieve_parent_token=(torch.tensor([[0, 0, 0, 2, 1, 4][:width], [0, 0, 1, 0, 2, 4][:width]],
            device=device, dtype=torch.int32) if tree else None))
    initial = args["ssm_states"].clone()
    for _ in range(3):
        consumer(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = consumer(**args)
    for iteration in range(2):
        for name in ("q", "k", "v", "a", "b"):
            args[name].copy_(torch.randn_like(args[name]))
        args["cache_indices"].copy_(torch.tensor([3, -1], device=device, dtype=torch.int32))
        args["intermediate_state_indices"].copy_(torch.tensor([4, 2 if tree else -1], device=device, dtype=torch.int32))
        args["query_start_loc"].copy_(torch.tensor([0, width - 1, tokens], device=device, dtype=torch.int32))
        if tree:
            args["query_start_loc"].copy_(torch.tensor([0, width, tokens], device=device, dtype=torch.int32))
            args["retrieve_parent_token"].copy_(torch.tensor([[0, 0, 1, 1, 3, 2][:width], [0, 0, 0, 2, 1, 4][:width]], device=device, dtype=torch.int32))
        args["intermediate_states_buffer"].fill_(7.)
        expected, cached = math_reference(args)
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0.002, rtol=0.02)
        torch.testing.assert_close(args["intermediate_states_buffer"], cached, atol=0.02, rtol=0.02)
        torch.testing.assert_close(args["ssm_states"], initial, atol=0, rtol=0)
