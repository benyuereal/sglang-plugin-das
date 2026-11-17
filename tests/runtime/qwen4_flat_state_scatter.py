import os

import pytest
import torch

from hcu_sglang.environ import register_envs
from hcu_sglang.models.qwen import qwen4_exp as Q, state_scatter as S
from sglang.srt.layers.attention import hybrid_linear_attn_backend as H
from sglang.srt.plugins.hook_registry import HookRegistry


@pytest.fixture
def registered(monkeypatch):
    register_envs()
    records = []
    monkeypatch.setattr(Q, "is_hcu", lambda: True)
    monkeypatch.setattr(HookRegistry, "register", lambda *args: records.append(args))
    Q.register_hooks()
    target = "sglang.srt.layers.attention.hybrid_linear_attn_backend.fused_mamba_state_scatter_with_mask"
    hook = next(record for record in records if record[0] == target)
    original = H.fused_mamba_state_scatter_with_mask
    monkeypatch.setattr(H, "fused_mamba_state_scatter_with_mask", original)
    HookRegistry._apply_target(target, [(hook[2], hook[1], None)])
    return original


def test_disabled_keeps_effective_original(registered, monkeypatch):
    monkeypatch.setenv("SGLANG_QWEN4_FLAT_STATE_SCATTER", "0")
    values = [object() for _ in range(5)]
    seen = []
    result = object()
    def effective(*args):
        seen.append(args)
        return result
    assert S.around_state_scatter(effective, *values) is result
    assert seen == [tuple(values)]


def test_cpu_keeps_effective_original(registered, monkeypatch):
    monkeypatch.setenv("SGLANG_QWEN4_FLAT_STATE_SCATTER", "1")
    dst = torch.ones(1)
    values = [dst] + [object() for _ in range(4)]
    seen = []
    def effective(*args):
        seen.append(args)
        return "baseline"
    assert S.around_state_scatter(effective, *values) == "baseline"
    assert seen == [tuple(values)]


GPU = os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1" and torch.cuda.is_available()


@pytest.mark.skipif(not GPU, reason="HCU GPU lane required")
@pytest.mark.parametrize("requests", [1, 4, 16])
def test_public_backend_remapped_rows_changing_graph(registered, monkeypatch, requests):
    monkeypatch.setenv("SGLANG_QWEN4_FLAT_STATE_SCATTER", "1")
    src = torch.randn(36, 2, 4, 48, 128, 128, device="cuda", dtype=torch.float32)
    dst = torch.randn(36, 4, 48, 128, 128, device="cuda", dtype=torch.float32)
    expected = dst.clone()
    dst_ids = torch.full((requests,), -1, device="cuda", dtype=torch.int64)
    steps = torch.full_like(dst_ids, -1)
    src_ids = torch.zeros_like(dst_ids)
    calls = []
    flat = S.flat_scatter
    def tracked(*args):
        calls.append(args[0].shape)
        return flat(*args)
    monkeypatch.setattr(S, "flat_scatter", tracked)
    scatter = H.HybridLinearAttnBackend._scatter_speculative_state_with_mask
    for _ in range(3):
        scatter(dst, src, dst_ids, steps, src_ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scatter(dst, src, dst_ids, steps, src_ids)
    assert calls
    for i in range(4):
        src.add_(0.125)
        dst.fill_(-31 - i)
        expected.copy_(dst)
        dst_ids.fill_(-1)
        steps.fill_(-1)
        src_ids.zero_()
        dst_ids[0] = i % 4
        steps[0] = i
        src_ids[0] = i % 2
        if requests >= 4:
            dst_ids[1:4] = torch.tensor([7, 2, 3], device="cuda")
            steps[1:4] = torch.tensor([1, 4, 2], device="cuda")
            src_ids[1:4] = torch.tensor([0, 1, -1], device="cuda")
        graph.replay()
        expected[:, i % 4].copy_(src[:, i % 2, i])
        torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@pytest.mark.skipif(not GPU, reason="HCU GPU lane required")
@pytest.mark.parametrize("variant", ["shape", "dtype", "empty", "architecture"])
def test_unsupported_delegates(registered, monkeypatch, variant):
    monkeypatch.setenv("SGLANG_QWEN4_FLAT_STATE_SCATTER", "1")
    dtype = torch.bfloat16 if variant == "dtype" else torch.float32
    dst = torch.ones(1, 2, 8, device="cuda", dtype=dtype)
    src = torch.ones(1, 2, 4, 8, device="cuda", dtype=dtype)
    ids = torch.zeros(0 if variant == "empty" else 1, device="cuda", dtype=torch.int32)
    if variant == "architecture":
        monkeypatch.setattr(S, "get_gcn_arch", lambda: "gfx938")
    seen = []
    def effective(*args):
        seen.append(args)
        return "baseline"
    assert S.around_state_scatter(effective, dst, src, ids, ids) == "baseline"
    assert seen == [(dst, src, ids, ids, None)]
