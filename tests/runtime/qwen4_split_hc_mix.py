import os
from functools import partial

import pytest
import torch

from hcu_sglang.environ import register_envs
from hcu_sglang.models.qwen import hc_mix, qwen4_exp as Q
from sglang.srt.layers import hyperconnection as H
from sglang.srt.plugins.hook_registry import HookRegistry


@pytest.fixture
def registered(monkeypatch):
    register_envs()
    records = []
    monkeypatch.setattr(Q, "is_hcu", lambda: True)
    monkeypatch.setattr(HookRegistry, "register", lambda *args: records.append(args))
    Q.register_hooks()
    target = "sglang.srt.layers.hyperconnection.fused_hc_mix"
    hook = next(record for record in records if record[0] == target)
    original = H.fused_hc_mix
    monkeypatch.setattr(H, "fused_hc_mix", original)
    HookRegistry._apply_target(target, [(hook[2], hook[1], None)])
    return original


def test_disabled_public_alias_preserves_effective_original(registered, monkeypatch):
    monkeypatch.setenv("SGLANG_QWEN4_SPLIT_HC_MIX", "0")
    value = object()
    seen = []
    def effective(*args):
        seen.append(args)
        return value
    x, down, up = object(), object(), object()
    assert hc_mix.around_fused_hc_mix(effective, x, down, up, 4, 2560) is value
    assert seen == [(x, down, up, 4, 2560)]


def test_enabled_cpu_input_preserves_effective_original(registered, monkeypatch):
    monkeypatch.setenv("SGLANG_QWEN4_SPLIT_HC_MIX", "1")
    x = torch.ones(1, 10240, dtype=torch.bfloat16)
    down, up = object(), object()
    value = object()
    seen = []
    def effective(*args):
        seen.append(args)
        return value
    assert hc_mix.around_fused_hc_mix(effective, x, down, up, 4, 2560) is value
    assert seen == [(x, down, up, 4, 2560)]


GPU = os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1" and torch.cuda.is_available()


@pytest.mark.skipif(not GPU, reason="HCU GPU lane required")
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_public_mix_changed_inputs_and_weights_graph(registered, monkeypatch, rows):
    from hcu_sglang import jit_overlay_loader as overlay
    from sglang.kernels.jit.utils.compile import paths
    from sglang.kernels.ops.layernorm import grouped_gemma_rmsnorm as grouped

    monkeypatch.setattr(paths, "DEFAULT_INCLUDE", paths.DEFAULT_INCLUDE.copy())
    overlay.prefer_overlay_headers()
    monkeypatch.setattr(grouped, "load_jit", partial(overlay.around_load_jit, grouped.load_jit))
    monkeypatch.setenv("SGLANG_QWEN4_SPLIT_HC_MIX", "1")
    # The split kernel is gfx936-only (around_fused_hc_mix gates on it),
    # so on the gfx938 re-verification nodes this case skips instead of
    # failing the lane.
    arch = hc_mix.get_gcn_arch()
    if not arch.startswith("gfx936"):
        pytest.skip(f"split hc_mix is gfx936-only; node arch={arch or 'unknown'}")
    torch.manual_seed(541 + rows)
    mixer = H.GatedResidual(H.HyperConnectionConfig(hc_count=4, hidden_size=2560,
        hc_lowrank=320, params_dtype=torch.bfloat16, hc_per_branch_norm=True)).cuda()
    mixer = mixer.to(dtype=torch.bfloat16)
    mixer._jit_mix_ok = False
    mixer.hc_norm.weight.data.uniform_(-0.1, 0.1)
    down = mixer.input_mix_weight_down.weight
    up = mixer.input_mix_weight_up.weight
    down.data.normal_(std=0.01)
    up.data.normal_(std=0.03)
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    calls = []
    split = hc_mix.split_mix
    def tracked(*args, **kwargs):
        calls.append(args[0].shape)
        return split(*args, **kwargs)
    monkeypatch.setattr(hc_mix, "split_mix", tracked)
    with torch.no_grad():
        for _ in range(3):
            mixer.mix(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, state = mixer.mix(x)
        assert calls
        assert state[0].data_ptr() == x.data_ptr()
        for i in range(4):
            x.copy_(torch.randn_like(x) * (0.2 + 0.5 * i))
            down.data.copy_(torch.randn_like(down) * (0.01 + 0.01 * i))
            up.data.copy_(torch.randn_like(up) * (0.02 + 0.01 * i))
            graph.replay()
            normalized = mixer.hc_norm(x)
            expected = registered(normalized, down, up, 4, 2560)
            torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.004)


@pytest.mark.skipif(not GPU, reason="HCU GPU lane required")
@pytest.mark.parametrize("variant", ["dtype", "rank", "rows", "architecture"])
def test_public_alias_unsupported_shapes_delegate(registered, monkeypatch, variant):
    monkeypatch.setenv("SGLANG_QWEN4_SPLIT_HC_MIX", "1")
    rows, rank, dtype = (0 if variant == "rows" else 4), (64 if variant == "rank" else 320), (torch.float16 if variant == "dtype" else torch.bfloat16)
    x = torch.randn(rows, 10240, device="cuda", dtype=dtype)
    down = torch.randn(rank, 10240, device="cuda", dtype=dtype) * 0.01
    up = torch.randn(10240, rank, device="cuda", dtype=dtype) * 0.01
    if variant == "architecture":
        monkeypatch.setattr(hc_mix, "get_gcn_arch", lambda: "gfx938")
    def refused(*args, **kwargs):
        raise AssertionError("Unsupported split HC path was selected")
    monkeypatch.setattr(hc_mix, "split_mix", refused)
    actual = H.fused_hc_mix(x, down, up, 4, 2560)
    expected = registered(x, down, up, 4, 2560)
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.004)
