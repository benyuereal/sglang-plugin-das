import os
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from hcu_sglang.models.qwen import ple, qwen4_exp as Q
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.plugins.hook_registry import HookRegistry

N = ple.N


@pytest.fixture
def registered(monkeypatch):
    records = []
    monkeypatch.setattr(Q, "is_hcu", lambda: True)
    monkeypatch.setattr(HookRegistry, "register", lambda *args: records.append(args))
    Q.register_hooks()
    saved = []
    for target, fn, kind in records:
        if target in ("sglang.srt.models.qwen4_exp.Qwen4ExpModel.forward",
                      "sglang.srt.models.qwen4_exp._MixStreams.__call__",
                      "sglang.srt.models.qwen4_exp.Qwen4ExpForConditionalGeneration.forward"):
            name, attr = target.rsplit(".", 2)[1:]
            cls = getattr(N, name)
            saved.append((cls, attr, getattr(cls, attr)))
            HookRegistry._apply_target(target, [(kind, fn, None)])
    yield
    for cls, attr, original in reversed(saved):
        setattr(cls, attr, original)


def model_with_mixer(mixer):
    model = N.Qwen4ExpModel.__new__(N.Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model._start_layer = model._end_layer = 0
    model.layers = torch.nn.ModuleList()
    model.has_ple = False
    model.hyper_connection_mixer = mixer
    return model


class Mixer(torch.nn.Module):
    def mix(self, x, normalized_input=None):
        result = x.view(x.shape[0], 4, -1).mean(1)
        x.add_(7)
        return result, (x, x, None)


def test_public_logits_handoff_does_not_borrow_model_cache(registered, monkeypatch):
    model = N.Qwen4ExpForConditionalGeneration.__new__(N.Qwen4ExpForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    raw = torch.arange(32.).reshape(2, 16)
    expected = raw.clone()
    model.model.last_hc_hidden_states = raw
    output = N.LogitsProcessorOutput(next_token_logits=torch.zeros(2, 8))
    monkeypatch.setattr(N.Qwen3VLForConditionalGeneration, "forward", lambda *a, **k: output)
    result = model(None, None, None)
    assert result is output
    torch.testing.assert_close(result.hidden_states, expected)
    assert result.hidden_states.data_ptr() != raw.data_ptr()
    raw.zero_()
    torch.testing.assert_close(result.hidden_states, expected)


def test_logits_snapshot_preserves_non_hcu_and_non_logits_outputs(registered, monkeypatch):
    result = N.LogitsProcessorOutput(next_token_logits=torch.zeros(1, 8), hidden_states=torch.ones(1, 16))
    raw = result.hidden_states
    monkeypatch.setattr(Q, "is_hcu", lambda: False)
    assert Q.after_conditional_forward(result, None) is result
    assert result.hidden_states is raw
    monkeypatch.setattr(Q, "is_hcu", lambda: True)
    other = object()
    assert Q.after_conditional_forward(other, None) is other
    model = SimpleNamespace(model=SimpleNamespace(last_hc_hidden_states=None))
    assert Q.after_conditional_forward(result, model) is result
    assert result.hidden_states is raw


@pytest.mark.skipif(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(), reason="HCU GPU lane required")
def test_public_logits_handoff_live_graph(registered, monkeypatch):
    model = N.Qwen4ExpForConditionalGeneration.__new__(N.Qwen4ExpForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    raw = torch.randn(4, 10240, device="cuda", dtype=torch.bfloat16)
    model.model.last_hc_hidden_states = raw
    output = N.LogitsProcessorOutput(next_token_logits=torch.zeros(4, 8, device="cuda"))
    monkeypatch.setattr(N.Qwen3VLForConditionalGeneration, "forward", lambda *a, **k: output)
    for _ in range(3):
        model(None, None, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = model(None, None, None)
        hc = result.hidden_states
    assert hc.data_ptr() != raw.data_ptr()
    for _ in range(2):
        raw.copy_(torch.randn_like(raw))
        expected = raw.clone()
        graph.replay()
        torch.testing.assert_close(hc, expected, rtol=0, atol=0)
        raw.zero_()
        torch.testing.assert_close(hc, expected, rtol=0, atol=0)


def test_public_model_returns_owned_pre_mix_hc_to_mtp(registered):
    raw = torch.arange(32.).reshape(2, 16)
    expected = raw.clone()
    model = model_with_mixer(Mixer())
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    output, hc = model(torch.zeros(2, dtype=torch.long), torch.arange(2), batch, inputs_embeds=raw)
    torch.testing.assert_close(output, expected.view(2, 4, 4).mean(1))
    torch.testing.assert_close(hc, expected)
    assert hc.data_ptr() != raw.data_ptr()
    raw.fill_(999)
    torch.testing.assert_close(hc, expected)
    assert batch.residual_stream is None


def test_snapshot_scope_clears_after_failure_and_preserves_effective_fallback(registered, monkeypatch):
    model = model_with_mixer(Mixer())
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    def failing(*args, **kwargs):
        raise ValueError("terminal failure")
    with pytest.raises(ValueError, match="terminal failure"):
        Q.around_model_forward(failing, model, None, None, batch)
    assert Q._hc_snapshot.get() is None
    monkeypatch.setattr(Q, "is_hcu", lambda: False)
    assert Q.around_model_forward(lambda *a, **k: "effective baseline", model, None, None, batch) == "effective baseline"


@pytest.mark.parametrize("mode,last", [(ForwardMode.IDLE, True), (ForwardMode.DECODE, False)])
def test_idle_and_nonterminal_pp_preserve_effective_fallback(registered, mode, last):
    model = model_with_mixer(Mixer())
    model.pp_group.is_last_rank = last
    batch = SimpleNamespace(forward_mode=mode)
    result = object()
    def effective(*args, **kwargs):
        assert Q._hc_snapshot.get() is None
        return result
    assert Q.around_model_forward(effective, model, None, None, batch) is result


def test_nested_models_keep_independent_hc_snapshots(registered):
    first, second = (model_with_mixer(Mixer()) for _ in range(2))
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    raw = torch.arange(32.).reshape(2, 16)
    expected = raw.clone()
    ids, positions = torch.zeros(2, dtype=torch.long), torch.arange(2)
    def effective(model, *args, **kwargs):
        state = Q._hc_snapshot.get()
        _, other_hc = second(ids, positions, batch, inputs_embeds=raw.clone())
        torch.testing.assert_close(other_hc, expected)
        assert Q._hc_snapshot.get() is state
        mixed = N._MixStreams(model.hyper_connection_mixer)(raw)
        return mixed, raw
    _, hc = Q.around_model_forward(effective, first, ids, positions, batch)
    torch.testing.assert_close(hc, expected)
    assert Q._hc_snapshot.get() is None


@pytest.mark.skipif(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(), reason="HCU GPU lane required")
@pytest.mark.parametrize("rows", [1, 4])
def test_public_model_hc_snapshot_live_graph(registered, monkeypatch, rows):
    from hcu_sglang import jit_overlay_loader as overlay
    from sglang.kernels.jit.utils.compile import paths
    from sglang.kernels.ops.layernorm import grouped_gemma_rmsnorm as grouped

    monkeypatch.setattr(paths, "DEFAULT_INCLUDE", paths.DEFAULT_INCLUDE.copy())
    overlay.prefer_overlay_headers()
    monkeypatch.setattr(grouped, "load_jit", partial(overlay.around_load_jit, grouped.load_jit))
    config = N.HyperConnectionConfig(hc_count=4, hidden_size=2560, hc_lowrank=16,
        params_dtype=torch.bfloat16, hc_per_branch_norm=True)
    mixer = N.GatedResidual(config, use_combine=False).to(device="cuda", dtype=torch.bfloat16)
    model = model_with_mixer(mixer)
    raw = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros(rows, device="cuda", dtype=torch.long)
    positions = torch.arange(rows, device="cuda")
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    for _ in range(3):
        model(ids, positions, batch, inputs_embeds=raw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, hc = model(ids, positions, batch, inputs_embeds=raw)
    assert hc.data_ptr() != raw.data_ptr()
    for _ in range(2):
        raw.copy_(torch.randn_like(raw))
        expected = raw.clone()
        eager = mixer.mix(raw.clone())[0]
        graph.replay()
        torch.testing.assert_close(hc, expected, atol=0, rtol=0)
        torch.testing.assert_close(output, eager, atol=0.004, rtol=0.02)
        raw.fill_(0)
        torch.testing.assert_close(hc, expected, atol=0, rtol=0)
