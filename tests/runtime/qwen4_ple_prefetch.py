import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from hcu_sglang.models.qwen import ple, qwen4_exp as Q
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
        if target in (
            "sglang.srt.models.qwen4_exp.Qwen4ExpPLELayer.start_prefetch",
            "sglang.srt.models.qwen4_exp.Qwen4ExpPLELayer._consume_prefetched_embeddings",
        ):
            name = target.rsplit(".", 1)[1]
            saved.append((name, getattr(N.Qwen4ExpPLELayer, name)))
            HookRegistry._apply_target(target, [(kind, fn, None)])
    yield
    for name, original in reversed(saved):
        setattr(N.Qwen4ExpPLELayer, name, original)


class Embedding(torch.nn.Module):
    def __init__(self, provider, ids):
        super().__init__()
        self.ngram_embedding = provider
        self.ids = ids
        self.ngram_size = 2
        self.ngram_heads = 2
        self.gather_dp_tokens = True

    def compute_ngram_ids(self, batch):
        return self.ids

    def _prepare_embedding_lookup(self, ids, forward_batch, physical_tokens):
        return ids, max(0, physical_tokens - 1)

    def _uses_per_row_scale(self):
        return True

    def _finish_embedding_lookup(self, x, semantic, forward_batch, physical):
        assert (semantic, physical) == (max(0, self.ids.shape[0] - 1), self.ids.shape[0])
        return x


def make_layer(monkeypatch, provider, ids, stream):
    embedding = Embedding(provider, ids)
    monkeypatch.setattr(N, "Qwen4ExpNGramEmbedding", lambda *a, **k: embedding)
    monkeypatch.setattr(N, "ReplicatedLinear", lambda *a, **k: torch.nn.Identity())
    monkeypatch.setattr(N, "Qwen4ExpPLEGroupedNorm", lambda *a, **k: torch.nn.Identity())
    config = SimpleNamespace(hidden_size=4, ple_embed_dim=16, ple_conv_kernel_size=2,
        hc_count=1, rms_norm_eps=1e-6, ple_offload_embedding=True)
    with monkeypatch.context() as local:
        local.setattr(torch.cuda, "Stream", lambda: stream)
        return N.Qwen4ExpPLELayer(config)


def test_public_prefetch_stays_on_current_stream_and_consumes_once(registered, monkeypatch):
    ids = torch.tensor([[1, 4], [3, 2], [7, 6]])
    calls = []
    provider = SimpleNamespace(
        allocate_output=lambda shape, device: torch.empty(shape, device=device),
        _maybe_register_uva=lambda: calls.append("register"),
        gather=lambda ids, out: (calls.append("gather"), out.copy_(ids[..., None].expand_as(out))),
        reduce=lambda x: x,
    )
    stream = Mock()
    stream.wait_stream.side_effect = AssertionError("HCU PLE gather used the side stream")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: Mock())
    layer = make_layer(monkeypatch, provider, ids, stream)
    batch = SimpleNamespace(physical_tokens=3)
    forward = SimpleNamespace(input_ids=torch.zeros(3, dtype=torch.long))
    layer.start_prefetch(batch, forward)
    assert calls == ["register", "gather"]
    with pytest.raises(RuntimeError, match="not consumed"):
        layer.start_prefetch(batch, forward)
    output = layer._consume_prefetched_embeddings(forward)
    torch.testing.assert_close(output.view(3, 2, 8), ids[..., None].expand(3, 2, 8).float())
    assert layer._prefetch_state is None
    with pytest.raises(RuntimeError, match="missing"):
        layer._consume_prefetched_embeddings(forward)
    layer.start_prefetch(None, forward)
    assert layer._prefetch_state[1:] == (2, 3)
    assert layer._prefetch_state[0].count_nonzero() == 0


def test_public_prefetch_empty_disabled_and_effective_fallback(registered, monkeypatch):
    ids = torch.empty(0, 2, dtype=torch.long)
    provider = SimpleNamespace(allocate_output=Mock(side_effect=AssertionError("empty allocation")))
    layer = make_layer(monkeypatch, provider, ids, Mock())
    forward = SimpleNamespace(input_ids=torch.empty(0, dtype=torch.long))
    layer.start_prefetch(SimpleNamespace(physical_tokens=0), forward)
    assert layer._prefetch_state is None
    layer.ple_embedding.gather_dp_tokens = False
    layer.start_prefetch(None, forward)
    layer._prefetch_stream = None
    layer.start_prefetch(object(), forward)
    if hasattr(Q, "around_start_prefetch"):
        original = Mock(return_value="effective baseline")
        monkeypatch.setattr(Q, "is_hcu", lambda: False)
        assert Q.around_start_prefetch(original, layer, None, forward) == "effective baseline"
        original.assert_called_once_with(layer, None, forward)


@pytest.mark.skipif(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(), reason="HCU GPU lane required")
@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16])
def test_public_prefetch_mapped_provider_live_graph(registered, monkeypatch, dtype):
    source = SimpleNamespace(**{name: None for name in ple.Qwen4ExpPinnedHostEmbedding._COPIED_ATTRIBUTES})
    source.__dict__.update(tp_size=1, num_added_embeddings=0, embedding_dim=8,
        shard_indices=SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=16),
        quant_method=ple.UnquantizedEmbeddingMethod.__new__(ple.UnquantizedEmbeddingMethod),
        weight=torch.nn.Parameter(torch.arange(128).reshape(16, 8).to(dtype).pin_memory(), requires_grad=False),
        weight_scale=torch.full((16, 1), 0.25, dtype=torch.bfloat16).pin_memory())
    monkeypatch.setattr(ple, "_ple_etp_reduce_scatter_enabled", lambda *a: False)
    checkpoint = source.weight.detach().clone()
    provider = ple.Qwen4ExpPinnedHostEmbedding(source)
    provider.weight.data.copy_(checkpoint)
    assert provider._device_weight_view is not None
    ids = torch.tensor([[1, 4], [3, 2], [7, 6]], device="cuda")
    stream = torch.cuda.Stream()
    layer = make_layer(monkeypatch, provider, ids, stream)
    batch = SimpleNamespace(physical_tokens=3)
    forward = SimpleNamespace(input_ids=torch.zeros(3, device="cuda", dtype=torch.long))
    def run():
        layer.start_prefetch(batch, forward)
        return layer._consume_prefetched_embeddings(forward)
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for selected in ([[9, 2], [15, 0], [5, 8]], [[13, 12], [11, 10], [7, 1]]):
        ids.copy_(torch.tensor(selected, device="cuda"))
        graph.replay()
        expected = checkpoint[torch.tensor(selected)].to(torch.bfloat16) * 0.25
        torch.testing.assert_close(output.view(3, 2, 8).cpu(), expected, rtol=0, atol=0)
    assert layer._prefetch_state is None
