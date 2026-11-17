from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from hcu_sglang.models.qwen import fla_prefill as F


def test_idle_metadata_and_unhinted_functions_preserve_current_fallback():
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda:False))
    assert F.after_gdn_metadata("result", None, batch) == "result"
    cu = torch.tensor([0, 3])
    for wrapper in (F.around_indices, F.around_offsets):
        assert wrapper(lambda a,b:(a,b), cu, 4) == (cu, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_host_lengths_match_official_math_without_tensor_readback():
    from sglang.kernels.ops.attention.fla import index as I
    cu = torch.tensor([0, 17, 17, 150], dtype=torch.int32, device="cuda")
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda:True),
        extend_seq_lens_cpu=[17,0,133])
    backend = SimpleNamespace(forward_metadata=SimpleNamespace(query_start_loc=cu))
    F.after_gdn_metadata(None, backend, batch)
    expected_i = torch.tensor([[0,0],[1,0],[1,1],[1,2]], device="cuda", dtype=torch.int64)
    expected_o = torch.tensor([0,1,1,4], device="cuda", dtype=torch.int64)
    with patch.object(torch.Tensor, "tolist", side_effect=AssertionError("device readback")), \
         patch.object(torch.Tensor, "item", side_effect=AssertionError("device readback")):
        indices = F.around_indices(lambda *_:None, cu, 64)
        offsets = F.around_offsets(lambda *_:None, cu, 64)
    torch.testing.assert_close(indices, expected_i)
    torch.testing.assert_close(offsets, expected_o)
    for name, value in (("prepare_chunk_indices", indices), ("prepare_chunk_offsets", offsets)):
        original = getattr(I, name)
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__
        torch.testing.assert_close(value.to(cu.dtype), original(cu, 64).to(cu.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_pool_reset_graph_reads_changed_indices_and_preserves_other_slots():
    indices = torch.tensor([1,3], device="cuda", dtype=torch.int32)
    conv = SimpleNamespace(conv_state=torch.ones(2,5,4, device="cuda"))
    ngram = SimpleNamespace(context=torch.ones(5,3, device="cuda", dtype=torch.long), eos_token_id=9)
    def reset():
        F.reset_conv(conv, indices)
        F.reset_ngram(ngram, indices)
    for _ in range(3):
        reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reset()
    conv.conv_state.fill_(1)
    ngram.context.fill_(1)
    indices.copy_(torch.tensor([0,4], device="cuda", dtype=torch.int32))
    graph.replay()
    expected_conv = torch.ones_like(conv.conv_state)
    expected_conv[:, [0,4], :] = 0
    expected_ngram = torch.ones_like(ngram.context)
    expected_ngram[[0,4]] = 9
    torch.testing.assert_close(conv.conv_state, expected_conv)
    torch.testing.assert_close(ngram.context, expected_ngram)
