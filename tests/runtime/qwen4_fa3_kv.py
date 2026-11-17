import pytest
import torch

from hcu_sglang.layers.attention.flashattention_interface import flash_attn_with_kvcache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU required")
@pytest.mark.parametrize("query_len", [1, 4, 12])
def test_qwen4_e5m2_hcu_fa_matches_math_with_live_graph_inputs(query_len):
    torch.manual_seed(135 + query_len)
    batch, heads, kv_heads, dim, page, max_len = 2, 24, 2, 256, 64, 513
    pages = (max_len + page - 1) // page
    q = torch.randn(batch * query_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    # Production default pool layout [NP, PS, H, D] (bshd pages): the
    # varlen path declares exactly this layout to the wheel, so the
    # test must carry it physically -- a .view() cannot reorder the
    # axes of a BHSD buffer and would mislabel every head/page.
    k = torch.randn(batch * pages, page, kv_heads, dim, device="cuda").to(torch.float8_e5m2)
    v = torch.randn(batch * pages, page, kv_heads, dim, device="cuda").to(torch.float8_e5m2)
    table = torch.randperm(batch * pages, device="cuda").reshape(batch, pages).int()
    lengths = torch.tensor([137, max_len], device="cuda", dtype=torch.int32)
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * query_len

    def call():
        return flash_attn_with_kvcache(q, k,
            v, cache_seqlens=lengths,
            page_table=table, cu_seqlens_q=cu_q, max_seqlen_q=query_len,
            max_seqlen_k=max_len, softmax_scale=dim**-0.5, causal=True)

    def reference():
        outputs = []
        for b in range(batch):
            n = int(lengths[b])
            keys = k.float()[table[b].long()].reshape(-1, kv_heads, dim)[:n]
            values = v.float()[table[b].long()].reshape(-1, kv_heads, dim)[:n]
            keys, values = (x.repeat_interleave(heads // kv_heads, dim=1) for x in (keys, values))
            logits = torch.einsum("qhd,khd->hqk", q.reshape(batch, query_len, heads, dim)[b].float(), keys) * dim**-0.5
            visible = torch.arange(n, device="cuda")[None, :] <= n-query_len+torch.arange(query_len, device="cuda")[:, None]
            logits.masked_fill_(~visible[None], -float("inf"))
            outputs.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), values))
        return torch.stack(outputs)

    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    for i in range(2):
        q.copy_(torch.randn_like(q))
        k.copy_(torch.randn(k.shape, device="cuda").to(k.dtype))
        v.copy_(torch.randn(v.shape, device="cuda").to(v.dtype))
        table.copy_(torch.roll(table, 1, dims=1))
        lengths.copy_(torch.tensor([129-i, max_len-i], device="cuda", dtype=torch.int32))
        graph.replay()
        actual, expected = output.reshape(batch, query_len, heads, dim).float(), reference()
        assert torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0) > 0.9999
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
