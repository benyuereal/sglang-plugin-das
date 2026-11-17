from types import SimpleNamespace
import pytest
import torch

from hcu_sglang.models.qwen import qsa_kv as K


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e5m2])
def test_hcu_prefix_pack_matches_logical_token_order(dtype):
    k = torch.randn(4,2,64,128,device="cuda").to(dtype)
    v = torch.randn(4,2,128,64,device="cuda").to(dtype)
    table = torch.randperm(256,device="cuda")[:30].reshape(3,10).to(torch.int32)
    requests = torch.tensor([2,0,1],device="cuda")
    cu = torch.tensor([0,3,3,8],device="cuda",dtype=torch.int32)
    pk,pv = K.around_pack_prefill(lambda **_:None, k=k,v=v,req_to_token=table,
        req_indices=requests,cu_k=cu,total_k=8,max_k=5,output_dtype=torch.bfloat16)
    slots = torch.cat((table[2,:3],table[1,:5])).long()
    expected_k = k.float()[slots//64,:,slots%64,:].to(torch.bfloat16)
    expected_v = v.float()[slots//64,:,:,slots%64].to(torch.bfloat16)
    torch.testing.assert_close(pk,expected_k)
    torch.testing.assert_close(pv,expected_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e5m2])
def test_hcu_layout_sparse_attention_matches_math_and_replays(dtype):
    torch.manual_seed(123)
    k = torch.randn(4, 2, 64, 128, device="cuda").to(dtype)
    v = torch.randn(4, 2, 128, 64, device="cuda").to(dtype)
    q = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(256, device="cuda")[:40].reshape(2, 20).to(torch.int32)
    requests = torch.tensor([0, 1], device="cuda")
    indices = torch.tensor([[3, 5, -1, -1], [2, 7, 9, -1]], device="cuda", dtype=torch.int32)
    meta = SimpleNamespace(sequence_lengths=torch.tensor([10, 12], device="cuda", dtype=torch.int32),
                           is_cuda_graph=False, row_req_pool_indices=requests)
    backend = SimpleNamespace(token_to_kv_pool=SimpleNamespace(get_key_buffer=lambda _:k, get_value_buffer=lambda _:v),
        req_to_token_pool=SimpleNamespace(req_to_token=table),
        _resolve_metadata=lambda _:meta, _expand_block_indices=lambda ids, _:ids,
        _get_fa2_scratch=lambda capacity, heads, dim, dt, device:(torch.empty(capacity,heads,dim,dtype=dt,device=device), torch.empty(capacity,heads,dim,dtype=dt,device=device)))
    layer = SimpleNamespace(layer_id=0, scaling=128**-0.5, k_scale_float=0.5, v_scale_float=0.25)
    def run():
        return K.around_paged_attention(lambda *_:None, backend, q, layer, None, indices)
    def expected():
        outputs = []
        for b in range(2):
            slots = table[b, indices[b][indices[b] >= 0].long()].long()
            key = k[slots // 64, :, slots % 64, :].float()
            value = v[slots // 64, :, :, slots % 64].float()
            if dtype == torch.float8_e5m2:
                key *= 0.5
                value *= 0.25
            key = key.repeat_interleave(4, dim=1)
            value = value.repeat_interleave(4, dim=1)
            logits = torch.einsum("hd,nhd->hn", q[b].float(), key) * layer.scaling
            outputs.append(torch.einsum("hn,nhd->hd", logits.softmax(-1), value))
        return torch.stack(outputs).flatten(1)
    for _ in range(3):
        run()
    torch.testing.assert_close(run().float(), expected(), atol=0.015, rtol=0.015)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    q.copy_(torch.randn_like(q))
    indices.copy_(torch.tensor([[1,8,-1,-1],[0,3,11,-1]], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(output.float(), expected(), atol=0.015, rtol=0.015)
