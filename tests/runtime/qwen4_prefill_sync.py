from types import SimpleNamespace
import os

import pytest
import torch

from hcu_sglang.models.qwen import qwen4_exp as Q, qsa as A
from sglang.srt.layers.attention import qwen_sparse_attn_backend as N
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.plugins.hook_registry import HookRegistry


@pytest.fixture
def registered(monkeypatch):
    hooks = []
    monkeypatch.setattr(Q, 'is_hcu', lambda: True)
    monkeypatch.setattr(HookRegistry, 'register', lambda *args: hooks.append(args))
    Q.register_hooks()
    owned = (
        'QwenSparseAttnBackend._metadata_from_forward_batch',
        'QwenSparseAttnBackend.forward_extend',
        'ForwardBatch._compute_mrope_positions_extend',
    )
    saved = []
    for target, fn, kind in hooks:
        if any(target.endswith(name) for name in owned):
            if os.environ.get('HCU_TEST_PREFILL_BASELINE') == '1' and not target.endswith('_metadata_from_forward_batch'):
                continue
            import importlib
            module_name, cls_name, method = target.rsplit('.', 2)
            cls = getattr(importlib.import_module(module_name), cls_name)
            saved.append((cls, method, getattr(cls, method)))
            HookRegistry._apply_target(target, [(kind, fn, None)])
    yield
    for cls, method, original in reversed(saved):
        setattr(cls, method, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU required')
@pytest.mark.parametrize('prefix', [False, True])
def test_public_prefill_reuses_lengths_without_scalar_sync(registered, monkeypatch, prefix):
    from hcu_sglang.models.qwen import qsa_prefill as P
    monkeypatch.setattr(A, 'get_parallel', lambda: SimpleNamespace(attn_cp_size=1))
    monkeypatch.setattr(P, 'get_parallel', lambda: SimpleNamespace(attn_cp_size=1))
    torch.manual_seed(772)
    extend = [5, 3]
    prefixes = [4, 8] if prefix else [0, 0]
    lengths = [p + e for p, e in zip(prefixes, extend)]
    table = torch.randperm(64, device='cuda').reshape(2, 32).to(torch.int32)
    full_k = torch.randn(64, 1, 64, device='cuda', dtype=torch.bfloat16)
    full_v = torch.randn_like(full_k)
    pool = SimpleNamespace(qsa_compress_ratio=4, qsa_block_topk=2,
                           get_key_buffer=lambda _: full_k, get_value_buffer=lambda _: full_v)
    runner = SimpleNamespace(token_to_kv_pool=pool, device='cuda',
                             req_to_token_pool=SimpleNamespace(req_to_token=table))
    backend = N.QwenSparseAttnBackend(runner)
    positions = torch.cat([torch.arange(p, p + e, device='cuda') for p, e in zip(prefixes, extend)])
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, seq_lens=torch.tensor(lengths, device='cuda', dtype=torch.int32),
                            seq_lens_cpu=torch.tensor(lengths, dtype=torch.int32), extend_seq_lens_cpu=extend,
                            extend_seq_lens=torch.tensor(extend, device='cuda', dtype=torch.int32),
                            req_pool_indices=torch.arange(2, device='cuda'), positions=positions,
                            mrope_positions=None, input_ids=torch.zeros(sum(extend), device='cuda', dtype=torch.int64),
                            out_cache_loc=torch.zeros(sum(extend), device='cuda', dtype=torch.int64), spec_info=None)
    backend.init_forward_metadata(batch)
    meta = backend.forward_metadata
    if hasattr(meta, 'hcu_sparse_prefill'):
        first_plan = meta.hcu_sparse_prefill
    q = torch.randn(10, 4, 64, device='cuda', dtype=torch.bfloat16)
    indices = torch.full((8, 4), -1, device='cuda', dtype=torch.int32)
    indices[:, :2] = torch.tensor([0, 1], device='cuda')
    if not prefix:
        indices[0, 1] = -1
        indices[extend[0], 1] = -1
    slots = torch.cat([table[b, p:p + e] for b, (p, e) in enumerate(zip(prefixes, extend))]).long()
    k, v = full_k[slots], full_v[slots]
    layer = SimpleNamespace(tp_q_head_num=4, head_dim=64, scaling=0.125, layer_id=0)
    expected = []
    for b, rows in enumerate(extend):
        key, value = full_k[table[b, :2].long(), 0].float(), full_v[table[b, :2].long(), 0].float()
        start = sum(extend[:b])
        for i, query in enumerate(q[start:start + rows].float()):
            limit = min(2, prefixes[b] + i + 1)
            logits = query @ key[:limit].T * layer.scaling
            expected.append(logits.softmax(-1) @ value[:limit])
    expected = torch.cat([torch.stack(expected).flatten(1), torch.zeros(2, 256, device='cuda')])
    scalar = torch.Tensor.item
    def forbid_gpu_item(tensor, *args, **kwargs):
        assert not tensor.is_cuda, 'per-layer GPU scalar synchronization'
        return scalar(tensor, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, 'item', forbid_gpu_item)
        output = backend.forward_extend(q, k, v, layer, batch, save_kv_cache=False, topk_indices=indices)
        second = backend.forward_extend(q, k, v, layer, batch, save_kv_cache=False, topk_indices=indices)
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(second, output)
    assert backend.forward_metadata.hcu_sparse_prefill is first_plan
    batch.seq_lens_cpu += 4
    batch.seq_lens += 4
    batch.positions += 4
    backend.init_forward_metadata(batch)
    assert backend.forward_metadata.hcu_sparse_prefill is not first_plan
    assert backend.forward_metadata.hcu_sparse_prefill.lengths == tuple(n + 4 for n in lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU required')
def test_public_mrope_extend_uses_existing_device_positions(registered, monkeypatch):
    from sglang.srt.model_executor import forward_batch_info
    monkeypatch.setattr(forward_batch_info, 'get_exec', lambda: SimpleNamespace(
        deterministic=SimpleNamespace(rl_on_policy_target=None)))
    positions = torch.tensor([8, 9, 10, 32, 33], device='cuda')
    fb = ForwardBatch(forward_mode=ForwardMode.EXTEND, batch_size=2,
                      input_ids=torch.zeros(5, device='cuda', dtype=torch.int64),
                      req_pool_indices=torch.arange(2, device='cuda'),
                      seq_lens=torch.tensor([11, 34], device='cuda'), out_cache_loc=torch.arange(5, device='cuda'),
                      seq_lens_sum=45, positions=positions, seq_lens_cpu=torch.tensor([11, 34]))
    runner = SimpleNamespace(device='cuda', model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type='qwen4_exp')))
    schedule = SimpleNamespace(multimodal_inputs=[None, None], extend_lens=[3, 2], prefix_lens=[8, 32])
    tensor_to = torch.Tensor.to
    def forbid_host_copy(tensor, *args, **kwargs):
        assert tensor.is_cuda, 'text MRoPE copied from host'
        return tensor_to(tensor, *args, **kwargs)
    with monkeypatch.context() as m:
        m.setattr(torch.Tensor, 'to', forbid_host_copy)
        fb._compute_mrope_positions_extend(runner, schedule)
    torch.testing.assert_close(fb.mrope_positions, positions.expand(3, -1))


def test_mrope_mm_and_other_model_preserve_effective_baseline():
    from hcu_sglang.models.qwen import qsa_prefill as P
    for model, mm in [('qwen4_exp', [object()]), ('other', [None])]:
        batch = SimpleNamespace(positions=torch.arange(3))
        runner = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model)))
        marker = object()
        assert P.around_mrope_extend(lambda *a: marker, batch, runner,
                                    SimpleNamespace(multimodal_inputs=mm)) is marker
