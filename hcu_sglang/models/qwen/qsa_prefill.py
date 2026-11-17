from __future__ import annotations

import msgspec
import torch
import torch.nn.functional as F
import triton

from sglang.srt.layers.attention import qwen_sparse_attn_backend as N
from sglang.srt.layers.attention.qsa import sparse_attn as S
from sglang.srt.runtime_context import get_parallel


class SparsePrefillPlan(msgspec.Struct, frozen=True):
    cu_q: object
    cu_k: object
    lengths: tuple
    extend_lengths: tuple


class SparseMetadata(N.QwenSparseAttnMetadata):
    hcu_sparse_prefill: object = None


def prepare_metadata(result, backend, batch):
    if (not batch.forward_mode.is_extend_without_speculative()
        or not batch.positions.is_cuda
        or get_parallel().attn_cp_size != 1
        or batch.seq_lens_cpu is None
        or batch.extend_seq_lens_cpu is None
        ):
        return result
    lengths = tuple(int(n) for n in batch.seq_lens_cpu)
    extend = tuple(int(n) for n in batch.extend_seq_lens_cpu)
    if not lengths:
        return result
    plan = SparsePrefillPlan(
        F.pad(batch.extend_seq_lens.to(torch.int32).cumsum(0, dtype=torch.int32), (1, 0)),
        F.pad(result.sequence_lengths.cumsum(0, dtype=torch.int32), (1, 0)),
        lengths, extend,
    )
    fields = msgspec.structs.asdict(result)
    fields['hcu_sparse_prefill'] = plan
    return SparseMetadata(**fields)


def sparse_chunk_prefill(q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_q):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_n, warps, stages = S._get_best_config(total_q)
    out = torch.empty_like(q)
    S._sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q, k, v, out, indices, cu_q, cu_k, kv_lens, scale, indices.shape[-1],
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        indices.stride(0), indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads, GROUP_SIZE=group_size,
        BLOCK_M=max(16, triton.next_power_of_2(group_size)), BLOCK_N=block_n,
        HEAD_DIM=head_dim, num_warps=warps, num_stages=stages,
    )
    return out


def around_forward_extend(original, backend, q, k, v, layer, forward_batch,
                          save_kv_cache=True, topk_indices=None, **kwargs):
    meta = backend.forward_metadata
    plan = getattr(meta, 'hcu_sparse_prefill', None)
    if (plan is None or topk_indices is None or not q.is_cuda
        or getattr(forward_batch, 'attn_cp_metadata', None) is not None):
        return original(backend, q, k, v, layer, forward_batch,
                        save_kv_cache=save_kv_cache, topk_indices=topk_indices, **kwargs)
    if save_kv_cache:
        fused = backend._try_fused_kv_attention(q, k, v, layer, forward_batch, topk_indices)
        if fused is not None:
            return fused
        backend.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
    q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
    num_rows, valid_rows = q.shape[0], topk_indices.shape[0]
    if valid_rows > num_rows:
        raise ValueError('QSA top-k rows exceed query rows')
    q = q[:valid_rows].contiguous()
    indices = topk_indices.to(torch.int32).contiguous()
    if all(length == extend for length, extend in zip(plan.lengths, plan.extend_lengths)):
        output = N.sparse_gqa_fwd_interface_triton(
            q, k[:valid_rows].contiguous(), v[:valid_rows].contiguous(),
            max(plan.lengths, default=1), indices, plan.cu_q, layer.scaling,
        )
    else:
        pool = backend.token_to_kv_pool
        packed_k, packed_v = N.pack_qsa_prefill_kv(
            k=pool.get_key_buffer(layer.layer_id), v=pool.get_value_buffer(layer.layer_id),
            req_to_token=backend.req_to_token_pool.req_to_token,
            req_indices=forward_batch.req_pool_indices, cu_k=plan.cu_k,
            total_k=sum(plan.lengths), max_k=max(plan.lengths, default=1), output_dtype=q.dtype,
        )
        output = sparse_chunk_prefill(q, packed_k, packed_v, indices, plan.cu_q,
                                     plan.cu_k, meta.sequence_lengths, layer.scaling,
                                     max(plan.extend_lengths, default=1))
    return backend._pad_extend_output(output, num_rows)


def around_mrope_extend(original, batch, model_runner, schedule_batch):
    config = model_runner.model_config.hf_config
    positions = batch.positions
    if (getattr(config, 'model_type', '').startswith('qwen4_exp')
        and positions is not None and positions.is_cuda and positions.ndim == 1
        and all(mm is None for mm in schedule_batch.multimodal_inputs)):
        batch.mrope_positions = positions.to(torch.int64).unsqueeze(0).expand(3, -1).contiguous()
        return
    return original(batch, model_runner, schedule_batch)
