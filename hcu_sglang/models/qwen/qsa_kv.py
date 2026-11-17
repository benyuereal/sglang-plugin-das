from __future__ import annotations

from typing import Optional
import torch
import triton
import triton.language as tl
from sglang.srt.layers.attention import qwen_sparse_attn_backend as N

_FP8_DTYPES = tuple(getattr(torch, name) for name in (
    "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"
) if hasattr(torch, name))


def around_paged_attention(original, backend, q, layer, forward_batch, topk_indices):
    pool = backend.token_to_kv_pool
    k = pool.get_key_buffer(layer.layer_id)
    v = pool.get_value_buffer(layer.layer_id)
    if k.ndim != 4 or v.ndim != 4:
        return original(backend, q, layer, forward_batch, topk_indices)
    if k.shape != v.shape:
        raise ValueError("QSA expected HCU FA BHSD K/V layout")
    meta = backend._resolve_metadata(forward_batch)
    indices = backend._expand_block_indices(topk_indices, meta).to(torch.int32).contiguous()
    batch, topk = indices.shape
    if meta.is_cuda_graph:
        counts, cu_k, cu_q = meta.fa2_valid_counts, meta.fa2_cu_seqlens_k, meta.fa2_cu_seqlens_q
    else:
        counts = torch.empty(batch, dtype=torch.int32, device=q.device)
        cu_k = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
        cu_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    N.qwen_sparse_fa2_cu_seqlens_triton(meta.sequence_lengths, indices, counts, cu_k, batch, topk)
    capacity = backend._cuda_graph_max_tokens * topk if meta.is_cuda_graph else batch * topk
    packed_k, packed_v = backend._get_fa2_scratch(capacity, k.shape[1], k.shape[3], q.dtype, k.device)
    requests = meta.row_req_pool_indices if meta.row_req_pool_indices is not None else forward_batch.req_pool_indices
    qwen_sparse_kv_extraction_compact_hcu_fa_triton(
        k, v, backend.req_to_token_pool.req_to_token, requests, indices,
        meta.sequence_lengths, cu_k, packed_k, packed_v, batch, topk, k.shape[2],
        k_scale=getattr(layer, "k_scale_float", None), v_scale=getattr(layer, "v_scale_float", None),
    )
    relative = torch.arange(topk, dtype=torch.int32, device=q.device).expand(batch, -1)
    relative = relative.masked_fill(relative >= counts[:, None], -1).contiguous()
    output = N.sparse_gqa_packed_decode_triton(q.contiguous(), packed_k, packed_v, relative, cu_q, cu_k, counts, layer.scaling)
    return output.reshape(q.shape[0], -1)


def around_pack_prefill(original, *, k, v, req_to_token, req_indices, cu_k, total_k, max_k, output_dtype=None):
    if k.ndim != 4 or v.ndim != 4:
        return original(k=k, v=v, req_to_token=req_to_token, req_indices=req_indices,
                        cu_k=cu_k, total_k=total_k, max_k=max_k, output_dtype=output_dtype)
    packed = torch.arange(total_k, device=k.device)
    rows = torch.bucketize(packed, cu_k[1:], right=True)
    slots = req_to_token[req_indices[rows].long(), packed - cu_k[rows]].long()
    pages, offsets = slots // k.shape[2], slots % k.shape[2]
    return (k[pages, :, offsets, :].to(output_dtype or k.dtype),
            v[pages, :, offsets, :].to(output_dtype or v.dtype))

def is_fp8_kv_dtype(dtype: torch.dtype) -> bool:
    return dtype in _FP8_DTYPES


def _unit_scale(scale: Optional[float]) -> float:
    return 1.0 if scale is None else float(scale)


@triton.jit
def _compact_kv_hcu_fa(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    page_size: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    k_sp,
    k_sh,
    k_so,
    k_sd,
    v_sp,
    v_sh,
    v_so,
    v_sd,
    ok0,
    ok1,
    ok2,
    ov0,
    ov1,
    ov2,
    k_scale,
    v_scale,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DEQUANTIZE_FP8: tl.constexpr,
):
    """Gather selected tokens from HCU FA paged KV into packed NHD rows.

    K and V are both (pages, heads, page, dim). Physical
    slots are page * page_size + offset, matching MHATokenToKVPool.set_kv_buffer
    under SGLANG_KV_LAYOUT_HCU_FA. This does not change the pool layout.
    """
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    ).to(tl.int64)
    page = slots // page_size
    off = slots % page_size
    k_ptrs = (
        k
        + page[:, None] * k_sp
        + head * k_sh
        + off[:, None] * k_so
        + dims[None, :] * k_sd
    )
    v_ptrs = (
        v
        + page[:, None] * v_sp
        + head * v_sh
        + off[:, None] * v_so
        + dims[None, :] * v_sd
    )
    dst = (pack_start + cols)[:, None]
    mask = valid[:, None] & (dims[None, :] < dim)
    k_dst = out_k + dst * ok0 + head * ok1 + dims[None, :] * ok2
    v_dst = out_v + dst * ov0 + head * ov1 + dims[None, :] * ov2
    k_values = tl.load(k_ptrs, mask=mask, other=0.0)
    v_values = tl.load(v_ptrs, mask=mask, other=0.0)
    if DEQUANTIZE_FP8:
        k_values = k_values.to(tl.float32) * k_scale
        v_values = v_values.to(tl.float32) * v_scale
    tl.store(k_dst, k_values, mask=mask)
    tl.store(v_dst, v_values, mask=mask)


def qwen_sparse_kv_extraction_compact_hcu_fa_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    page_size: int,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
):
    """QSA-only compact from HCU FA paged KV. Pool layout is left unchanged."""
    if k.dtype != v.dtype or out_k.dtype != out_v.dtype:
        raise ValueError("QSA compact K/V input and output dtype pairs must match")
    dequantize_fp8 = is_fp8_kv_dtype(k.dtype) and not is_fp8_kv_dtype(out_k.dtype)
    heads = k.shape[1]
    dim = k.shape[3]
    block_topk = 16
    _compact_kv_hcu_fa[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        page_size,
        req_to_token.stride(0),
        indices.stride(0),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        out_v.stride(0),
        out_v.stride(1),
        out_v.stride(2),
        _unit_scale(k_scale),
        _unit_scale(v_scale),
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        DEQUANTIZE_FP8=dequantize_fp8,
        num_warps=8,
    )
