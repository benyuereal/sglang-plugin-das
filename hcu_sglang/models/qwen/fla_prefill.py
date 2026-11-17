import torch
import triton
from sglang.kernels.ops.attention.fla.utils import tensor_cache


def after_gdn_metadata(result, backend, forward_batch):
    if not forward_batch.forward_mode.is_extend_without_speculative():
        return result
    lengths = forward_batch.extend_seq_lens_cpu
    cu = backend.forward_metadata.query_start_loc
    if (forward_batch.forward_mode.is_extend_without_speculative()
        and lengths is not None and cu is not None and len(lengths) + 1 == cu.numel()):
        cu._fla_lengths_cpu = tuple(int(n) for n in lengths)
    return result


@tensor_cache
def _indices(cu, chunk):
    indices = torch.cat([torch.arange(triton.cdiv(n, chunk)) for n in cu._fla_lengths_cpu])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).pin_memory().to(device=cu.device, non_blocking=True)


@tensor_cache
def _offsets(cu, chunk):
    offsets = [0]
    for n in cu._fla_lengths_cpu:
        offsets.append(offsets[-1] + triton.cdiv(n, chunk))
    return torch.tensor(offsets, dtype=torch.int64, pin_memory=True).to(device=cu.device, non_blocking=True)


def around_indices(original, cu_seqlens, chunk_size):
    return _indices(cu_seqlens, chunk_size) if hasattr(cu_seqlens, "_fla_lengths_cpu") else original(cu_seqlens, chunk_size)


def around_offsets(original, cu_seqlens, chunk_size):
    return _offsets(cu_seqlens, chunk_size) if hasattr(cu_seqlens, "_fla_lengths_cpu") else original(cu_seqlens, chunk_size)


def reset_conv(pool, indices):
    if pool.conv_state is not None and indices.numel() > 0:
        pool.conv_state.index_fill_(1, indices.to(dtype=torch.long), 0)


def reset_ngram(pool, indices):
    if pool.context is not None and indices.numel() > 0:
        pool.context.index_fill_(0, indices.to(dtype=torch.long), pool.eos_token_id)
