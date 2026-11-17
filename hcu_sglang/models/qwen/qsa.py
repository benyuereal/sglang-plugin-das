from __future__ import annotations
import msgspec
from sglang.srt.layers.attention.qsa import metadata as M
from sglang.srt.runtime_context import get_parallel
from sglang.srt.layers.attention.qsa import qsa_indexer as N
globals().update({name: value for name, value in vars(N).items() if not name.startswith("__")})


class PrefillPlan(msgspec.Struct, frozen=True):
    compressed_locs: object
    row_starts: object
    row_ends: object


class IndexerMetadata(M.QSAIndexerMetadata):
    hcu_prefill: object = None


def after_metadata(result, backend, forward_batch):
    from hcu_sglang.models.qwen.qsa_prefill import prepare_metadata
    result = prepare_metadata(result, backend, forward_batch)
    if (not forward_batch.forward_mode.is_extend_without_speculative()
        or get_parallel().attn_cp_size != 1
        or forward_batch.seq_lens_cpu is None
        or forward_batch.extend_seq_lens_cpu is None
        or result.indexer_metadata.compress_ratio <= 1):
        return result
    meta = result.indexer_metadata
    ratio = meta.compress_ratio
    lengths = tuple(int(n) for n in forward_batch.seq_lens_cpu)
    parts = [meta.token_slot_table[i, :length // ratio * ratio:ratio].long() // ratio
             for i, length in enumerate(lengths) if length >= ratio]
    locs = torch.cat(parts) if parts else meta.token_slot_table.new_empty((0,), dtype=torch.long)
    positions = forward_batch.positions
    if positions.ndim == 2:
        positions = positions[0]
    starts, ends, _ = M.build_qsa_row_ranges(meta.sequence_lengths,
        positions[:meta.token_to_batch_idx.numel()], meta.token_to_batch_idx, ratio)
    fields = msgspec.structs.asdict(meta)
    fields["prefill_sequence_lengths_cpu"] = lengths
    fields["hcu_prefill"] = PrefillPlan(locs, starts, ends)
    return msgspec.structs.replace(result, indexer_metadata=IndexerMetadata(**fields))


def around_prefill_inputs(original, meta, layer_id, positions, query_sequence_ids=None):
    plan = getattr(meta, "hcu_prefill", None)
    if plan is None or query_sequence_ids is not None:
        return original(meta, layer_id, positions, query_sequence_ids)
    buffer = meta.token_to_kv_pool.get_qsa_compressed_k_buffer(layer_id)
    return (buffer.index_select(0, plan.compressed_locs), plan.row_starts,
            plan.row_ends, meta.sequence_lengths)

def project_qk(self, hidden_states: torch.Tensor, positions: torch.Tensor, pool=None, cache_loc: torch.Tensor | None=None, q_heads_padded: int | None=None) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    (qk, _) = self.index_qk_proj(hidden_states)
    token_k = qk[:, self.index_n_heads * self.index_head_dim:].reshape(-1, self.index_kv_heads, self.index_head_dim)
    if pool is not None and cache_loc is not None and (qk.shape[0] > 0) and self._use_fused_prep(qk):
        from sglang.kernels.ops.attention.qsa_indexer import qsa_index_q_norm_rope_store
        key_state_buffer = pool.get_qsa_key_state_buffer(self.layer_id)
        q = qsa_index_q_norm_rope_store(qk, positions.long(), self.rotary_emb.cos_sin_cache, self._rope_axis_map(qk.device), self.q_layernorm.weight.data, cache_loc[:qk.shape[0]].long(), key_state_buffer.view(key_state_buffer.shape[0], -1), pool.qsa_rope_position_buffer, self.index_n_heads, self.rotary_emb.rotary_dim, self.q_layernorm.variance_epsilon, self.rotary_emb.is_neox_style, q_heads_padded=q_heads_padded, out_dtype=pool.qsa_compressed_dtype)
        return (q, token_k, True)
    q_raw = qk[:, :self.index_n_heads * self.index_head_dim]
    q = self.q_layernorm(q_raw.reshape(-1, self.index_head_dim)).reshape(-1, self.index_n_heads, self.index_head_dim)
    q = self.apply_rope(positions, q)
    if pool is not None:
        q = q.to(pool.qsa_compressed_dtype)
    return (q, token_k, False)

def apply_rope(self, positions: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    positions = positions.long()
    num_positions = positions.shape[-1] if positions.ndim == 2 else positions.numel()
    if num_positions != tensor.shape[0]:
        raise ValueError('QSA RoPE positions must match the token dimension')
    self.rotary_emb.get_cos_sin_with_position(positions)
    rotary_dim = self.rotary_emb.rotary_dim
    half_rotary_dim = rotary_dim // 2
    cos = self.rotary_emb.position_cos.reshape(num_positions, -1)[:, :half_rotary_dim]
    sin = self.rotary_emb.position_sin.reshape(num_positions, -1)[:, :half_rotary_dim]
    rotated = apply_rotary_emb(tensor[..., :rotary_dim], cos, sin, self.rotary_emb.is_neox_style)
    return torch.cat([rotated, tensor[..., rotary_dim:]], dim=-1)
