from __future__ import annotations

import logging
import re
from contextvars import ContextVar

import torch
import triton
import triton.language as tl

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers import dp_attention as DP
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.runtime_context import get_parallel
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from hcu_sglang.models.qwen.prefill_policy import compact_prefill_counts, pure_prefill_modes
from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)
_ffn = ContextVar("qwen4_ffn", default=None)
_hc_snapshot = ContextVar("qwen4_hc_snapshot", default=None)


def compact_layout(batch, runner, counts):
    batch.qwen4_compact_prefill = False
    if not envs.SGLANG_QWEN4_COMPACT_PREFILL.get():
        return None
    from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend, is_tbo_enabled
    from sglang.srt.runtime_context import get_exec
    p = get_parallel()
    eligible = (
        isinstance(runner.model_config.hf_config.get_text_config(), Qwen4ExpTextConfig)
        and not runner.is_draft_worker
        and not getattr(runner, "enable_elastic_ep", False)
        and not DP.world_dp_gather_enabled()
        and p.tp_size == p.attn_dp_size == p.moe_ep_size
        and p.attn_tp_size == p.attn_cp_size == 1
        and get_moe_a2a_backend().is_none()
        and not is_tbo_enabled()
        and get_exec().graph.cuda_graph_config.prefill.backend == "disabled"
        and DP._USE_DP_GATHERV
    )
    result = compact_prefill_counts(counts, pure_prefill=batch.dp_pure_prefill, eligible=eligible)
    if result is not None:
        if batch.spec_info is not None and (
            batch.spec_info.num_tokens_per_req != 1
            or batch.spec_info.num_tokens_for_logprob_per_req != 1
        ):
            raise RuntimeError("compact prefill received a speculative-width layout")
        batch.qwen4_compact_prefill = True
        if envs.SGLANG_QWEN4_PREFILL_DIAGNOSTICS.get() and not getattr(runner, "_compact_reported", False):
            logger.info("QWEN4_COMPACT_PREFILL rank=%s counts=%s global_M=%s", p.attn_dp_rank, result, sum(result))
            runner._compact_reported = True
    return result


def after_update_gather(result, batch, mlp_sync_info, require_mlp_tp_gather, draft_require_mlp_tp_gather=None, skip_global_metadata=False):
    batch.dp_pure_prefill = not skip_global_metadata and pure_prefill_modes(
        mlp_sync_info.tp0_info_cpu[:, 5].tolist(),
        extend=ForwardMode.EXTEND.value, idle=ForwardMode.IDLE.value,
    )
    return result


def around_ffn(original, layer, hidden_states, forward_batch):
    active = (getattr(forward_batch, "qwen4_compact_prefill", False)
              and envs.SGLANG_QWEN4_LOCAL_ROUTING.get())
    wire_int8 = envs.SGLANG_ENABLE_DP_GATHER_INT8.get()
    if not active and not wire_int8:
        return original(layer, hidden_states, forward_batch)
    state = {"mlp": layer.mlp, "local": active, "batch": forward_batch,
             "prequant": None, "topk": None, "consumed": False}
    token = _ffn.set(state)
    try:
        result = original(layer, hidden_states, forward_batch)
        if state["prequant"] is not None and not state["consumed"]:
            raise RuntimeError("Qwen4 INT8 handoff was not consumed")
        return result
    finally:
        _ffn.reset(token)


@triton.jit
def _dequant(q, scale, output, HIDDEN: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    s = tl.load(scale + row)
    for start in range(triton.cdiv(HIDDEN, BLOCK)):
        cols = start * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(q + row * HIDDEN + cols, cols < HIDDEN, 0).to(tl.float32)
        tl.store(output + row * HIDDEN + cols, value * s, cols < HIDDEN)


def around_dp_gather(original, hidden_states, forward_batch, cp_shard_counts=None):
    state = _ffn.get()
    if state is None or state["batch"] is not forward_batch or forward_batch.dp_padding_mode.is_max_len():
        return original(hidden_states, forward_batch, cp_shard_counts)
    sizes = DP.get_dp_global_num_tokens()
    mlp = state["mlp"]
    if cp_shard_counts is not None or not DP.is_dp_gatherv_active():
        raise RuntimeError("Qwen4 optimized gather requires TP=DP=EP with gatherv")
    if hidden_states.shape[0] != sizes[get_parallel().attn_dp_rank] or min(sizes) < 1:
        raise RuntimeError("Qwen4 compact token geometry is inconsistent")
    if mlp.enable_shared_expert_fusion or mlp.experts.reduce_results:
        raise RuntimeError("Qwen4 gather requires separate shared experts and one boundary reduction")
    use_int8 = (envs.SGLANG_QWEN4_ROUTED_INT8.get() if state["local"]
                else envs.SGLANG_ENABLE_DP_GATHER_INT8.get())
    if state["local"]:
        logits, _ = mlp.gate(hidden_states)
        selected = mlp.topk(hidden_states, logits)
        if not TopKOutputChecker.format_is_standard(selected):
            raise RuntimeError("Qwen4 local routing requires StandardTopKOutput")
    group = get_tp_group()
    output = DP.get_global_dp_buffer(group)
    if use_int8:
        if not getattr(mlp.experts.quant_method, "supports_int8_prequant", False):
            raise RuntimeError("Qwen4 INT8 transport requires an explicit INT8 quant method")
        from lightop.quant import per_token_quant_int8
        q, scale = per_token_quant_int8(hidden_states.contiguous())
        inputs = [q, scale]
        if state["local"]:
            inputs += [selected.topk_weights, selected.topk_ids]
        gathered = group.all_gatherv(inputs, sizes=sizes)
        gq, gs = gathered[:2]
        state["prequant"] = (gq, gs)
        _dequant[(sum(sizes),)](gq, gs, output, HIDDEN=hidden_states.shape[1], BLOCK=2048)
        if state["local"]:
            weights, ids = gathered[2:]
    else:
        group.all_gatherv(hidden_states, sizes=sizes, output=output)
        weights, ids = group.all_gatherv([selected.topk_weights, selected.topk_ids], sizes=sizes)
    if state["local"]:
        state["topk"] = StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=None)
    if not getattr(around_dp_gather, "reported", False):
        logger.info("QWEN4_LOCAL_ROUTING sizes=%s local=%s int8_direct=%s", sizes, state["local"], use_int8)
        around_dp_gather.reported = True
    return output


def around_router(original, mlp, hidden_states, *, defer_finalize=False):
    state = _ffn.get()
    if state is None or state["mlp"] is not mlp or state["topk"] is None:
        return original(mlp, hidden_states, defer_finalize=defer_finalize)
    if defer_finalize:
        raise RuntimeError("Qwen4 INT8 routing cannot defer finalize")
    selected = state["topk"]
    state["topk"] = None
    return mlp.experts(hidden_states, selected)


def around_moe_core(original, layer, dispatch_output):
    state = _ffn.get()
    if state is None or state["mlp"].experts is not layer or state["prequant"] is None:
        return original(layer, dispatch_output)
    if state["consumed"]:
        raise RuntimeError("Qwen4 INT8 handoff consumed twice")
    q, scale = state["prequant"]
    state["consumed"] = True
    return layer.quant_method.apply(layer, dispatch_output, i_q=q, i_s=scale, use_int8_prequant=True)


def around_load_weights(original, model, weights, *, mtp=False):
    from hcu_sglang.models.qwen import ple
    modules = {name: value for name, value in model.named_modules()
               if isinstance(value, ple.N.Qwen4ExpNGramEmbedding)}
    config = getattr(model.config, "text_config", model.config)
    shards = int(getattr(config, "split_ngram_parts", getattr(model.config, "split_ngram_parts", 512)))
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded = set()
    def stream():
        for name, value in weights:
            normalized = name.replace("model.language_model.", "model.")
            if mtp and name.startswith("mtp."):
                normalized = name.replace("mtp.", "model.")
            expert = re.search(r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)\.weight_scale$", normalized)
            if expert is not None:
                target_name = expert[1] + (".w13_weight_scale" if expert[2] == "gate_up_proj" else ".w2_weight_scale")
                if target_name in params:
                    param = params[target_name]
                    pieces = zip(("w1", "w3"), value.chunk(2, dim=-2)) if expert[2] == "gate_up_proj" else (("w2", value),)
                    for shard_id, piece in pieces:
                        for expert_id in range(value.shape[0]):
                            param.weight_loader(param, piece[expert_id], target_name, shard_id, expert_id)
                    loaded.add(target_name)
                    continue
            match = re.search(r"^(.*)\.ngram_embedding\.shard_(\d+)\.(weight|weight_scale)$", normalized)
            if match is None or match[1] not in modules:
                yield name, value
                continue
            emb = modules[match[1]].ngram_embedding
            field = match[3]
            destination = getattr(emb, field)
            # Same placement as sglang-das: shard i fills slice i. A wide
            # tensor spills into the next slices; those slices are then
            # overwritten by the following shards' first blocks.
            shard_size = (emb.org_vocab_size + shards - 1) // shards
            start = int(match[2]) * shard_size
            lo = max(start, emb.shard_indices.org_vocab_start_index)
            hi = min(start + value.shape[0], emb.shard_indices.org_vocab_end_index)
            if hi > lo:
                target = destination[lo - emb.shard_indices.org_vocab_start_index:hi - emb.shard_indices.org_vocab_start_index]
                source = value[lo - start:hi - start].reshape(target.shape)
                target.copy_(source.to(device=target.device, dtype=target.dtype))
            loaded.add(match[1]+".ngram_embedding."+field)
    result = original(model, stream())
    # Scale tensors are copied straight into the CPU buffers (shard
    # slices above, the official non-sharded loader below), so the
    # construction-time GPU replica of a scalar scale would keep its
    # init value 1 while _scale_ple_embeddings prefers the replica --
    # a scale!=1 checkpoint would dequantize with 1. Refresh every
    # replica now, before warmup/graph capture; reloads re-enter here.
    for _, emb in model.named_modules():
        if isinstance(emb, ple.Qwen4ExpPinnedHostEmbedding):
            replica = getattr(emb, "_device_scalar_scale", None)
            if replica is not None and replica.shape == emb.weight_scale.shape:
                replica.copy_(emb.weight_scale)
    return result | loaded


def around_mtp_load_weights(original, model, weights, *args, **kwargs):
    return around_load_weights(lambda m, w: original(m, w, *args, **kwargs), model, weights, mtp=True)


def around_model_forward(original, model, input_ids, positions, forward_batch, *args, **kwargs):
    if not (is_hcu() and model.pp_group.is_last_rank
            and not forward_batch.forward_mode.is_idle()):
        return original(model, input_ids, positions, forward_batch, *args, **kwargs)
    state = {"mixer": model.hyper_connection_mixer, "snapshot": None}
    token = _hc_snapshot.set(state)
    try:
        result = original(model, input_ids, positions, forward_batch, *args, **kwargs)
        if state["snapshot"] is None:
            return result
        return result[0], state["snapshot"]
    finally:
        _hc_snapshot.reset(token)


def around_mix_streams(original, mixer, hidden_states):
    state = _hc_snapshot.get()
    if state is not None and state["mixer"] is mixer.hyper_connection_mixer:
        state["snapshot"] = hidden_states.detach().contiguous().clone()
    return original(mixer, hidden_states)


def after_conditional_forward(result, model, *args, **kwargs):
    if is_hcu():
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        if (isinstance(result, LogitsProcessorOutput)
                and model.model.last_hc_hidden_states is not None
                and result.hidden_states is not None):
            result.hidden_states = result.hidden_states.detach().contiguous().clone()
    return result


def around_start_prefetch(original, layer, batch, forward_batch):
    if not is_hcu():
        return original(layer, batch, forward_batch)
    if layer._prefetch_stream is None:
        return
    if layer._prefetch_state is not None:
        raise RuntimeError("PLE prefetch state was not consumed before reuse")
    if batch is None:
        if not layer.ple_embedding.gather_dp_tokens:
            return
        physical_tokens = forward_batch.input_ids.numel()
        ngram_ids = forward_batch.input_ids.new_zeros(
            (physical_tokens, layer.ple_embedding.ngram_heads)
        )
    else:
        physical_tokens = batch.physical_tokens
        ngram_ids = layer.ple_embedding.compute_ngram_ids(batch)
    lookup_ids, semantic_tokens = layer.ple_embedding._prepare_embedding_lookup(
        ngram_ids, forward_batch, physical_tokens
    )
    lookup_tokens = lookup_ids.shape[0]
    if lookup_tokens == 0:
        return
    prefetched = layer._get_prefetch_buffer(lookup_tokens, lookup_ids)
    output_view = prefetched.view(lookup_tokens, layer.ple_embedding.ngram_heads, -1)
    offloaded_embedding = layer.ple_embedding.ngram_embedding
    if hasattr(offloaded_embedding, "_maybe_register_uva"):
        offloaded_embedding._maybe_register_uva()
    offloaded_embedding.gather(lookup_ids, out=output_view)
    layer._prefetch_state = prefetched, semantic_tokens, physical_tokens


def consume_prefetched(layer, forward_batch):
    if layer._prefetch_state is None:
        raise RuntimeError("PLE prefetch state is missing")
    embeddings, semantic_tokens, physical_tokens = layer._prefetch_state
    torch.cuda.current_stream().wait_stream(layer._prefetch_stream)
    embeddings = layer.ple_embedding.ngram_embedding.reduce(embeddings)
    if not layer.ple_embedding._uses_per_row_scale():
        embeddings = layer.ple_embedding._scale_ple_embeddings(embeddings, embeddings.new_empty(0, dtype=torch.long))
    embeddings = layer.ple_embedding._finish_embedding_lookup(embeddings, semantic_tokens, forward_batch, physical_tokens)
    layer._prefetch_state = None
    return embeddings


def after_capture_prepare(result, runner, bs, stream_idx=None, num_tokens=None):
    if not runner.require_mlp_tp_gather:
        return result
    fb = result[0]
    if not getattr(runner.model_runner.model_config.hf_config, "model_type", "").startswith("qwen4_exp"):
        return result
    rows = fb.input_ids.shape[0]
    key = runner._capture_graph_size(bs=bs, num_tokens=rows)
    snapshots = getattr(runner, "_qwen4_dp_snapshots", None)
    if snapshots is None:
        snapshots = runner._qwen4_dp_snapshots = {}
    snapshots[key] = (fb.global_dp_buffer_len, rows, fb.dp_padding_mode.is_max_len(), list(fb.global_num_tokens_cpu))
    return result


def after_load_batch(result, runner, *args, **kwargs):
    snapshots = getattr(runner, "_qwen4_dp_snapshots", {})
    snapshot = snapshots.get(runner._replay_graph_key.size) if snapshots else None
    if snapshot is not None:
        DP.set_dp_buffer_len(*snapshot)
        DP.set_is_extend_in_batch(False)
    return result


def register_hooks():
    if not is_hcu():
        return
    from hcu_sglang.models.qwen import ple, qsa, qsa_kv, qsa_prefill, qwen4_exp_mtp, fla_prefill, hc_mix, state_scatter
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    def register(target, fn, kind=HookType.AROUND):
        HookRegistry.register(target, fn, kind)
    register("sglang.srt.layers.hyperconnection.fused_hc_mix", hc_mix.around_fused_hc_mix)
    register("sglang.srt.layers.attention.hybrid_linear_attn_backend.fused_mamba_state_scatter_with_mask", state_scatter.around_state_scatter)
    register("sglang.srt.models.qwen4_exp.Qwen4ExpPinnedHostEmbedding", ple.Qwen4ExpPinnedHostEmbedding, HookType.REPLACE)
    for name, fn in vars(ple.NGramMethods).items():
        if callable(fn):
            if name in ("_scale_ple_embeddings", "_uses_per_row_scale", "_lookup_ngram_embeddings"):
                setattr(ple.N.Qwen4ExpNGramEmbedding, name, fn)
            else:
                register("sglang.srt.models.qwen4_exp.Qwen4ExpNGramEmbedding."+name, fn, HookType.REPLACE)
    register("sglang.srt.models.qwen4_exp.Qwen4ExpModel.forward", around_model_forward)
    register("sglang.srt.models.qwen4_exp._MixStreams.__call__", around_mix_streams)
    register("sglang.srt.models.qwen4_exp.Qwen4ExpForConditionalGeneration.forward", after_conditional_forward, HookType.AFTER)
    register("sglang.srt.models.qwen4_exp.Qwen4ExpPLELayer.start_prefetch", around_start_prefetch)
    register("sglang.srt.models.qwen4_exp.Qwen4ExpPLELayer._consume_prefetched_embeddings", consume_prefetched, HookType.REPLACE)
    register("sglang.srt.models.qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights", around_load_weights)
    register("sglang.srt.models.qwen4_exp_mtp.Qwen4ExpForCausalLMMTP.load_weights", around_mtp_load_weights)
    for name in ("_init_pre_fc_norms", "_init_linear_projections", "_fuse_residual_linear_shared"):
        register("sglang.srt.models.qwen4_exp_mtp.Qwen4ExpForCausalLMMTP."+name, getattr(qwen4_exp_mtp, name), HookType.REPLACE)
    register("sglang.srt.models.qwen4_exp.Qwen4ExpLayerExtensionMixin._run_ffn_stage", around_ffn)
    register("sglang.srt.layers.layer_boundary.ops.dp_gather", around_dp_gather)
    register("sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock._forward_router_experts", around_router)
    register("sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE.run_moe_core", around_moe_core)
    register("sglang.srt.managers.scheduler_components.dp_attn._update_gather_batch", after_update_gather, HookType.AFTER)
    register("sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.capture_prepare", after_capture_prepare, HookType.AFTER)
    register("sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner.load_batch", after_load_batch, HookType.AFTER)
    for name in ("project_qk", "apply_rope"):
        register("sglang.srt.layers.attention.qsa.qsa_indexer.QSAIndexer."+name, getattr(qsa, name), HookType.REPLACE)
    register("sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._metadata_from_forward_batch", qsa.after_metadata, HookType.AFTER)
    register("sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend.forward_extend", qsa_prefill.around_forward_extend)
    register("sglang.srt.model_executor.forward_batch_info.ForwardBatch._compute_mrope_positions_extend", qsa_prefill.around_mrope_extend)
    register("sglang.srt.layers.attention.qsa.metadata.QSAIndexerMetadata.get_prefill_mqa_inputs", qsa.around_prefill_inputs)
    register("sglang.srt.layers.attention.qwen_sparse_attn_backend.QwenSparseAttnBackend._forward_paged_attention", qsa_kv.around_paged_attention)
    register("sglang.srt.layers.attention.qsa.sparse_attn.pack_qsa_prefill_kv", qsa_kv.around_pack_prefill)
    register("sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend.init_forward_metadata", fla_prefill.after_gdn_metadata, HookType.AFTER)
    register("sglang.kernels.ops.attention.fla.index.prepare_chunk_indices", fla_prefill.around_indices)
    register("sglang.kernels.ops.attention.fla.index.prepare_chunk_offsets", fla_prefill.around_offsets)
    register("sglang.srt.mem_cache.ple_state_pool.ShortConvPool.reset_slots", fla_prefill.reset_conv, HookType.REPLACE)
    register("sglang.srt.mem_cache.ple_state_pool.NGramPool.reset_slots", fla_prefill.reset_ngram, HookType.REPLACE)
