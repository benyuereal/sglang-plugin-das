from types import SimpleNamespace
from unittest.mock import patch

import torch
import pytest
from hcu_sglang.models.qwen import ple, qwen4_exp as Q
from hcu_sglang.environ import register_envs

register_envs()


class FakeVocab(torch.nn.Module):
    def __init__(self, rows, cols, params_dtype, **kwargs):
        super().__init__()
        self.tp_size = 1
        self.num_embeddings_per_partition = rows
        self.shard_indices = SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=rows)
        self.weight = torch.nn.Parameter(torch.empty(rows, cols, dtype=params_dtype), requires_grad=False)


def test_public_ngram_receiver_keeps_int8_row_scales():
    config = SimpleNamespace(ngram_size=2, heads_per_ngram=2, vocab_size=32,
        ngram_vocab_size_base=16, make_ngram_vocab_size_divisible_by=1,
        eos_token_id=1, ple_offload_embedding=False,
        compression_config={"mixed_precision":{"ngram_embedding":{"weight_bits":8}}})
    quant = SimpleNamespace(get_name=lambda:"slimquant_w4a8_marlin", ignore=[])
    cls = ple.N.Qwen4ExpNGramEmbedding
    module = cls.__new__(cls)
    with patch.object(ple, "VocabParallelEmbedding", FakeVocab), \
         patch.object(ple, "is_dp_attention_enabled", return_value=False), \
         patch.object(ple, "_use_attn_tp_ngram", return_value=False), \
         patch.object(ple, "_ple_etp_reduce_scatter_enabled", return_value=False):
        ple.NGramMethods.__init__(module, config, 8, quant_config=quant, prefix="model.layers.0.ple.ple_embedding")
    assert isinstance(module, torch.nn.Module)
    assert module.ngram_embedding.weight.dtype == torch.int8
    assert module.ngram_embedding.weight_scale.shape == (module.ngram_embedding.num_embeddings_per_partition, 1)


def test_local_routing_consume_once_and_scope_cleanup():
    experts = SimpleNamespace(quant_method=SimpleNamespace())
    mlp = SimpleNamespace(experts=experts)
    q = torch.ones(3, 4, dtype=torch.int8)
    scale = torch.ones(3, 1)
    state={"mlp":mlp,"prequant":(q,scale),"consumed":False}
    seen=[]
    def apply(layer, output, **kwargs):
        seen.append(kwargs)
        return "done"
    experts.quant_method.apply=apply
    token=Q._ffn.set(state)
    try:
        assert Q.around_moe_core(lambda *_:None, experts, "dispatch") == "done"
        assert seen[0]["i_q"] is q
        assert seen[0]["i_s"] is scale
        assert seen[0]["use_int8_prequant"] is True
        try:
            Q.around_moe_core(lambda *_:None, experts, "dispatch")
        except RuntimeError as error:
            assert "twice" in str(error)
        else:
            raise AssertionError("second INT8 consumption was allowed")
    finally:
        Q._ffn.reset(token)
    assert Q._ffn.get() is None


def test_ffn_disabled_fallback_and_exception_clear_context():
    batch = SimpleNamespace(qwen4_compact_prefill=False)
    layer = SimpleNamespace(mlp=object())
    def original(l, x, fb):
        assert Q._ffn.get() is None
        return x
    with patch.object(Q.envs.SGLANG_ENABLE_DP_GATHER_INT8, "get", return_value=False):
        assert Q.around_ffn(original, layer, "hidden", batch) == "hidden"
    batch.qwen4_compact_prefill = True
    def failing(l, x, fb):
        assert Q._ffn.get()["mlp"] is layer.mlp
        raise ValueError("kernel failure")
    with patch.object(Q.envs.SGLANG_QWEN4_LOCAL_ROUTING, "get", return_value=True), \
         patch.object(Q.envs.SGLANG_ENABLE_DP_GATHER_INT8, "get", return_value=False), \
         pytest.raises(ValueError, match="kernel failure"):
        Q.around_ffn(failing, layer, "hidden", batch)
    assert Q._ffn.get() is None


def test_decode_replay_restores_capture_geometry_after_compact_prefill():
    runner=SimpleNamespace(require_mlp_tp_gather=True,
        model_runner=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen4_exp"))),
        _capture_graph_size=lambda **kw:kw["bs"])
    fb=SimpleNamespace(input_ids=torch.zeros(12), global_dp_buffer_len=48,
        dp_padding_mode=SimpleNamespace(is_max_len=lambda:True), global_num_tokens_cpu=[12]*4)
    result=(fb,None,None)
    assert Q.after_capture_prepare(result,runner,3,num_tokens=12) is result
    runner._replay_graph_key=SimpleNamespace(size=3)
    seen=[]
    with patch.object(Q.DP,"set_dp_buffer_len",side_effect=lambda *args:seen.append(args)), \
         patch.object(Q.DP,"set_is_extend_in_batch",side_effect=lambda value:seen.append(value)):
        Q.after_load_batch(None,runner,None)
    assert seen==[(48,12,True,[12]*4),False]


@pytest.mark.parametrize("gatherv", [False, True])
def test_compact_layout_uses_official_gatherv_switch(gatherv):
    from sglang.srt.configs.qwen4_exp import Qwen4ExpTextConfig
    config = Qwen4ExpTextConfig.__new__(Qwen4ExpTextConfig)
    runner = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(get_text_config=lambda:config)),
        is_draft_worker=False)
    batch = SimpleNamespace(dp_pure_prefill=True, spec_info=None)
    parallel = SimpleNamespace(tp_size=4, attn_dp_size=4, moe_ep_size=4,
        attn_tp_size=1, attn_cp_size=1, attn_dp_rank=0)
    execution = SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(
        prefill=SimpleNamespace(backend="disabled"))))
    with patch.object(Q.envs.SGLANG_QWEN4_COMPACT_PREFILL, "get", return_value=True), \
         patch.object(Q.envs.SGLANG_QWEN4_PREFILL_DIAGNOSTICS, "get", return_value=False), \
         patch.object(Q.DP, "_USE_DP_GATHERV", gatherv), \
         patch.object(Q.DP, "world_dp_gather_enabled", return_value=False), \
         patch.object(Q, "get_parallel", return_value=parallel), \
         patch("sglang.srt.layers.moe.utils.get_moe_a2a_backend", return_value=SimpleNamespace(is_none=lambda:True)), \
         patch("sglang.srt.layers.moe.utils.is_tbo_enabled", return_value=False), \
         patch("sglang.srt.runtime_context.get_exec", return_value=execution):
        result = Q.compact_layout(batch, runner, [17, 0, 4, 0])
    assert result == ([17, 1, 4, 1] if gatherv else None)
    assert batch.qwen4_compact_prefill is gatherv


def test_scheduler_public_prepare_reaches_gather_hook_on_idle_rank():
    from contextlib import ExitStack
    from hcu_sglang import prefill_graph_sync as P
    from sglang.srt.plugins.hook_registry import _wrap_fn, HookType
    mode = Q.ForwardMode
    batch = SimpleNamespace(forward_mode=mode.IDLE, reqs=[])
    parallel = SimpleNamespace(num_dp_ranks=4, attn_tp_size=1, attn_cp_size=1,
        enable_cp_tp_group_sharing=False, tp_group=SimpleNamespace(cpu_group=None))
    def gather(info, **kwargs):
        info.tp0_info_cpu = torch.zeros(4,12,dtype=torch.long)
        info.tp0_info_cpu[:,5] = torch.tensor([mode.EXTEND.value]+[mode.IDLE.value]*3)
        info.global_num_tokens = [17,0,0,0]
        info.global_num_tokens_for_logprob = [1,0,0,0]
        info.global_cp_num_tokens = [17,0,0,0]
        info.can_run_decode_cuda_graph = False
        info.can_run_draft_cuda_graph = False
        info.can_run_prefill_cuda_graph = False
    preparer = SimpleNamespace(prepare_all_gather=lambda _:(False,mode.IDLE.value),
        compute_output=lambda _:(None,mode.EXTEND))
    wrapped = _wrap_fn(lambda *_a,**_k:pytest.fail("bypassed HCU gather update"),
        P.around__update_gather_batch, HookType.AROUND)
    wrapped = _wrap_fn(wrapped, Q.after_update_gather, HookType.AFTER)
    values = {"get_parallel":lambda:parallel, "world_dp_gather_enabled":lambda:False,
        "require_mlp_tp_gather":True, "_ENABLE_METRICS_DP_ATTENTION":False,
        "_local_decode_cuda_graph_vote":lambda **_:False,
        "_spec_input_cuda_graph_compatible":lambda _:False,
        "check_cuda_graph_backend":lambda *_:False,
        "_local_prefill_cuda_graph_vote":lambda **_:False,
        "TboDPAttentionPreparer":lambda:preparer,
        "should_skip_scheduler_all_gather":lambda _:False,
        "_local_attention_pairs":lambda _:0,
        "_update_gather_batch":wrapped}
    with ExitStack() as stack:
        for name,value in values.items():
            stack.enter_context(patch.object(P.N,name,value))
        stack.enter_context(patch.object(P,"is_hcu",return_value=True))
        stack.enter_context(patch.object(P.MLPSyncBatchInfo,"all_gather",gather))
        stack.enter_context(patch.object(P.N.envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION,"get",return_value=False))
        result = P.prepare_mlp_sync_batch_raw(batch,
            SimpleNamespace(spec_algorithm=None,model_config=None),lambda:batch,
            False,True,False,set())
    assert result is batch
    assert batch.dp_pure_prefill is True
    assert batch.global_num_tokens == [17,0,0,0]


def test_idle_public_init_and_padding_materialize_three_axis_positions():
    from dataclasses import fields, MISSING
    from hcu_sglang import prefill_graph_sync as P, forward_batch as F
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.forward_batch import ForwardBatch
    values = {f.name:None for f in fields(P.ScheduleBatch)
        if f.default is MISSING and f.default_factory is MISSING}
    empty = torch.empty(0,dtype=torch.long)
    values.update(reqs=[],forward_mode=Q.ForwardMode.IDLE,input_ids=empty,
        seq_lens=empty,seq_lens_cpu=empty,orig_seq_lens=empty,seq_lens_sum=0,
        req_pool_indices=empty,req_pool_indices_cpu=empty,out_cache_loc=empty,
        dp_pure_prefill=True)
    batch = P.ScheduleBatch(**values)
    backend = SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda:1,
        get_cpu_graph_seq_len_fill_value=lambda:1)
    runner = SimpleNamespace(server_args=SimpleNamespace(disaggregation_mode=None),
        is_draft_worker=False,device="cpu",lora_manager=None,attn_backend=backend,
        model_config=SimpleNamespace(model_is_mrope=True),
        kv_index_translator=SimpleNamespace(own_plan=lambda _:SimpleNamespace(bind=lambda *a,**k:None)))
    with patch.object(F.N,"_maybe_build_forward_token_modalities",return_value=None), \
         patch.object(F.N,"enable_num_token_non_padded",return_value=False), \
         patch.object(ForwardBatch,"init_mlp_sync_metadata",F.ForwardBatchMethods.init_mlp_sync_metadata):
        ret = F.ForwardBatchMethods.init_new.__func__(ForwardBatch,batch,runner,
            capture_hidden_mode=F.N.CaptureHiddenMode.NULL,return_hidden_states_before_norm=False)
    assert ret.mrope_positions.shape == (3,0)
    assert ret.dp_pure_prefill is True
    ret._pad_inputs_to_size(runner,1,1)
    torch.testing.assert_close(ret.mrope_positions,torch.zeros(3,1,dtype=torch.long))
    assert batch.input_ids.numel() == 0


@pytest.mark.parametrize("mtp", [False, True])
def test_public_weight_loader_accepts_fused_expert_scales(mtp):
    from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
    cls = Qwen4ExpForCausalLMMTP if mtp else ple.N.Qwen4ExpForConditionalGeneration
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=2, encoder_only=False, tie_word_embeddings=False)
    model.language_model_only = False
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.num_fused_shared_experts = 0
    model.model = torch.nn.Module()
    model.model.start_layer, model.model.end_layer = 0, 1
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = model.model.layers[0]
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = torch.nn.Module()
    seen = []
    for key in ("w13_weight_scale", "w2_weight_scale", "w13_weight", "w2_weight"):
        param = torch.nn.Parameter(torch.empty(1), requires_grad=False)
        param.weight_loader = lambda p, v, name, shard_id, expert_id: seen.append((name, shard_id, expert_id, v.clone()))
        layer.mlp.experts.register_parameter(key, param)
    prefix = ("mtp." if mtp else "model.language_model.") + "layers.0.mlp.experts."
    gate = torch.arange(16.).reshape(2, 8, 1)
    down = torch.arange(8.).reshape(2, 4, 1)
    original = cls.load_weights
    while hasattr(original, "__wrapped__"):
        original = original.__wrapped__
    inputs = [(prefix + "gate_up_proj.weight_scale", gate),
              (prefix + "down_proj.weight_scale", down),
              (prefix + "gate_up_proj", gate), (prefix + "down_proj", down)]
    wrapper = Q.around_mtp_load_weights if mtp else Q.around_load_weights
    loaded = wrapper(original, model, iter(inputs))
    assert len(seen) == 12
    assert len(loaded) == 4
    for name, shard, expert_id, value in seen:
        source = gate if "w13" in name else down
        if shard == "w1":
            source = source.chunk(2, dim=-2)[0]
        elif shard == "w3":
            source = source.chunk(2, dim=-2)[1]
        torch.testing.assert_close(value, source[expert_id])


def test_mtp_group_norm_and_projection_keep_all_logical_rows():
    from hcu_sglang.models.qwen import qwen4_exp_mtp as M
    module = torch.nn.Module()
    module.hc_count, module.hidden_size = 2, 4
    config = SimpleNamespace(hidden_size=4, rms_norm_eps=1e-6)
    M._init_pre_fc_norms(module, config)
    module.pre_fc_norm_embedding.weight.data.zero_()
    module.pre_fc_norm_hidden.weight.data.zero_()
    rows = []
    class Projection(torch.nn.Module):
        def forward(self, x):
            assert x.ndim == 2
            rows.append(x.shape[0])
            return 2 * x, None
    module.fc_embedding = Projection()
    module.fc_hidden = Projection()
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1
    h = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) + 1
    h[:, 1] *= 100
    result = M._fuse_residual_linear_shared(module, x, h.flatten(1))
    norm = lambda a: a * torch.rsqrt(a.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    expected = 2 * norm(x)[:, None, :] + 2 * norm(h)
    torch.testing.assert_close(result, expected.flatten(1))
    assert rows == [3, 6]


def test_prefill_address_plan_reads_current_layer_data():
    from hcu_sglang.models.qwen import qsa as S
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnMetadata
    buffer = torch.arange(48.).reshape(6,1,8)
    pool = SimpleNamespace(get_qsa_compressed_k_buffer=lambda _:buffer)
    lens = torch.tensor([8,4], dtype=torch.int32)
    ids = torch.tensor([0]*8+[1]*4, dtype=torch.int32)
    table = torch.stack((torch.arange(8), torch.arange(8,16))).to(torch.int32)
    meta = S.M.QSAIndexerMetadata(lens, ids, table, torch.arange(12), pool, 4, 2)
    result = QwenSparseAttnMetadata(lens, ids, table, meta)
    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda:True),
        seq_lens_cpu=lens, extend_seq_lens_cpu=[8,4], positions=torch.tensor(list(range(8))+list(range(4))))
    with patch.object(S,"get_parallel",return_value=SimpleNamespace(attn_cp_size=1)):
        cached = S.after_metadata(result,None,fb).indexer_metadata
    original = S.M.QSAIndexerMetadata.get_prefill_mqa_inputs
    while hasattr(original,"__wrapped__"):
        original = original.__wrapped__
    reference = original(meta,0,fb.positions)
    actual = S.around_prefill_inputs(original,cached,0,fb.positions)
    for a,b in zip(actual,reference):
        torch.testing.assert_close(a,b)
    buffer.add_(100)
    updated = S.around_prefill_inputs(original,cached,1,fb.positions)
    torch.testing.assert_close(updated[0],actual[0]+100)


@pytest.mark.parametrize("draft", [None, SimpleNamespace(), SimpleNamespace(draft_swa_window=64)])
def test_cache_builder_accepts_official_draft_worker_without_swa_extension(draft):
    from hcu_sglang.mem_cache import cache_retention as C, cache_builder as B
    scheduler = SimpleNamespace(draft_worker=draft)
    def init(s):
        return C.around_build_kv_cache(lambda **kw:None, model_runner="runner")
    with patch.object(C,"is_hcu",return_value=True), patch.object(B,"build_kv_cache",side_effect=lambda **kw:kw):
        result = C.around_scheduler_init(init,scheduler)
    assert result == {"model_runner":"runner", "draft_swa_window":getattr(draft,"draft_swa_window",0)}
    assert C._SCHEDULER.get() is None
