from __future__ import annotations

import types
import dis
import builtins
import inspect
from dataclasses import replace
from contextlib import contextmanager, nullcontext, ExitStack
from sglang.srt.layers.moe.utils import speculative_moe_backend_context, speculative_moe_a2a_backend_context
from typing import List, Optional
import sglang.srt.models.deepseek_v4_dspark as M
import sglang.srt.speculative.dspark_components.dspark_worker_v2 as W
import sglang.srt.speculative.dspark_components.dspark_kv_inject as I
from sglang.srt.speculative.dspark_components import dspark_config as C

globals().update({name: value for name, value in vars(M).items() if not name.startswith("__")})
globals().update({name: value for name, value in vars(I).items() if not name.startswith("__")})
globals().update({name: value for name, value in vars(W).items() if not name.startswith("__")})
from sglang.srt.distributed.parallel_state import model_parallel_is_initialized
from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import CommitInjectLayoutResult
from hcu_sglang.models.deepseek.current_v4.hcu_backend.dspark_attn_metadata import swa_ring_rows
from hcu_sglang.models.deepseek.current_v4.execution import lazy_boundary as hc_boundary
from sglang.srt.utils import is_cuda
_is_cuda = is_cuda()
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import inverse_transform_scale_ue8m0, transform_scale_ue8m0
from hcu_sglang.models.deepseek.current_v4.execution.core import DeepseekV4PlatformDecoderLayer
from hcu_sglang.models.deepseek.current_v4.execution.platform_stage import NonCudaDSparkMixin
from hcu_sglang.models.deepseek.current_v4.dspark import DSparkAttention
from hcu_sglang.utils.detection import is_hcu
_is_hcu = is_hcu()

def build_verify_generation_result(target_verify, **fields):
    return GenerationBatchResult(
        expert_distribution_metrics=getattr(target_verify, "expert_distribution_metrics", None),
        **fields,
    )

def build_verify_epilogue(factory, *, simulate_fold, **kwargs):
    if "simulate_fold" in inspect.signature(factory).parameters:
        return factory(simulate_fold=simulate_fold, **kwargs)
    if simulate_fold:
        raise NotImplementedError("The installed DSpark epilogue does not support simulate_fold")
    return factory(**kwargs)

def build_ring_commit_inject_layout(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_pos_offsets: torch.Tensor,
    commit_lens: torch.Tensor,
    stride: int,
    ring_stride: int,
) -> CommitInjectLayoutResult:
    """Per-request SWA ring counterpart of build_commit_inject_layout.

    Paged injection translates the verify tokens' full cache locs through
    ``full_to_swa_mapping``; a per-request ring is addressed directly by
    `swa_ring_rows`, so compute the ring row here instead. Uncommitted tokens
    (col >= commit_len) get loc = -1 and are skipped by the scatter. All ops are
    static-shape (CUDA-graph safe).
    """
    bs = req_pool_indices.shape[0]
    device = req_pool_indices.device
    positions_2d = prefix_lens.unsqueeze(1) + block_pos_offsets[:stride]
    positions = positions_2d.reshape(-1).to(torch.int64)
    state_slot = (
        req_pool_indices.to(torch.int64).view(-1, 1).expand(bs, stride).reshape(-1)
    )
    loc = swa_ring_rows(state_slot, positions, ring_stride)
    col = torch.arange(stride, device=device).view(1, -1)
    committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
    swa_loc = torch.where(committed, loc, torch.full_like(loc, -1)).to(torch.int32)
    return CommitInjectLayoutResult(swa_loc=swa_loc, positions=positions)

def resolve_single_owner_pp_rank(
    *, target_layer_ids: List[int], num_hidden_layers: int, pp_size: int
) -> Optional[int]:
    from sglang.srt.distributed.utils import get_pp_indices

    if not target_layer_ids:
        return None
    for pp_rank in range(pp_size):
        start_layer, end_layer = get_pp_indices(
            num_hidden_layers=num_hidden_layers,
            pp_rank=pp_rank,
            pp_size=pp_size,
        )
        if all(start_layer <= layer_id < end_layer for layer_id in target_layer_ids):
            return pp_rank
    return None

def use_empty_draft_model_for_pp_prefill(
    *,
    disaggregation_mode: str,
    pp_rank: int,
    pp_size: int,
    target_layer_ids: List[int],
    num_hidden_layers: int,
) -> bool:
    owner_pp_rank = resolve_single_owner_pp_rank(
        target_layer_ids=target_layer_ids,
        num_hidden_layers=num_hidden_layers,
        pp_size=pp_size,
    )
    return (
        disaggregation_mode == "prefill"
        and pp_size > 1
        and owner_pp_rank == pp_size - 1
        and pp_rank != owner_pp_rank
    )

def _is_context_only_pp_prefill_rank(
    *, disaggregation_mode: str, pp_rank: int, pp_size: int
) -> bool:
    return disaggregation_mode == "prefill" and pp_size > 1 and pp_rank < pp_size - 1

class _BlockFp8LinearSlice(nn.Module):
    def __init__(
        self,
        *,
        source: ReplicatedLinear,
        feature_indices: List[int],
        feature_width: int,
    ) -> None:
        super().__init__()
        quant_method = source.quant_method
        if not (
            isinstance(quant_method, Fp8LinearMethod)
            and quant_method.block_quant
            and not quant_method.use_mxfp8
            and not quant_method.use_marlin
        ):
            raise ValueError(
                "DSpark block-FP8 projection slice requires a non-MXFP8 "
                "block-quantized Fp8LinearMethod."
            )

        block_k = int(quant_method.weight_block_size[1])
        if feature_width % block_k != 0:
            raise ValueError(
                f"DSpark feature width {feature_width} must align to FP8 "
                f"block_k={block_k}."
            )
        blocks_per_feature = feature_width // block_k
        device = source.weight.device
        weight_columns = torch.cat(
            [
                torch.arange(
                    feature_index * feature_width,
                    (feature_index + 1) * feature_width,
                    device=device,
                )
                for feature_index in feature_indices
            ]
        )
        scale_columns = torch.cat(
            [
                torch.arange(
                    feature_index * blocks_per_feature,
                    (feature_index + 1) * blocks_per_feature,
                    device=device,
                )
                for feature_index in feature_indices
            ]
        )

        self.quant_method = quant_method
        self.weight = nn.Parameter(
            source.weight.detach().index_select(1, weight_columns).contiguous(),
            requires_grad=False,
        )
        source_scale = source.weight_scale_inv.detach()
        scale_is_ue8m0 = source.weight_scale_inv.format_ue8m0
        if scale_is_ue8m0:
            source_scale = inverse_transform_scale_ue8m0(
                source_scale, mn=source.weight.shape[0]
            )
        local_scale = source_scale.index_select(1, scale_columns).contiguous()
        if scale_is_ue8m0:
            local_scale = transform_scale_ue8m0(local_scale, mn=source.weight.shape[0])
        self.weight_scale_inv = nn.Parameter(
            local_scale,
            requires_grad=False,
        )
        self.weight_scale_inv.format_ue8m0 = scale_is_ue8m0
        self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, hidden_states, bias=None)

class DSparkV4StageMixin:
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        stage_id: int,
        num_stages: int,
        num_target_layers: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        hc_stats_stream: Optional[torch.cuda.Stream] = None,
        moe_routed_quant_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config=_dspark_stage_config(config),
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            is_nextn=True,
            boundary_pre_from_prev=config.hc_pre_from_prev_sublayer,
            boundary_mhc_fusions=False,
            alt_streams=alt_streams,
            hc_stats_stream=hc_stats_stream,
            moe_routed_quant_stream=moe_routed_quant_stream,
        )
        # Draft stage IDs overlap target layers, but draft experts are not EPLB-reordered.
        self.mlp.experts._skip_expert_location = True
        self.stage_id = stage_id
        self.dim = config.hidden_size

        if stage_id == 0:
            if num_target_layers <= 0:
                raise ValueError(
                    "DSpark needs target layers for the target-hidden projection."
                )
            self.main_proj = ReplicatedLinear(
                config.hidden_size * num_target_layers,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("main_proj", prefix),
            )
            self.main_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if stage_id == num_stages - 1:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.hc_pre_from_prev_sublayer:
                # V4.1 collapses the head with the last FFN's pre-mix; the
                # checkpoint carries no hc_head_* tensors for the stages.
                self.hc_head_fn = self.hc_head_base = self.hc_head_scale = None
            else:
                (
                    self.hc_head_fn,
                    self.hc_head_base,
                    self.hc_head_scale,
                ) = make_hc_head_params(config.hc_mult, config.hidden_size)

    def _build_self_attn(
        self,
        *,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[List[torch.cuda.Stream]],
        compress_ratio_override: Optional[int],
    ) -> nn.Module:
        del compress_ratio_override
        return DSparkAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_streams=alt_streams,
        )

class DSparkV4PlatformStage(
    NonCudaDSparkMixin, DSparkV4StageMixin, DeepseekV4PlatformDecoderLayer
):
    """Non-CUDA draft with platform mHC operations."""

DSparkV4Stage = DSparkV4PlatformStage

class _ModelMethods:
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super(DeepseekV4ForCausalLMDSpark, self).__init__()
        self.config = config
        self.quant_config = quant_config
        self.num_fused_shared_experts = (
            0 if is_shared_experts_fusion_disabled() else config.n_shared_experts
        )

        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if not dspark_config.require_markov():
            raise ValueError(
                "DSpark V4 draft requires markov_rank > 0, "
                f"got markov_rank={dspark_config.markov_rank}."
            )
        self.gamma = int(
            dspark_config.resolve_gamma(default=int(config.num_hidden_layers))
        )
        self.sample_from_anchor = get_dspark_sample_from_anchor(config)
        self.block_size = self.gamma
        if dspark_config.target_layer_ids is not None:
            self.num_stages = len(dspark_config.target_layer_ids)
        else:
            self.num_stages = int(getattr(config, "num_nextn_predict_layers", 1) or 1)

        target_num_layers = (
            int(dspark_config.num_target_layers)
            if dspark_config.num_target_layers is not None
            else int(getattr(config, "num_hidden_layers", 1))
        )
        if dspark_config.target_layer_ids is not None:
            self.num_target_features = len(dspark_config.target_layer_ids)
        else:
            self.num_target_features = target_num_layers

        self.start_layer = 0
        self.end_layer = self.num_stages
        if model_parallel_is_initialized():
            parallel = get_parallel()
            pp_rank = parallel.pp_rank
            pp_size = parallel.pp_size
        else:
            pp_rank = 0
            pp_size = 1
        self.is_lifecycle_only = _is_hcu and use_empty_draft_model_for_pp_prefill(
            disaggregation_mode=get_disagg().disaggregation_mode,
            pp_rank=pp_rank,
            pp_size=pp_size,
            target_layer_ids=[
                int(layer_id) for layer_id in (dspark_config.target_layer_ids or [])
            ],
            num_hidden_layers=target_num_layers,
        )
        self.hc_mult = int(config.hc_mult)
        self.norm_eps = float(config.rms_norm_eps)
        self.hc_eps = float(config.hc_eps)
        self.hc_pre_from_prev_sublayer = bool(
            getattr(config, "hc_pre_from_prev_sublayer", False)
        )
        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self._partial_feature_indices: tuple[int, ...] = ()
        self._partial_main_proj: Optional[_BlockFp8LinearSlice] = None
        self._use_fp32_lm_head = envs.SGLANG_DSPARK_FP32_LM_HEAD.get()
        self._opt_markov_w2_tp_shard = envs.SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD.get()
        if self.is_lifecycle_only:
            self.stages = nn.ModuleList()
            self.markov_head = None
            self.confidence_head = None
            logger.info(
                "DSpark PP rank %s uses a lifecycle-only draft model; all target "
                "features are owned by final PP rank %s.",
                pp_rank,
                pp_size - 1,
            )
            return

        use_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and envs.SGLANG_DSPARK_ENABLE_MULTI_STREAM.get()
            and torch.cuda.is_available()
        )
        self.alt_streams: Optional[List[torch.cuda.Stream]] = (
            [torch.cuda.Stream()] if use_multi_stream else None
        )
        self.moe_routed_quant_stream = (
            torch.cuda.Stream()
            if use_multi_stream
            and torch.version.cuda is not None
            and getattr(config, "hc_pre_from_prev_sublayer", False)
            else None
        )
        self.hc_stats_stream = (
            torch.cuda.Stream()
            if use_multi_stream
            and torch.version.cuda is not None
            and get_platform().is_blackwell
            and getattr(config, "hc_pre_from_prev_sublayer", False)
            else None
        )
        with (
            layer_stack(final_read=hc_boundary.FinalRead())
            if _is_cuda and not _is_hcu
            else nullcontext()
        ):
            self.stages = nn.ModuleList(
                [
                    DSparkV4Stage(
                        config=config,
                        layer_id=stage_id,
                        stage_id=stage_id,
                        num_stages=self.num_stages,
                        num_target_layers=self.num_target_features,
                        quant_config=quant_config,
                        prefix=add_prefix(f"stages.{stage_id}", prefix),
                        alt_streams=self.alt_streams,
                        hc_stats_stream=self.hc_stats_stream,
                        moe_routed_quant_stream=self.moe_routed_quant_stream,
                    )
                    for stage_id in range(self.num_stages)
                ]
            )
        self.markov_head = DSparkV4MarkovHead(
            vocab_size=int(config.vocab_size),
            markov_rank=int(dspark_config.markov_rank),
            is_dsv41=getattr(config, "model_type", None) == "deepseek_v41",
        )
        self.confidence_head = build_dspark_v4_confidence_head(
            config=config, markov_rank=int(dspark_config.markov_rank)
        )
        if self.uses_own_vocab_modules:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embed_tokens", prefix),
                enable_tp=not is_dp_attention_enabled(),
            )
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_parallel().enable_dp_lm_head,
            )
        else:
            self.embed_tokens: Optional[nn.Module] = None
            self.lm_head: Optional[nn.Module] = None
        if self.lm_head is not None:
            self.markov_head.configure_tp_shard(lm_head=self.lm_head)

    def prune_to_ctx_projection(self) -> None:
        if vars(self).get("is_lifecycle_only", False):
            return
        stage0 = self.stages[0]
        projection_stage = nn.Module()
        projection_stage.main_proj = stage0.main_proj
        projection_stage.main_norm = stage0.main_norm
        self.stages = nn.ModuleList([projection_stage])
        self.markov_head = None
        self.confidence_head = None
        self.embed_tokens = None
        self.lm_head = None
        del stage0
        torch.cuda.empty_cache()

    def project_target_hidden(self, main_hidden: torch.Tensor) -> torch.Tensor:
        projected, _ = self.stages[0].main_proj(main_hidden)
        return self.stages[0].main_norm(projected)

    def prepare_target_hidden_partial(self, feature_indices: List[int]) -> None:
        feature_indices = [int(index) for index in feature_indices]
        main_proj = self.stages[0].main_proj
        quant_method = main_proj.quant_method
        self._partial_feature_indices = tuple(feature_indices)
        if not (
            isinstance(quant_method, Fp8LinearMethod)
            and quant_method.block_quant
            and not quant_method.use_mxfp8
            and not quant_method.use_marlin
        ):
            self._partial_main_proj = None
            logger.warning(
                "DSpark partial projection cannot slice quant method %s; "
                "falling back to the full-K projection.",
                type(quant_method).__name__,
            )
            return
        self._partial_main_proj = _BlockFp8LinearSlice(
            source=main_proj,
            feature_indices=feature_indices,
            feature_width=int(self.config.hidden_size),
        )
        logger.info(
            "DSpark block-FP8 partial projection uses feature columns %s "
            "(local K=%s, full K=%s).",
            feature_indices,
            len(feature_indices) * int(self.config.hidden_size),
            int(main_proj.weight.shape[1]),
        )

    def project_target_hidden_partial(
        self, main_hidden: torch.Tensor, feature_indices: list[int]
    ) -> torch.Tensor:
        if not feature_indices:
            raise ValueError("feature_indices must be non-empty.")
        feature_indices = [int(index) for index in feature_indices]
        if min(feature_indices) < 0 or max(feature_indices) >= self.num_target_features:
            raise ValueError(
                "DeepSeek-V4 DSpark feature_indices out of range: "
                f"{feature_indices=} {self.num_target_features=}."
            )

        hidden_size = int(self.config.hidden_size)
        expected = len(feature_indices) * hidden_size
        if main_hidden.ndim != 2 or int(main_hidden.shape[-1]) != expected:
            raise ValueError(
                "DeepSeek-V4 DSpark partial main_hidden feature dim mismatch. "
                f"Expected shape [N, {expected}] for {feature_indices=}, "
                f"but got shape={tuple(main_hidden.shape)}."
            )

        if (
            self._partial_main_proj is not None
            and tuple(feature_indices) == self._partial_feature_indices
        ):
            return self._partial_main_proj(main_hidden)

        local_features = main_hidden.view(
            main_hidden.shape[0], len(feature_indices), hidden_size
        )
        full_features = main_hidden.new_zeros(
            main_hidden.shape[0], self.num_target_features, hidden_size
        )
        feature_index = torch.tensor(
            feature_indices, dtype=torch.long, device=main_hidden.device
        )
        full_features.index_copy_(1, feature_index, local_features)
        projected, _ = self.stages[0].main_proj(
            full_features.view(main_hidden.shape[0], -1)
        )
        return projected

    def write_target_hidden_kv(
        self,
        *,
        main_hidden: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        main_x = self.project_target_hidden(main_hidden)
        self._write_context_hidden_kv(
            main_x=main_x,
            swa_loc=swa_loc,
            positions=positions,
            pool=pool,
        )

    def write_projected_context_kv(
        self,
        *,
        projected_context: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        main_x = self.stages[0].main_norm(projected_context)
        self._write_context_hidden_kv(
            main_x=main_x,
            swa_loc=swa_loc,
            positions=positions,
            pool=pool,
        )

    def _write_context_hidden_kv(
        self,
        *,
        main_x: torch.Tensor,
        swa_loc: torch.Tensor,
        positions: torch.Tensor,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        swa_loc = swa_loc.to(torch.int32)
        kvs = CommitKvProj.execute(
            main_x=main_x,
            wkv_linears=[stage.self_attn.wkv for stage in self.stages],
            # The FlashMLA writer reads an explicit KV row stride, so views are fine.
            allow_strided_output=(
                get_platform().is_blackwell
                and not is_unified_kv_triton()
                and not pool.uniform_fp8
            ),
        )
        # Under unified_kv the swa_kv_pool is None; the caller passes a unified
        # ring loc (state_slot * ring + pos % ring, -1 for uncommitted) so the
        # store just needs to target the bf16 ring instead of the fp8 flashmla
        # buffer. Same swa_loc/positions contract either way.
        store_kv = (
            pool.set_unified_key_buffer_radix_fused_norm_rope
            if is_unified_kv_triton()
            else pool.set_swa_key_buffer_radix_fused_norm_rope
        )
        for stage, kv in zip(self.stages, kvs):
            attn = stage.self_attn
            store_kv(
                layer_id=attn.layer_id,
                swa_loc=swa_loc,
                kv=kv,
                kv_weight=attn.kv_norm.weight.data,
                eps=attn.eps,
                freqs_cis=attn.freqs_cis,
                positions=positions,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if _is_hcu or not _is_cuda:
            return self._forward_impl(input_ids, positions, forward_batch, input_embeds)
        try:
            return self._forward_impl(input_ids, positions, forward_batch, input_embeds)
        finally:
            residual_batch.clear(forward_batch)

    def _forward_impl(self, input_ids, positions, forward_batch, input_embeds):
        if input_embeds is None:
            input_embeds = self.forward_embed(input_ids)
        x = input_embeds
        pre = None
        if _is_cuda and not _is_hcu:
            residual_batch.start(forward_batch)
        # Draft stages do not have recorder layer indices.
        with get_global_expert_distribution_recorder().disable_this_region():
            for stage in self.stages:
                if _is_cuda and not _is_hcu:
                    x = stage(positions, x, forward_batch)
                else:
                    x, pre = stage(positions, x, forward_batch, pre)
        if _is_cuda and not _is_hcu:
            x = residual_batch.read_output(
                x, forward_batch, hc_boundary.FinalRead(), None
            )
            if self.hc_pre_from_prev_sublayer:
                pre, x = x.pre, x.residual
        if self.hc_pre_from_prev_sublayer:
            from sglang.kernels.ops.layernorm.mhc import hc_combine

            x = hc_combine(x.flatten(1).float(), pre, self.hc_mult, x.dtype)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=x)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        if vars(self).get("is_lifecycle_only", False):
            return
        params_dict = dict(self.named_parameters())
        loaded_params = set()

        # DSpark draft weights use the same checkpoint-side ``.scale`` name as
        # the target. Match the parameter layout created by the active quant
        # method: per-channel FP8 exposes ``weight_scale`` while block FP8 uses
        # ``weight_scale_inv``.
        self._dspark_scale_suffix = (
            "weight_scale"
            if any(
                key.endswith((".weight_scale", "_weight_scale"))
                for key in params_dict
            )
            else "weight_scale_inv"
        )

        weights = _dequant_fp8_wo_a_streaming(weights)

        stacked_params_mapping = DEEPSEEK_V4_STACKED_PARAMS_MAPPING
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=(self.config.n_routed_experts + self.num_fused_shared_experts),
        )

        for name, loaded_weight in weights:
            mapped = (
                self._remap_dspark_weight_name_npu(name)
                if _is_npu
                else self._remap_dspark_weight_name(name)
            )
            if mapped is None:
                continue
            if self.num_fused_shared_experts > 0 and ".mlp.shared_experts." in mapped:
                mapped = mapped.replace(
                    ".mlp.shared_experts.",
                    f".mlp.experts.{self.config.n_routed_experts}.",
                )
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped:
                    continue
                candidate = mapped.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(candidate)
                break
            else:
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in mapped:
                        continue
                    candidate = mapped.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    param = params_dict[candidate]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        candidate,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(candidate)
                    break
                else:
                    if mapped not in params_dict:
                        logger.warning(
                            "DSpark V4 draft: unexpected weight %r -> %r", name, mapped
                        )
                        continue
                    param = params_dict[mapped]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(mapped)

        self._assert_confidence_head_loaded(
            params_dict=params_dict, loaded_params=loaded_params
        )

    def _remap_dspark_weight_name(self, name: str) -> Optional[str]:
        if name.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
            return None
        if "rotary_emb.inv_freq" in name:
            return None

        if not name.startswith("mtp."):
            return None
        parts = name.split(".", 2)
        if len(parts) < 3:
            return None
        stage_id, rest = parts[1], parts[2]

        if rest.startswith("markov_head."):
            rest = rest.replace("markov_head.embed.", "markov_head.markov_w1.", 1)
            rest = rest.replace("markov_head.head.", "markov_head.markov_w2.", 1)
            return f"markov_head.{rest[len('markov_head.') :]}"

        if rest.startswith("confidence_head."):
            if self.confidence_head is None:
                return None
            return f"confidence_head.{rest[len('confidence_head.') :]}"

        mapped_rest = rest
        mapped_rest = mapped_rest.replace("attn.", "self_attn.", 1)
        mapped_rest = mapped_rest.replace("ffn.", "mlp.", 1)
        mapped_rest = mapped_rest.replace("attn_norm.", "input_layernorm.", 1)
        mapped_rest = mapped_rest.replace("ffn_norm.", "post_attention_layernorm.", 1)
        mapped_rest = mapped_rest.replace(".w1.", ".gate_proj.")
        mapped_rest = mapped_rest.replace(".w2.", ".down_proj.")
        mapped_rest = mapped_rest.replace(".w3.", ".up_proj.")
        mapped_rest = mapped_rest.replace(".gate.tid2eid", ".topk.tid2eid")
        if mapped_rest.endswith(".gate.bias_vl"):
            return None
        mapped_rest = mapped_rest.replace(".gate.bias", ".gate.e_score_correction_bias")
        scale_suffix = getattr(
            self, "_dspark_scale_suffix", "weight_scale_inv"
        )
        # Some loaders normalize checkpoint scale names to weight_scale_inv
        # before the model remapper runs. Per-channel FP8 parameters expose
        # weight_scale, so normalize either input form to
        # the suffix selected from the instantiated draft parameters.
        if mapped_rest.endswith(".weight_scale_inv"):
            mapped_rest = (
                mapped_rest.removesuffix(".weight_scale_inv")
                + "."
                + scale_suffix
            )
        else:
            mapped_rest = mapped_rest.replace(".scale", "." + scale_suffix)
        return f"stages.{stage_id}.{mapped_rest}"

class _InjectorMethods:
    def swa_ring_size(self) -> Optional[int]:
        """The ring length when the draft's SWA KV is a per-request ring."""
        return getattr(
            self.draft_model_runner.token_to_kv_pool, "swa_req_ring_size", None
        )

    def inject_target_hidden(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        state_slot: Optional[torch.Tensor] = None,
        final_pos: Optional[torch.Tensor] = None,
        target_hidden_is_projected: bool = False,
    ) -> Optional[torch.cuda.Event]:
        if target_hidden is None or target_hidden.numel() == 0:
            return None
        device = self.model_runner.device
        cache_loc = cache_loc.to(device=device, dtype=torch.int64, non_blocking=True)
        positions = positions.to(device=device, dtype=torch.int64, non_blocking=True)
        target_hidden = target_hidden.to(device=device, non_blocking=True)
        n_real = positions.shape[0]
        if target_hidden.shape[0] > n_real:
            target_hidden = target_hidden[:n_real]
        if target_hidden.shape[0] != n_real or cache_loc.shape[0] != n_real:
            raise ValueError(
                "DSpark target hidden injection requires one hidden row and cache "
                f"location per position; got hidden_rows={target_hidden.shape[0]}, "
                f"cache_locs={cache_loc.shape[0]}, positions={n_real}."
            )
        if cache_loc_2d is not None:
            cache_loc_2d = cache_loc_2d.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            if cache_loc_2d.numel() != n_real:
                raise ValueError(
                    "DSpark target hidden injection requires cache_loc_2d to cover "
                    f"every position; got cache_loc_2d={cache_loc_2d.numel()}, "
                    f"positions={n_real}."
                )
        if commit_lens is not None:
            commit_lens = commit_lens.to(
                device=device, dtype=torch.int32, non_blocking=True
            )
        if state_slot is not None:
            state_slot = state_slot.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if final_pos is not None:
            final_pos = final_pos.to(
                device=device, dtype=torch.int64, non_blocking=True
            )

        # `cache_loc` arrives in the draft pool's id space (`ids_for`).
        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            if target_hidden_is_projected:
                raise RuntimeError(
                    "Pre-gather target-hidden projection is not supported by the "
                    "DSpark MLA KV injection path."
                )
            self._inject_mla(
                pool=pool,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
            )
            return self._record_cuda_event(device)

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                target_hidden=target_hidden,
                pool=pool,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                target_hidden_is_projected=target_hidden_is_projected,
            )
        return self._record_cuda_event(device)

    def _record_cuda_event(device) -> Optional[torch.cuda.Event]:
        if not torch.cuda.is_available():
            return None
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        return event

    def inject_projected_context(
        self,
        *,
        projected_context: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        state_slot: Optional[torch.Tensor] = None,
        final_pos: Optional[torch.Tensor] = None,
    ) -> None:
        if projected_context is None or projected_context.numel() == 0:
            return
        device = self.model_runner.device
        cache_loc = cache_loc.to(device=device, dtype=torch.int64, non_blocking=True)
        positions = positions.to(device=device, dtype=torch.int64, non_blocking=True)
        projected_context = projected_context.to(device=device, non_blocking=True)
        n_real = positions.shape[0]
        if projected_context.shape[0] > n_real:
            projected_context = projected_context[:n_real]
        if cache_loc_2d is not None:
            cache_loc_2d = cache_loc_2d.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if commit_lens is not None:
            commit_lens = commit_lens.to(
                device=device, dtype=torch.int32, non_blocking=True
            )
        if state_slot is not None:
            state_slot = state_slot.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if final_pos is not None:
            final_pos = final_pos.to(
                device=device, dtype=torch.int64, non_blocking=True
            )

        pool = self.draft_model_runner.token_to_kv_pool
        if isinstance(pool, DeepSeekV4TokenToKVPool):
            if pool.swa_req_ring_size is not None:
                swa_loc = self._ring_inject_loc(
                    pool=pool,
                    positions=positions,
                    cache_loc_2d=cache_loc_2d,
                    commit_lens=commit_lens,
                    state_slot=state_slot,
                    final_pos=final_pos,
                )
            else:
                swa_loc = pool.translate_loc_from_full_to_swa(cache_loc).to(torch.int32)
                if commit_lens is not None and cache_loc_2d is not None:
                    bs, verify_len = cache_loc_2d.shape
                    col = torch.arange(verify_len, device=cache_loc.device).view(1, -1)
                    committed_mask = (
                        col < commit_lens.to(torch.long).view(-1, 1)
                    ).reshape(-1)
                    swa_loc = torch.where(
                        committed_mask, swa_loc, torch.full_like(swa_loc, -1)
                    )
            with torch.inference_mode():
                self.draft_model.write_projected_context_kv(
                    projected_context=projected_context,
                    swa_loc=swa_loc,
                    positions=positions,
                    pool=pool,
                )
            return

        with torch.inference_mode():
            self.draft_model.write_projected_context_kv(
                projected_context=projected_context,
                pool=pool,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )

    def _inject_mla(
        self,
        *,
        pool,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        state_slot: Optional[torch.Tensor] = None,
        final_pos: Optional[torch.Tensor] = None,
    ) -> None:
        if pool.swa_req_ring_size is not None:
            swa_loc = self._ring_inject_loc(
                pool=pool,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
            )
        elif (
            cache_loc.is_cuda
            and cache_loc.is_contiguous()
            and commit_lens is not None
            and cache_loc_2d is not None
            and commit_lens.is_contiguous()
            and cache_loc.numel() == cache_loc_2d.numel()
        ):
            from sglang.kernels.ops.speculative.dspark.commit_swa import (
                committed_swa_locations,
            )

            swa_loc = committed_swa_locations(
                cache_loc,
                pool.full_to_swa_index_mapping,
                commit_lens,
                cache_loc_2d.shape[1],
            )
        else:
            swa_loc = pool.translate_loc_from_full_to_swa(cache_loc).to(torch.int32)
            if commit_lens is not None and cache_loc_2d is not None:
                bs, verify_len = cache_loc_2d.shape
                col = torch.arange(verify_len, device=cache_loc.device).view(1, -1)
                committed_mask = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(
                    -1
                )
                swa_loc = torch.where(
                    committed_mask, swa_loc, torch.full_like(swa_loc, -1)
                )

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=target_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )

    def _ring_inject_loc(
        self,
        *,
        pool,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        state_slot: Optional[torch.Tensor],
        final_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ring row for target-hidden injection into a per-request SWA ring.

        loc = swa_ring_rows(state_slot, pos, ring), with two skip (-1) rules:
          * SWA window: only the last ``win`` tokens per req land in the ring;
            older tokens share a ring slot (pos % ring) and would race, so drop
            them (needed for long prefill chunks).
          * commit gate: uncommitted verify tokens (col >= commit_len) are dropped.
        """
        if state_slot is None:
            raise RuntimeError(
                "SWA ring target-hidden injection requires state_slot "
                "(per-token draft req_pool_indices)."
            )
        ring = pool.swa_req_ring_size
        win = pool.sliding_window
        pos = positions.to(torch.int64)
        loc = swa_ring_rows(state_slot, pos, ring)
        if final_pos is not None:
            keep = pos > (final_pos.to(torch.int64) - win)
            loc = torch.where(keep, loc, torch.full_like(loc, -1))
        if commit_lens is not None and cache_loc_2d is not None:
            bs, verify_len = cache_loc_2d.shape
            col = torch.arange(verify_len, device=positions.device).view(1, -1)
            committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            loc = torch.where(committed, loc, torch.full_like(loc, -1))
        return loc.to(torch.int32)

    def inject_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        hidden_strided: torch.Tensor,
        commit_lens: torch.Tensor,
        bs: int,
        kv_loc_plan: KVLocPlan,
    ) -> None:
        stride = self.verify_num_draft_tokens
        prefix_lens = batch.seq_lens
        hidden = hidden_strided.view(bs, stride, -1)

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            if hidden_strided.numel() == 0:
                return
            if pool.swa_req_ring_size is not None:
                inject_layout = build_ring_commit_inject_layout(
                    req_pool_indices=batch.req_pool_indices,
                    prefix_lens=prefix_lens,
                    block_pos_offsets=self._block_pos_offsets[:stride],
                    commit_lens=commit_lens,
                    stride=stride,
                    ring_stride=pool.swa_req_ring_size,
                )
            else:
                inject_layout = BuildCommitInjectLayout.execute(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    prefix_lens=prefix_lens,
                    block_pos_offsets=self._block_pos_offsets[:stride],
                    full_to_swa_mapping=pool.full_to_swa_index_mapping,
                    commit_lens=commit_lens,
                    stride=stride,
                )
            with torch.inference_mode():
                self.draft_model.write_target_hidden_kv(
                    main_hidden=hidden.reshape(-1, hidden.shape[-1]),
                    swa_loc=inject_layout.swa_loc,
                    positions=inject_layout.positions,
                    pool=pool,
                )
            return

        positions_2d = prefix_lens.unsqueeze(1) + self._block_pos_offsets
        verify_cache_loc = self.ids_for(kv_loc_plan)
        verify_cache_loc_2d = verify_cache_loc.view(bs, stride)
        self.inject_target_hidden(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_cache_loc,
            cache_loc_2d=verify_cache_loc_2d,
            positions=positions_2d.reshape(-1),
            commit_lens=commit_lens,
        )

class _WorkerMethods:
    def weight_update_runners(self) -> list[tuple[str, "ModelRunner"]]:
        return (
            [("draft", self.draft_model_runner)]
            if self._hosts_draft and not self._is_lifecycle_only_pp_prefill_rank
            else []
        )

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
        draft_worker_cls: type[TpModelWorker] = TpModelWorker,
    ):
        super(DSparkWorkerV2, self).__init__()

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = get_schedule().page_size
        self.device = target_worker.device
        self._draft_worker = None
        self.enable_dp_spec_prefill_coordination = (
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
        )
        self.ps = ps = get_parallel()
        self._draft_is_moe = draft_is_deepseek_v4()
        disaggregation_mode = get_disagg().disaggregation_mode
        self._is_pd_prefill = disaggregation_mode == "prefill"
        self._is_context_only_pp_prefill_rank = (
            _is_hcu
            and self._draft_is_moe
            and _is_context_only_pp_prefill_rank(
                disaggregation_mode=disaggregation_mode,
                pp_rank=ps.pp_rank,
                pp_size=ps.pp_size,
            )
        )
        self._hosts_draft = ps.pp_group.is_last_rank or (
            self._is_context_only_pp_prefill_rank
        )
        self._is_lifecycle_only_pp_prefill_rank = False
        if not self._hosts_draft:
            return

        self._draft_dp_context_enabled = (
            get_parallel().attn_dp_enabled and not self._draft_is_moe
        )
        self._use_full_projection_prefill = False
        if self._is_pd_prefill and self._draft_is_moe and ps.pp_size > 1:
            target_layer_ids = [
                int(layer_id)
                for layer_id in (
                    self.model_runner.spec_aux_config.dflash_target_layer_ids or []
                )
            ]
            owner_pp_rank = resolve_single_owner_pp_rank(
                target_layer_ids=target_layer_ids,
                num_hidden_layers=self.model_runner.model_config.num_hidden_layers,
                pp_size=ps.pp_size,
            )
            self._use_full_projection_prefill = (
                owner_pp_rank == ps.pp_size - 1 and ps.pp_rank == owner_pp_rank
            )
        self._decode_graph_allowed = (
            get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
            and not self._is_pd_prefill
        )
        if (
            get_parallel().attn_dp_enabled
            and self._draft_is_moe
            and get_parallel().attn_tp_size > 1
        ):
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        # Inside the draft scope the context answers the draft's narrowed rank.
        self._target_tp_rank = get_parallel().tp_rank
        load_group = get_pp_stage_load_group()
        # HCU context-only ranks must retain their original PP placement so
        # the draft model can decide whether it owns target features.
        draft_pp_scope = (
            nullcontext()
            if self._is_context_only_pp_prefill_rank
            else draft_pp_context()
        )
        with draft_pp_scope, self._draft_context():
            bundle = build_draft_tp_worker(
                server_args=server_args,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                target_model_config=target_worker.model_runner.model_config,
                algo_label="DSPARK",
                attention_backend_override=(
                    DSV4_DRAFT_ATTENTION_BACKEND if self._draft_is_moe else None
                ),
                draft_worker_cls=draft_worker_cls,
                random_seed=target_worker.random_seed,
                load_group=load_group,
            )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model
        self._draft_sampler = None
        self._is_lifecycle_only_pp_prefill_rank = (
            self._draft_is_moe and self.draft_model.is_lifecycle_only
        )
        self._linear_accept_index_cache = None

        # The mask token is input-only (it is embedded, never sampled), so its
        # bound is the embedding-table row count: the PADDED vocab when the
        # target pads its embedding (e.g. Inkling true vocab 200058, padded
        # 201024, mask 200064), else the plain vocab size.
        target_model_config = self.target_worker.model_runner.model_config
        target_embed_rows = (
            getattr(target_model_config.hf_text_config, "padded_vocab_size", None)
            or target_model_config.vocab_size
        )
        # muP targets declare logits_mup_width_multiplier; the draft was
        # trained against the folded head, so compute_base_logits divides.
        self.draft_model.logits_mup_width_multiplier = getattr(
            target_model_config.hf_text_config, "logits_mup_width_multiplier", None
        )
        self._target_is_mambaish = mambaish_config(target_model_config) is not None
        runtime_config = resolve_runtime_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config,
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            target_vocab_size=int(target_embed_rows),
        )
        self.gamma = runtime_config.gamma
        self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
        self.sample_from_anchor = bool(self.draft_model.sample_from_anchor)
        self.query_token_num = self.gamma if self.sample_from_anchor else self.gamma + 1
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens
        self._mask_token_id = runtime_config.mask_token_id
        self._pp_context_feature_indices: Optional[list[int]] = None
        self._next_pp_proxy_tensors = None

        if self._is_lifecycle_only_pp_prefill_rank:
            self._init_lifecycle_only_prefill()
            return

        parallel = get_parallel()
        self._tp_sync = SpecTpSync(get_dp_tp_group())
        self._draft_graph_group = (
            parallel.attn_tp_group
            if self._draft_dp_context_enabled
            else parallel.tp_group
        )

        if get_parallel().tp_rank == 0:
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, query_token_num=%s, "
                "sample_from_anchor=%s, mask_token_id=%s, markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self.query_token_num,
                self.sample_from_anchor,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=int(self.query_token_num), device=self.device
        )

        if not getattr(self.draft_model, "uses_own_vocab_modules", False):
            self._attach_shared_modules()
        self._target_hidden_projection_enabled = False

        self._verify_planner = DSparkVerifyPlanner(
            draft_model=self.draft_model,
            gamma=self.gamma,
            model_runner=self.model_runner,
            device=self.device,
            tp_rank=get_parallel().tp_rank,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_sync=self._tp_sync,
        )
        if (
            get_parallel().attn_dp_enabled
            and not self._draft_is_moe
            and self._verify_planner.is_compact_mode
            and self._decode_graph_allowed
        ):
            raise ValueError(
                "DSpark dense-draft compact verify under attention DP does not "
                "yet support cuda graph (idle DP groups cannot join the token-keyed "
                "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                "cuda graph under DP."
            )
        self._kv_injector = TargetHiddenKvInjector(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            model_runner=self.model_runner,
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
        )
        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        # The step's simulated correct length, read by the eager and graph accepts.
        self._simulated_correct_len = (
            torch.zeros((1,), dtype=torch.int64, device=self.device)
            if self._simulate_acc_len > 0
            else None
        )
        self._greedy_draft = envs.SGLANG_DSPARK_USE_GREEDY_DRAFT.get()
        self._proposer = DraftBlockProposer(
            draft_model=self.draft_model,
            draft_model_runner=self.draft_model_runner,
            gamma=self.gamma,
            mask_token_id=self._mask_token_id,
            draft_block_spec_info=self._draft_block_spec_info,
            tp_sync=self._tp_sync,
            random_seed=target_worker.random_seed,
            greedy_draft=self._greedy_draft,
            dp_moe_sync=self._draft_is_moe and get_parallel().attn_dp_enabled,
        )
        self._verify_epilogue = None
        target_is_dsv41 = (
            getattr(
                self.target_worker.model_runner.model_config.hf_text_config,
                "model_type",
                None,
            )
            == "deepseek_v41"
        )
        static_epilogue_supported = (
            target_is_dsv41
            and self._verify_planner.mode_value == "static"
            and self._draft_is_moe
            and not get_parallel().attn_dp_enabled
            and get_parallel().pp_size == 1
        )
        # ROCm (V4.1 target only): inside a HIP graph the accept-site TP broadcasts
        # need the group's pynccl communicator
        if (
            (self._verify_planner.is_compact_mode or static_epilogue_supported)
            and self._decode_graph_allowed
            and (is_cuda() or (is_cuda_alike() and target_is_dsv41))
        ):
            self._verify_epilogue = build_verify_epilogue(
                DsparkVerifyEpilogue,
                max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                tp_sync=self._tp_sync,
                fused_argmax=target_is_dsv41,
                simulated_correct_len=self._simulated_correct_len,
                simulate_fold=_is_hip and envs.SGLANG_DSPARK_SIMULATE_FOLD.get(),
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                    kv_injector=self._kv_injector,
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )

        self._simulate_acc_greedy = (
            _is_hip
            and self._simulate_acc_len > 0
            and envs.SGLANG_SIMULATE_ACC_GREEDY.get()
        )
        if (
            self._simulate_acc_len > 0
            and self._simulate_acc_len != 1.0
            and not self._verify_planner.is_verify_all
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN>1.0 with DSpark requires a verify-all "
                "schedule (SGLANG_RAGGED_VERIFY_MODE=static, or =compact with the "
                "uninitialized/flat SPS table): a constant simulated correct_len>0 "
                "can exceed a trimmed request's verify budget (cap-accept, or "
                "compact with a profiled SPS table) and break the cutoff/cap "
                "accounting. SGLANG_SIMULATE_ACC_LEN=1.0 yields correct_len=0 "
                "(commit is the bonus token only), which stays within every verify "
                "budget and is safe in any mode. Got mode="
                f"{self._verify_planner.mode_value!r}, simulate_acc_len="
                f"{self._simulate_acc_len}."
            )

        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            tp_sync=self._tp_sync,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
            simulated_correct_len=self._simulated_correct_len,
            block_verification=get_spec().speculative_use_block_verification,
        )

        self._forced_budget_frac: Optional[float] = None
        self._need_mamba_verify_commit = False

        self._observers = DsparkStepObservers(
            planner=self._verify_planner,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            tp_rank=get_parallel().tp_rank,
            device=self.device,
            simulate_acc_len=self._simulate_acc_len,
        )

        if self._is_pd_prefill:
            if self._draft_is_moe and self._is_context_only_pp_prefill_rank:
                self.draft_model.prune_to_ctx_projection()
            elif not self._draft_is_moe:
                self.draft_model.prune_to_ctx_kv_injection()
            if _is_hcu and self._draft_is_moe:
                self._init_pp_context_feature_indices()

    def _init_lifecycle_only_prefill(self) -> None:
        self._verify_planner = None
        self._kv_injector = None
        self._proposer = None
        self._verify_epilogue = None
        self._verify_executor = None
        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        self._forced_budget_frac = None
        self._need_mamba_verify_commit = False
        self._observers = None

    def _attach_shared_modules(self) -> None:
        if getattr(self.draft_model, "uses_own_vocab_modules", False):
            if self.ps.tp_rank == 0:
                logger.info(
                    "DSpark draft uses its checkpoint-local embedding and LM head."
                )
            return

        # Non-final PP prefill ranks only compute their partial context
        # projection. They do not own the target embedding or LM head.
        if self._is_context_only_pp_prefill_rank:
            return

        target_model = self.target_worker.model_runner.model
        lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DSpark requires the target model to expose `lm_head` with `weight`."
            )
        self.draft_model.attach_shared_modules(
            embed_tokens=unwrap_lora_layer(
                self._resolve_target_embed_tokens(target_model)
            ),
            lm_head=lm_head,
        )

    def _init_pp_context_feature_indices(self) -> None:
        if not (_is_hcu and self._is_pd_prefill and self._draft_is_moe):
            return
        if self.ps.pp_size <= 1:
            return

        target_layer_ids = self.model_runner.spec_aux_config.dflash_target_layer_ids
        if not target_layer_ids:
            return

        target_model = self.target_worker.model_runner.model
        start_layer = int(target_model.start_layer)
        end_layer = int(target_model.end_layer)
        target_layer_to_feature = {
            int(layer_id): idx for idx, layer_id in enumerate(target_layer_ids)
        }
        local_feature_indices = [
            target_layer_to_feature[layer_id]
            for layer_id in range(start_layer, end_layer)
            if layer_id in target_layer_to_feature
        ]
        if not local_feature_indices:
            return

        self._pp_context_feature_indices = local_feature_indices
        if self._draft_is_moe and not self._use_full_projection_prefill:
            self.draft_model.prepare_target_hidden_partial(local_feature_indices)
        if self.ps.tp_rank == 0:
            logger.info(
                "DSpark PP-local context projection will accumulate feature indices %s "
                "from target layers %s for PP rank %s local layers [%s, %s).",
                local_feature_indices,
                list(target_layer_ids),
                self.ps.pp_rank,
                start_layer,
                end_layer,
            )

    def carries_confidence(self) -> bool:
        if (
            not self._hosts_draft
            or self._is_context_only_pp_prefill_rank
            or self._is_lifecycle_only_pp_prefill_rank
        ):
            return False
        return self._verify_planner.carries_confidence

    def is_lifecycle_only_pp_prefill_rank(self) -> bool:
        return self._is_lifecycle_only_pp_prefill_rank

    def _draft_model_runners(self) -> tuple:
        if not self._hosts_draft or self._is_lifecycle_only_pp_prefill_rank:
            return ()
        return super(DSparkWorkerV2, self)._draft_model_runners()

    def spec_v2_attn_backends(self) -> tuple:
        if not self._hosts_draft:
            return super(DSparkWorkerV2, self).spec_v2_attn_backends
        if self._is_context_only_pp_prefill_rank:
            return (self._target_worker.model_runner.attn_backend,)
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def _draft_context(self):
        with ExitStack() as stack:
            if self._draft_dp_context_enabled:
                stack.enter_context(draft_tp_context(True))
            stack.enter_context(speculative_moe_backend_context())
            stack.enter_context(speculative_moe_a2a_backend_context())
            yield

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if not self._hosts_draft or self._is_lifecycle_only_pp_prefill_rank:
            return
        if memory_pool_config is not None and self._is_context_only_pp_prefill_rank:
            page_size = int(self.page_size)

            def _minimal_capacity(capacity):
                if capacity is None or capacity == 0:
                    return capacity
                return page_size

            memory_pool_config = replace(
                memory_pool_config,
                max_total_num_tokens=page_size,
                full_max_total_num_tokens=_minimal_capacity(
                    memory_pool_config.full_max_total_num_tokens
                ),
                swa_max_total_num_tokens=_minimal_capacity(
                    memory_pool_config.swa_max_total_num_tokens
                ),
            )
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        if not self._hosts_draft or self._is_context_only_pp_prefill_rank:
            return
        with draft_pp_context(), self._draft_context():
            self._draft_worker.init_attention_backends()
        self._target_hidden_projection_enabled = _configure_target_hidden_projection(
            target_model=self.target_worker.model_runner.model,
            draft_model=self.draft_model,
            is_deepseek_v4_draft=self._draft_is_moe,
        )
        if self._target_hidden_projection_enabled and get_parallel().tp_rank == 0:
            logger.info(
                "DSpark prefill target-hidden projection runs before "
                "sequence-parallel gather."
            )
        self._need_mamba_verify_commit = mambaish_config(
            self.model_runner.model_config
        ) is not None and hasattr(
            self.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )

    def init_cuda_graphs(self):
        if (
            not self._hosts_draft
            or self._is_context_only_pp_prefill_rank
            or self._is_lifecycle_only_pp_prefill_rank
        ):
            return
        capture_decode_cuda_graph = self._decode_graph_allowed
        available_mem = self._tp_sync.available_memory_gb(
            SpecTpSyncSite.DSPARK_MEM,
            self.device,
            self.gpu_id,
            group=self._draft_graph_group,
        )
        if is_cuda_alike() and capture_decode_cuda_graph:
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        # Draft graph capture is inference-only.  Keep autograd disabled during
        # both warmup and capture: some HCU/AITER MoE outputs are views of leaf
        # tensors, so the shared-expert in-place accumulation is otherwise
        # rejected by PyTorch even though no backward pass can occur here.
        with draft_pp_context(), self._draft_context(), torch.inference_mode():
            if capture_decode_cuda_graph:
                # Keep the draft model graph enabled when folded proposal is
                # disabled, but do not capture the proposal head as a tail
                # hook. The proposer will compute base logits and the Markov
                # block eagerly from the graph's hidden states instead. Apart
                # from being the intended precision fallback, skipping the
                # unused hook avoids paying for two proposal computations.
                if envs.SGLANG_DSPARK_FOLDED_PROPOSAL.get():
                    self._draft_sampler = self._maybe_build_draft_sampler()
                    if self._draft_sampler is not None:
                        self.draft_model_runner.capture_tail_hooks.append(
                            make_draft_sampler_capture_hook(self._draft_sampler)
                        )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )

    def _refresh_partial_projection(self) -> None:
        if (
            self._draft_is_moe
            and not self._is_lifecycle_only_pp_prefill_rank
            and not self._use_full_projection_prefill
            and self._pp_context_feature_indices is not None
        ):
            self.draft_model.prepare_target_hidden_partial(
                self._pp_context_feature_indices
            )

    def update_weights_from_disk(self, recv_req):
        success, message = super(DSparkWorkerV2, self).update_weights_from_disk(recv_req)
        if success:
            self._refresh_partial_projection()
        return success, message

    def update_weights_from_ipc(self, recv_req):
        success, message = super(DSparkWorkerV2, self).update_weights_from_ipc(recv_req)
        if success:
            self._refresh_partial_projection()
        return success, message

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        if not self._hosts_draft:
            return
        self._forced_budget_frac = frac
        if self._is_lifecycle_only_pp_prefill_rank:
            return
        self._verify_planner.set_forced_budget_frac(frac)

    def dump_info_records(self) -> Optional[dict]:
        if not self._hosts_draft or self._is_lifecycle_only_pp_prefill_rank:
            return None
        return self._observers.dump_info_records()

    def clear_info_records(self) -> None:
        if not self._hosts_draft or self._is_lifecycle_only_pp_prefill_rank:
            return
        self._observers.clear_info_records()

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        if not self._hosts_draft or self._is_lifecycle_only_pp_prefill_rank:
            return None
        return self._observers.block_accept_estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        if not self._hosts_draft or self._is_lifecycle_only_pp_prefill_rank:
            return
        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)

    def set_pp_proxy_tensors_for_next_forward(self, pp_proxy_tensors) -> None:
        self._next_pp_proxy_tensors = pp_proxy_tensors

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if pp_proxy_tensors is None:
            pp_proxy_tensors = getattr(self, "_next_pp_proxy_tensors", None)
        self._next_pp_proxy_tensors = None
        if not self._hosts_draft:
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            batch_output.new_seq_lens = batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            return batch_output

        if self._is_lifecycle_only_pp_prefill_rank:
            return self._forward_lifecycle_only_prefill(
                batch=batch,
                on_publish=on_publish,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if batch.is_extend_in_batch and self.enable_dp_spec_prefill_coordination:
                plan = DPSpecPrefillCoordinationPlan(
                    *batch.dp_spec_prefill_coordination_metadata,
                    draft_width=self._proposer.query_token_num,
                    verify_width=self.verify_num_draft_tokens,
                )
                if plan.heterogeneous:
                    return self._forward_dp_spec_prefill_coordination(
                        batch, plan, on_publish, grammar_barrier, pp_proxy_tensors
                    )
            self._verify_planner.note_non_decode_step()
            self._observers.note_prefill_step()
            return self._forward_prefill(batch, on_publish, pp_proxy_tensors)

        return self._forward_decode(batch, on_publish, grammar_barrier)

    def _forward_lifecycle_only_prefill(
        self, *, batch: ScheduleBatch, on_publish, pp_proxy_tensors
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            return self._forward_idle_prefill(
                batch=batch,
                on_publish=on_publish,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
        if not (batch.forward_mode.is_extend() or batch.is_extend_in_batch):
            raise RuntimeError(
                "Lifecycle-only DSpark worker only supports prefill batches."
            )

        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        if batch_output.logits_output is not None:
            batch_output.logits_output.hidden_states = None
        incoming_ctx = (
            pp_proxy_tensors.tensors.get("dspark_ctx_acc")
            if pp_proxy_tensors is not None
            else None
        )
        outgoing_proxy = batch_output.pp_hidden_states_proxy_tensors
        if incoming_ctx is not None and outgoing_proxy is not None:
            outgoing_proxy.tensors["dspark_ctx_acc"] = incoming_ctx
        return batch_output

    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            return self._forward_idle_prefill(
                batch=batch,
                on_publish=on_publish,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )

        batch_output = self.target_worker.forward_batch_generation(
            batch,
            pp_proxy_tensors=pp_proxy_tensors,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_kv_loc_plan=True,
        )
        logits_output = batch_output.logits_output
        output_pp_proxy_tensors = batch_output.pp_hidden_states_proxy_tensors
        target_hidden = (
            logits_output.hidden_states if logits_output is not None else None
        )
        if target_hidden is None and output_pp_proxy_tensors is not None:
            target_hidden = output_pp_proxy_tensors.tensors.get(
                "dspark_aux_hidden_states"
            )
        next_token_ids = batch_output.next_token_ids
        self._tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, next_token_ids)
        new_seq_lens = batch.seq_lens
        batch_output.new_seq_lens = new_seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if target_hidden is None and self.ps.pp_size <= 1:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        has_local_target_hidden = (
            target_hidden is not None and target_hidden.numel() > 0
        )

        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        # Must inject before prefill returns: the scheduler may update radix
        # afterward, invalidating out_cache_loc.
        device = self.device
        pin_memory = is_pin_memory_available(device)
        ctx_lens = torch.tensor(
            batch.extend_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        positions, _ = compute_position(
            self.model_runner.prefill_attention_backend_str,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )

        # A SWA ring is keyed by (draft req slot, position); thread the per-token
        # state_slot + the req's final position so the injector keeps only the
        # last SWA window (older prefill tokens share a ring slot and would race).
        state_slot = final_pos = None
        if self._kv_injector.swa_ring_size is not None:
            repeats = ctx_lens.to(torch.int64)
            num_tokens = sum(batch.extend_lens)
            state_slot = torch.repeat_interleave(
                batch.req_pool_indices.to(device=device, dtype=torch.int64),
                repeats,
                output_size=num_tokens,
            )
            final_pos = torch.repeat_interleave(
                (draft_seq_lens + ctx_lens - 1).to(torch.int64),
                repeats,
                output_size=num_tokens,
            )
        target_hidden_is_projected = (
            self._target_hidden_projection_enabled
            and self.target_worker.model_runner.model.should_project_dspark_target_hidden(
                forward_mode=batch.forward_mode,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
        )

        draft_pool = self._kv_injector.draft_model_runner.token_to_kv_pool
        if (
            getattr(draft_pool, "swa_valid_from", None) is not None
            and batch.encoder_swa_reset is not None
        ):
            # A newly placed request's ring holds nothing below its cached prefix.
            placed = [i for i, reset in enumerate(batch.encoder_swa_reset) if reset]
            if placed:
                draft_pool.set_swa_valid_from(
                    batch.req_pool_indices[placed],
                    torch.tensor(
                        [batch.prefix_lens[i] for i in placed],
                        dtype=torch.int64,
                        device=device,
                    ),
                )
        # The draft KV goes to the slots the target prefill just wrote.
        cache_loc = self._kv_injector.ids_for(batch_output.kv_loc_plan)
        batch_output.kv_loc_plan = None
        token_indices = logits_output.hidden_states_token_indices
        if token_indices is not None:
            cache_loc = cache_loc[token_indices]
            positions = positions[token_indices]
            if state_slot is not None:
                state_slot = state_slot[token_indices]
                final_pos = final_pos[token_indices]

        hcu_pp_context = (
            _is_hcu
            and self._is_pd_prefill
            and self._draft_is_moe
            and self.ps.pp_size > 1
        )
        if not hcu_pp_context:
            if not has_local_target_hidden:
                raise RuntimeError("DSpark prefill requires target hidden states.")
            self._kv_injector.inject_target_hidden(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                state_slot=state_slot,
                final_pos=final_pos,
                target_hidden_is_projected=target_hidden_is_projected,
            )
        elif self._use_full_projection_prefill:
            if not has_local_target_hidden or output_pp_proxy_tensors is not None:
                raise RuntimeError(
                    "Single-owner DSpark prefill requires target hidden states "
                    "on the final PP rank."
                )
            self._kv_injector.inject_target_hidden(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                state_slot=state_slot,
                final_pos=final_pos,
            )
        else:
            incoming_ctx = (
                pp_proxy_tensors.tensors.get("dspark_ctx_acc")
                if pp_proxy_tensors is not None
                else None
            )
            local_ctx = None
            if (
                self.ps.pp_size > 1
                and has_local_target_hidden
                and self._pp_context_feature_indices is not None
            ):
                local_ctx = self.draft_model.project_target_hidden_partial(
                    target_hidden, self._pp_context_feature_indices
                )
            if output_pp_proxy_tensors is not None:
                output_pp_proxy_tensors.tensors.pop("dspark_aux_hidden_states", None)
            ctx_acc = None
            if incoming_ctx is not None and local_ctx is not None:
                ctx_acc = (
                    incoming_ctx.to(device=local_ctx.device, dtype=local_ctx.dtype)
                    + local_ctx
                )
            elif incoming_ctx is not None:
                ctx_acc = incoming_ctx.to(device=self.device, non_blocking=True)
            elif local_ctx is not None:
                ctx_acc = local_ctx

            if output_pp_proxy_tensors is not None:
                if ctx_acc is not None:
                    output_pp_proxy_tensors.tensors["dspark_ctx_acc"] = ctx_acc
            elif ctx_acc is not None:
                self._kv_injector.inject_projected_context(
                    projected_context=ctx_acc,
                    cache_loc=cache_loc,
                    positions=positions,
                    state_slot=state_slot,
                    final_pos=final_pos,
                )
            elif has_local_target_hidden and not (
                self.ps.pp_size > 1 and not self._draft_is_moe
            ):
                self._kv_injector.inject_target_hidden(
                    target_hidden=target_hidden,
                    cache_loc=cache_loc,
                    positions=positions,
                    state_slot=state_slot,
                    final_pos=final_pos,
                )
        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        if logits_output is not None:
            logits_output.hidden_states = None
            logits_output.hidden_states_token_indices = None

        if next_token_ids is not None:
            batch_output.next_draft_input = make_next_draft_input(
                bonus_tokens=next_token_ids,
                new_seq_lens=batch.seq_lens,
            )
        return batch_output

    def _forward_idle_prefill(
        self,
        *,
        batch: ScheduleBatch,
        on_publish,
        pp_proxy_tensors,
        capture_hidden_mode: CaptureHiddenMode,
    ) -> GenerationBatchResult:
        expert_distribution_metrics = None
        if get_parallel().attn_dp_enabled:
            batch_output = self.target_worker.forward_batch_generation(
                batch,
                pp_proxy_tensors=pp_proxy_tensors,
                capture_hidden_mode=capture_hidden_mode,
            )
            if self._is_context_only_pp_prefill_rank:
                return batch_output
            expert_distribution_metrics = batch_output.expert_distribution_metrics
        return self._decode_idle_result(
            on_publish=on_publish,
            expert_distribution_metrics=expert_distribution_metrics,
        )

    def inject_pd_hidden_chunk(
        self,
        req,
        hidden: torch.Tensor,
        hidden_start: int,
    ) -> Optional[torch.cuda.Event]:
        if hidden is None or hidden.numel() == 0:
            return None
        row_len = int(hidden.shape[0])
        pos = torch.arange(
            int(hidden_start),
            int(hidden_start) + row_len,
            dtype=torch.int64,
            device=self.device,
        )
        req_pool_idx = int(req.kv.req_pool_idx)
        cache_loc = self.model_runner.req_to_token_pool.req_to_token[req_pool_idx, pos]
        return self._kv_injector.inject_target_hidden(
            target_hidden=hidden,
            cache_loc=cache_loc,
            positions=pos,
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
        expert_distribution_metrics=None,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            expert_distribution_metrics=expert_distribution_metrics,
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _forward_decode(
        self,
        batch: ScheduleBatch,
        on_publish,
        grammar_barrier=None,
        coordination_plan=None,
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        global_num_reqs = (
            max(batch.global_num_tokens)
            if self._draft_is_moe
            and get_parallel().attn_dp_enabled
            and batch.global_num_tokens is not None
            else None
        )
        if coordination_plan is not None:
            rank = get_parallel().attn_dp_rank
            target_local_only = len(batch.global_num_tokens) == 1
            coordination_plan.apply(
                batch,
                "draft",
                rank,
                local_only=target_local_only or self._draft_dp_context_enabled,
            )

        if batch.forward_mode.is_idle():
            self._observers.note_idle_decode_step()
            expert_distribution_metrics = None
            if get_parallel().attn_dp_enabled:
                if self._draft_is_moe:
                    with self._draft_context():
                        self._proposer.run_idle_participation(batch)
                if coordination_plan is not None:
                    coordination_plan.apply(
                        batch, "target", rank, local_only=target_local_only
                    )
                target_verify = self._verify_executor.run_idle_participation(
                    batch=batch, idle_layout=self._idle_verify_ragged_layout(batch)
                )
                expert_distribution_metrics = getattr(
                    target_verify, "expert_distribution_metrics", None
                )
            return self._decode_idle_result(
                on_publish=on_publish,
                expert_distribution_metrics=expert_distribution_metrics,
            )

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        self._observers.begin_step()

        target_model = self.target_worker.model_runner.model
        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
            seq_lens_cpu=(
                batch.seq_lens_cpu
                if batch.seq_lens_cpu is not None
                else draft_input.nxt_kv_lens_cpu
            ),
        )

        sampling_info = batch.sampling_info
        # Simulated acceptance overrides correct_len, so only the bonus token needs the
        # request's temperature: draft + accept run greedy, the bonus is still sampled.
        simulate_bonus_sampling_info = None
        if (
            self._simulate_acc_greedy
            and sampling_info is not None
            and not sampling_info.is_all_greedy
            and not batch.has_grammar
            and verify_logits_adjustments_are_noop(sampling_info)
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        ):
            simulate_bonus_sampling_info = sampling_info
            sampling_info = None
        with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
            proposal = self._proposer.propose(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=bs,
                device=device,
                target_model=target_model,
                sampling_info=sampling_info,
            )
        draft_block_ids = proposal.draft_block_ids
        draft_block = proposal.draft_block
        draft_tokens = draft_block.draft_tokens
        if coordination_plan is not None:
            coordination_plan.apply(batch, "target", rank, local_only=target_local_only)

        confidence = proposal.confidence

        verify_token_budget = self._verify_planner.resolve_verify_token_budget(
            draft_input=draft_input,
            confidence=confidence,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
        )

        layout = self._verify_planner.schedule_layout(
            req_pool_indices=batch.req_pool_indices,
            prefix_lens=prefix_lens,
            device=device,
            confidence=confidence,
            budget=verify_token_budget,
            global_num_reqs=global_num_reqs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )
        run_compact = self._verify_planner.should_run_compact(layout=layout)

        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()

        # Must stay ahead of the target verify launch below.
        grammar_tree = (
            GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
        )

        # A live grammar forces the eager path: the folded epilogue accepts inside
        # the cuda graph off its own buffers, where the mask below never lands.
        fold_eligible = (
            self._verify_executor.verify_epilogue is not None
            and proposal.folded
            # The epilogue's in-graph accept is greedy (AcceptGreedy);
            # sampling batches must take the eager accept path even when the
            # draft proposal itself folded.
            and (sampling_info is None or sampling_info.is_all_greedy)
            and verify_logits_adjustments_are_noop(sampling_info)
            # The graph accept is greedy: a sampled simulated bonus stays in the graph
            # only when the epilogue draws it there (SGLANG_DSPARK_SIMULATE_FOLD).
            and (
                simulate_bonus_sampling_info is None
                or getattr(self._verify_executor.verify_epilogue, "simulate_fold", False)
            )
            and not batch.has_grammar
        )
        self._verify_executor.begin_simulated_step()
        if fold_eligible and getattr(self._verify_executor.verify_epilogue, "simulate_fold", False):
            self._verify_executor.verify_epilogue.begin_sim_bonus(
                bs,
                (
                    None
                    if simulate_bonus_sampling_info is None
                    else simulate_bonus_sampling_info.temperatures[:bs]
                ),
            )
        prepare_mamba_track_for_verify(batch)
        with self._observers.segment(InfoSegment.TARGET_VERIFY):
            if run_compact:
                target_verify, hidden_strided = self._verify_executor.run_compact(
                    batch=batch,
                    layout=layout,
                    draft_block_ids=draft_block_ids,
                    draft_tokens=draft_tokens,
                    bs=bs,
                    device=device,
                    sampling_info=sampling_info,
                    verify_window=verify_window,
                    inject_gate=fold_eligible,
                )
            else:
                if (
                    self._verify_epilogue is not None
                    and self._verify_planner.mode_value == "static"
                ):
                    self._verify_epilogue.begin_static_step(bs, fold_eligible)
                target_verify = self._verify_executor.run_non_compact(
                    batch=batch,
                    draft_input=draft_input,
                    verify_ids_2d=verify_ids_2d,
                    verify_window=verify_window,
                    sampling_info=sampling_info,
                )
                hidden_strided = None
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph
        if batch.has_grammar:
            # run_compact scatters its rows back to (bs * chain_len), so the mask
            # lines up with the logits on both verify paths.
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )
            if grammar_mask is not None:
                grammar_mask.apply(logits_output.next_token_logits)

        epilogue = self._verify_executor.verify_epilogue
        folded_accept = (
            fold_eligible
            and can_run_cuda_graph
            and (run_compact or self._verify_planner.mode_value == "static")
        )
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=folded_accept,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=layout,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            simulate_bonus_sampling_info=simulate_bonus_sampling_info,
        )
        self.model_runner.ngram_embedding_manager.update_after_verify(
            verify_ids_2d=verify_ids_2d,
            req_pool_indices=batch.req_pool_indices,
            commit_lens=accept.commit_lens,
        )
        sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(
            sampling_info,
            next_token_logits=logits_output.next_token_logits,
            draft_input=draft_input,
            draft_token_num=self.verify_num_draft_tokens,
            bs=bs,
            greedy_mask=draft_block.greedy_mask,
        )
        if sampling_mask_capture is not None:
            logits_output.sampling_mask_output = sampling_mask_capture.build_output(
                out_tokens=accept.out_tokens,
                commit_lens=accept.commit_lens,
            )
        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                accept.out_tokens.reshape(-1),
                chain_stride=self.verify_num_draft_tokens,
            )

        if on_publish is not None:
            if confidence is not None:
                on_publish(accept.new_seq_lens, confidence=confidence)
            else:
                on_publish(accept.new_seq_lens)

        self._commit_target_mamba_states_after_verify(
            batch=batch,
            seq_lens_pre_verify=prefix_lens,
            seq_lens_post_verify=accept.new_seq_lens,
            commit_lens=accept.commit_lens,
        )

        folded_commit = folded_accept and epilogue.folds_commit
        # Consume in this step: every decode graph size shares one aux output,
        # which the next target forward overwrites (resolve_aux_hidden_states_width).
        if not folded_commit:
            self._verify_executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=accept.commit_lens,
                bs=bs,
                run_compact=run_compact,
            )
        logits_output.hidden_states = None

        self._observers.observe_verify_step(
            forward_ct=int(batch.forward_iter),
            reqs=batch.reqs,
            bs=bs,
            proposal_folded=proposal.folded,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            layout=layout,
            confidence=confidence,
            confidence_raw=proposal.confidence_raw,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            draft_block=draft_block,
            sampling_info=sampling_info,
            correct_len=accept.correct_len,
            cap_trim_lens=accept.cap_trim_lens,
            bonus=accept.bonus,
            commit_lens=accept.commit_lens,
            verify_token_budget=verify_token_budget,
            req_pool_indices=batch.req_pool_indices,
            verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )

        next_draft_input = make_next_draft_input(
            bonus_tokens=accept.bonus,
            new_seq_lens=accept.new_seq_lens,
        )
        return build_verify_generation_result(
            target_verify,
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            cap_lens=(
                layout.verify_lens.to(torch.int32) if layout is not None else None
            ),
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
        )

    def get_confidence_budget_prepare(self):
        if not self._hosts_draft or self._is_lifecycle_only_pp_prefill_rank:
            return None
        return self._verify_planner.confidence_budget_prepare()
_REGISTERED = False


def _required_globals(code):
    names = {op.argval for op in dis.get_instructions(code) if op.opname == "LOAD_GLOBAL"}
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_required_globals(constant))
    return names


def _bind(raw, module, owner, name, decorators):
    for symbol in _required_globals(raw.__code__):
        if symbol not in vars(module) and symbol in globals():
            setattr(module, symbol, globals()[symbol])
    missing = _required_globals(raw.__code__) - vars(module).keys() - vars(builtins).keys()
    if missing:
        raise ImportError("DSpark runtime namespace lacks " + str(sorted(missing)))
    fn = types.FunctionType(raw.__code__, vars(module), name, raw.__defaults__, raw.__closure__)
    fn.__kwdefaults__ = raw.__kwdefaults__
    fn.__annotations__ = raw.__annotations__
    fn.__qualname__ = owner.__qualname__ + "." + name
    for decorator in reversed(decorators):
        if decorator == "property":
            fn = property(fn)
        elif decorator == "staticmethod":
            fn = staticmethod(fn)
        elif decorator == "classmethod":
            fn = classmethod(fn)
        elif decorator == "contextmanager":
            fn = contextmanager(fn)
        else:
            raise RuntimeError("Unmapped DSpark descriptor: " + decorator)
    return fn


def register_hooks():
    global _REGISTERED
    if _REGISTERED or not is_hcu():
        return
    C.resolve_single_owner_pp_rank = resolve_single_owner_pp_rank
    C.use_empty_draft_model_for_pp_prefill = use_empty_draft_model_for_pp_prefill
    M._is_hcu = is_hcu()
    M.Fp8LinearMethod = Fp8LinearMethod
    for name in ("inverse_transform_scale_ue8m0", "transform_scale_ue8m0", "_BlockFp8LinearSlice", "DSparkV4StageMixin", "DSparkV4PlatformStage", "DSparkV4Stage", "NonCudaDSparkMixin", "DeepseekV4PlatformDecoderLayer", "DSparkAttention", "use_empty_draft_model_for_pp_prefill"):
        setattr(M, name, globals()[name])
    W._is_hcu = is_hcu()
    W.resolve_single_owner_pp_rank = resolve_single_owner_pp_rank
    W._is_context_only_pp_prefill_rank = _is_context_only_pp_prefill_rank
    for name, decorators in [('__init__', []), ('prune_to_ctx_projection', []), ('project_target_hidden', []), ('prepare_target_hidden_partial', []), ('project_target_hidden_partial', []), ('write_target_hidden_kv', []), ('write_projected_context_kv', []), ('_write_context_hidden_kv', []), ('forward', []), ('_forward_impl', []), ('load_weights', []), ('_remap_dspark_weight_name', [])]:
        raw = vars(_ModelMethods)[name]
        setattr(M.DeepseekV4ForCausalLMDSpark, name, _bind(raw, M, M.DeepseekV4ForCausalLMDSpark, name, decorators))
    for name, decorators in [('swa_ring_size', ['property']), ('inject_target_hidden', []), ('_record_cuda_event', ['staticmethod']), ('inject_projected_context', []), ('_inject_mla', []), ('_ring_inject_loc', []), ('inject_ragged', [])]:
        raw = vars(_InjectorMethods)[name]
        setattr(I.TargetHiddenKvInjector, name, _bind(raw, I, I.TargetHiddenKvInjector, name, decorators))
    for name, decorators in [('weight_update_runners', []), ('__init__', []), ('_init_lifecycle_only_prefill', []), ('_attach_shared_modules', []), ('_init_pp_context_feature_indices', []), ('carries_confidence', ['property']), ('is_lifecycle_only_pp_prefill_rank', ['property']), ('_draft_model_runners', []), ('spec_v2_attn_backends', ['property']), ('_draft_context', ['contextmanager']), ('alloc_memory_pool', []), ('init_attention_backends', []), ('init_cuda_graphs', []), ('_refresh_partial_projection', []), ('update_weights_from_disk', []), ('update_weights_from_ipc', []), ('set_dspark_forced_budget_frac', []), ('dump_info_records', []), ('clear_info_records', []), ('block_accept_estimate_log_suffix', []), ('note_request_finished', []), ('set_pp_proxy_tensors_for_next_forward', []), ('forward_batch_generation', []), ('_forward_lifecycle_only_prefill', []), ('_forward_prefill', []), ('_forward_idle_prefill', []), ('inject_pd_hidden_chunk', []), ('_decode_idle_result', []), ('_forward_decode', []), ('get_confidence_budget_prepare', [])]:
        raw = vars(_WorkerMethods)[name]
        setattr(W.DSparkWorkerV2, name, _bind(raw, W, W.DSparkWorkerV2, name, decorators))
    _REGISTERED = True
