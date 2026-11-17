# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from sglang.srt.models.deepseek_v4 import (
    Aligner,
    Backend,
    CaptureHiddenMode,
    DEEPSEEK_V4_STACKED_PARAMS_MAPPING,
    DeepSeekV4Config,
    Engram,
    EngramHasher,
    EngramLayout,
    F,
    Fp8LinearMethod,
    FusedMoE,
    GPU_PLAN_KEY,
    Iterable,
    LazyValue,
    List,
    LogitsMetadata,
    LogitsProcessor,
    MM_PAD_SHIFT_VALUE,
    ModelConfigForExpertLocation,
    MultimodalInputs,
    Optional,
    PPMissingLayer,
    PPProxyTensors,
    ParallelLMHead,
    Phase,
    RMSNorm,
    Set,
    SimpleNamespace,
    Tuple,
    Union,
    ViT,
    VocabParallelEmbedding,
    WeightsMapper,
    _FP8_WO_A_GEMM,
    _FP8_WO_A_UE8M0,
    _HC_PRENORM_DEEPGEMM_MIN_TOKENS,
    _QUARK_CKPT_TO_SGLANG_MAPPER,
    _clone_if_runai_streamed_tensor,
    _flashinfer_mhc_pre_num_splits,
    _fuse_deepseek_v4_wqkv_a_pair,
    _gather_late_layer_ids,
    _is_cuda,
    _is_fused_mhc_post_pre_enabled_xpu,
    _is_hip,
    _is_npu,
    _is_quark_checkpoint,
    _is_xpu,
    _prepare_deepseek_v4_weights,
    _use_aiter_bpreshuffle_gfx95,
    _wo_a_weight_scale_to_e8m0,
    add_prefix,
    array,
    bcg_deepseek_v4_engram_hash_ids,
    bcg_deepseek_v4_enter_late_layer_tail,
    bcg_deepseek_v4_exit_late_layer_tail,
    check_cuda_graph_backend,
    compile_in_capture_mode,
    concurrent,
    deepseek_v2,
    default_weight_loader,
    dp_gather_replicate,
    embed_mm_inputs,
    enter_late_layer_dp,
    envs,
    exit_late_layer_dp,
    functools,
    get_attn_backend,
    get_attn_tp_context,
    get_exec,
    get_forward,
    get_global_dp_buffer_len,
    get_global_expert_distribution_recorder,
    get_layer_id,
    get_mhc_ops,
    get_moe_a2a_backend,
    get_parallel,
    get_platform,
    hc_head_torch,
    image_token_types,
    is_allocation_symmetric,
    is_cp_active,
    is_dp_attention_enabled,
    is_dsa_enable_prefill_cp,
    is_in_breakable_cuda_graph,
    is_shared_experts_fusion_disabled,
    is_wint4afp8_or_wint4a16_config,
    late_layer_dp_counts,
    log_info_on_rank0,
    logger,
    make_hc_head_params,
    make_hc_mixing_params,
    make_pp_layers,
    materialize_image_gpu,
    maybe_executor_submit,
    nn,
    nullcontext,
    prime_rope_cos_sin,
    quant_blocks_shared_experts_fusion,
    should_async_load,
    should_skip_post_experts_all_reduce,
    time,
    torch,
    use_npu_arch35_mxfp8_wo_a,
    use_symmetric_memory,
    uses_per_rank_fused_shared_slots,
    view_aiter_fused_rms_transposed_fp8_scale,
    wo_a_fp8_gemm_enabled,
)
from sglang.srt.models.deepseek_v4 import DeepseekV4Model as NativeDeepseekV4Model, DeepseekV4ForCausalLM as NativeDeepseekV4ForCausalLM
from hcu_sglang.models.deepseek.current_v4.attention import MqaAttentionBase, MQALayer
from hcu_sglang.models.deepseek.current_v4.execution import mhc, hip as _hip, lazy_boundary as hc_boundary
from hcu_sglang.models.deepseek.current_v4.execution.platform_stage import NonCudaDecoderMixin, forward_layer_fused_boundary
from hcu_sglang.models.deepseek.current_v4.execution.replay_tail import DecoderReplayTail
from hcu_sglang.models.deepseek.current_v4.execution.fused_mhc import is_cross_layer_mhc_fusion_enabled
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateAccumulator, AuxHiddenStatePacker, pack_aux_hidden_states
from sglang.srt.batch_overlap.single_batch_overlap import SboFlags
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.runtime_context import get_disagg
from hcu_sglang.models.deepseek.current_v4.execution.cp_helpers import is_cp_v2_active
from hcu_sglang.models.deepseek.deepseek_nextn.methods.model import enable_cp_v2
from hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_cp_utils import cp_all_gather_rerange_output, cp_round_robin_input_ids, cp_split_and_rebuild_data, cp_split_and_rebuild_position, prepare_context_parallel_metadata
from hcu_sglang.models.deepseek.current_v4.hcu_backend.compat import is_dsa_prefill_cp_round_robin_split
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from hcu_sglang.models.deepseek.current_v4.execution.cp_helpers import can_dsa_cp_split
from hcu_sglang.models.deepseek.current_v4.execution.mm_padding import pad_mm_tokens
from hcu_sglang.utils.detection import is_hcu
from sglang.srt.utils import get_bool_env_var
_is_hcu = is_hcu()
if _is_hcu:
    _is_cuda = False
MHC_POST_MULT_VALUE = mhc.MHC_POST_MULT_VALUE
_FUSE_MHC_REPEAT_CP_SPLIT = envs.SGLANG_DSV4_FUSE_MHC_REPEAT_CP_SPLIT.get()
_use_aiter_tilelang_mhc = get_bool_env_var("SGLANG_ROCM_USE_AITER_TILELANG_MHC")
_logged_mid_chunk_decoder_skip = False
try:
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
except ImportError:
    pass


def _can_defer_mhc_repeat_cp_split(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    hc_mult: int,
    cp_size: int,
    cp_rank: int,
) -> bool:
    """Require an unpadded, equal-sized legacy RR split without device reads."""
    if (
        hidden_states.ndim != 2
        or hidden_states.shape[1] == 0
        or hc_mult <= 0
        or cp_size <= 1
        or not 0 <= cp_rank < cp_size
    ):
        return False
    num_tokens = hidden_states.shape[0]
    if num_tokens == 0 or num_tokens % cp_size != 0:
        return False
    # Existing CPU batch metadata proves logical rows; no device-value reads.
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if not isinstance(extend_lens, (list, tuple)) or not all(
        isinstance(length, int) and not isinstance(length, bool) and length >= 0
        for length in extend_lens
    ):
        return False
    if sum(extend_lens) != num_tokens:
        return False
    # None is the real ForwardBatch default, set only when padding is applied.
    original_num_tokens = getattr(forward_batch, "_original_num_tokens", None)
    if original_num_tokens is not None and original_num_tokens != num_tokens:
        return False
    return all(
        tensor is not None and tensor.ndim > 0 and tensor.shape[0] == num_tokens
        for tensor in (
            getattr(forward_batch, "input_ids", None),
            getattr(forward_batch, "positions", None),
            getattr(forward_batch, "out_cache_loc", None),
        )
    )

def _repeat_mhc_input_on_cp_rank(
    hidden_states: torch.Tensor, hc_mult: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """Copy directly to local [tokens / CP, hc_mult, hidden] mHC storage."""
    repeat_ops = _get_mhc_repeat_cp_ops()
    if repeat_ops is not None and repeat_ops[0](
        hidden_states, hc_mult, cp_size, cp_rank
    ):
        return repeat_ops[1](hidden_states, hc_mult, cp_size, cp_rank)
    # Slice is a view; repeat is the only output materialization.
    return hidden_states[cp_rank::cp_size].unsqueeze(1).repeat(1, hc_mult, 1)

@functools.cache
def _get_mhc_repeat_cp_ops():
    """Resolve the categorized LightOp attention APIs once per process."""
    try:
        from lightop.attention import (
            mhc_repeat_cp_sglang,
            supports_mhc_repeat_cp_sglang,
        )
    except (ImportError, OSError) as exc:
        logger.warning("DSV4 mHC repeat+CP unavailable; using torch repeat: %s", exc)
        return None
    return supports_mhc_repeat_cp_sglang, mhc_repeat_cp_sglang

def _flashinfer_hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flashinfer.mhc import mhc_pre_big_fuse

    from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

    num_tokens, hc_mult, hidden_size = x.shape
    hc_hidden_size = hc_mult * hidden_size
    mix_dim = hc_fn.shape[0]  # hc_mult * (2 + hc_mult) == 24
    n_splits = _flashinfer_mhc_pre_num_splits(num_tokens, hc_hidden_size)

    dot_mix = torch.empty(
        (n_splits, num_tokens, mix_dim), dtype=torch.float32, device=x.device
    )
    sqrsum = torch.empty((n_splits, num_tokens), dtype=torch.float32, device=x.device)
    tf32_hc_prenorm_gemm(
        x.reshape(num_tokens, hc_hidden_size), hc_fn, dot_mix, sqrsum, n_splits
    )
    if n_splits == 1:
        dot_mix = dot_mix.squeeze(0)
        sqrsum = sqrsum.squeeze(0)

    post, comb, layer_input = mhc_pre_big_fuse(
        dot_mix,
        sqrsum,
        x,
        hc_scale,
        hc_base,
        hc_hidden_size,
        rms_eps=rms_eps,
        mhc_pre_eps=hc_eps,
        mhc_sinkhorn_eps=hc_eps,
        mhc_post_mult_value=MHC_POST_MULT_VALUE,
        sinkhorn_repeat=sinkhorn_iters,
        num_splits=n_splits,
    )
    return layer_input, post.squeeze(-1), comb

@functools.cache
def _hcu_arch_supports_tilelang_mmac() -> bool:
    """Whether the current HCU can JIT-compile tilelang T.gemm (MLS/GEMM_MLS).

    tilelang's ``hcu_mmac_k_dim`` only accepts gfx938 / gfx92a / gfx946; other
    archs (e.g. gfx936) fatal at LayerInference for any ``T.gemm``. On those
    archs we must skip sglang's tilelang split-k mhc_pre and go straight to
    AITER's fully-fused ``mhc_pre_big_fuse`` instead.
    """
    if not _is_hcu:
        return True
    try:
        gcn_arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    except Exception:
        # Fail closed: an unreadable arch must not take the tilelang path,
        # which fatals at LayerInference on unsupported HCUs.
        return False
    return any(a in gcn_arch for a in ("gfx938", "gfx92a", "gfx946"))

def _fused_rmsnorm_fp8_quant(hidden_states, weight, eps, e8m0: bool = False):
    if e8m0:
        # Consumers run the e8m0 mxscale GEMM (SGLANG_DSV4_DENSE_MXSCALE).
        from hcu_sglang.kernels.rmsnorm_fp8_e8m0 import (
            rmsnorm_fp8_e8m0_quant,
        )

        return rmsnorm_fp8_e8m0_quant(hidden_states, weight, eps)
    x_quant, x_bf16, _, _ = fused_rms_fp8_group_quant(
        hidden_states,
        weight,
        eps,
        inp2=None,
        inp2_weight=None,
        inp2_epsilon=None,
        group_size=128,
        dtype_quant=torch.float8_e4m3fn,
        res1=None,
        output_unquantized_inp1=True,
        transpose_scale=_use_aiter_bpreshuffle_gfx95,
    )
    if _use_aiter_bpreshuffle_gfx95:
        x_quant = (
            x_quant[0],
            view_aiter_fused_rms_transposed_fp8_scale(x_quant[1]),
        )
    return x_quant, x_bf16

class DeepseekV4DecoderLayerBase(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
        engram_layout: Optional[EngramLayout] = None,
        hc_stats_stream: Optional[torch.cuda.Stream] = None,
        moe_routed_quant_stream: Optional[torch.cuda.Stream] = None,
        *,
        boundary_pre_from_prev: Optional[bool] = None,
        boundary_mhc_fusions: bool = True,
    ) -> None:
        super().__init__()
        self.hc_stats_stream = hc_stats_stream
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.boundary_pre_from_prev = (
            config.hc_pre_from_prev_sublayer and not is_nextn
            if boundary_pre_from_prev is None
            else boundary_pre_from_prev
        )
        self.boundary_mhc_fusions = boundary_mhc_fusions
        self.self_attn = self._build_self_attn(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )
        moe_alt_stream = (
            alt_streams[0]
            if (
                alt_streams is not None
                and (
                    _is_cuda
                    or (_is_hcu and SboFlags.fuse_shared_experts_inside_sbo())
                    or envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
                    or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
                )
            )
            else None
        )
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=moe_alt_stream,
            routed_quant_stream=moe_routed_quant_stream,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
            vl_correction_bias=config.model_type == "deepseek_v41"
            and config.vision_n_layers > 0
            and not getattr(config, "language_model_only", False),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        (
            self.hc_attn_fn,
            self.hc_ffn_fn,
            self.hc_attn_base,
            self.hc_ffn_base,
            self.hc_attn_scale,
            self.hc_ffn_scale,
        ) = make_hc_mixing_params(hc_mult, config.hidden_size)
        self.rms_norm_eps = config.rms_norm_eps
        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.use_fused_mhc_post_pre = (
            is_cross_layer_mhc_fusion_enabled() or _is_fused_mhc_post_pre_enabled_xpu()
        )
        # The fused post+pre boundary bakes in the same-sublayer pre-mix.
        self.hc_pre_from_prev_sublayer = config.hc_pre_from_prev_sublayer
        if self.hc_pre_from_prev_sublayer:
            self.use_fused_mhc_post_pre = False
        if _is_hip:
            _hip.init_decoder_layer(self, quant_config)
        self.hc_cfg = mhc.HcConfig(
            mult=hc_mult,
            sinkhorn_iters=config.hc_sinkhorn_iters,
            eps=config.hc_eps,
            rms_eps=config.rms_norm_eps,
            hidden=config.hidden_size,
            pre_from_prev=self.hc_pre_from_prev_sublayer,
            cp_prefill=self.dsa_enable_prefill_cp,
        )
        self.attn_hc: Optional[mhc.HcSubLayer] = None
        self.ffn_hc: Optional[mhc.HcSubLayer] = None
        self._hc_attn_tf32_parts = self._hc_ffn_tf32_parts = None
        self._hc_attn_bf16_parts = self._hc_ffn_bf16_parts = None
        self._init_hyper_connections()
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(
                config,
                layer_id,
                engram_layout,
                quant_config=quant_config,
                prefix=add_prefix("engram", prefix),
            )
        self._input_layernorm_weight_bf16 = None
        self._post_attention_layernorm_weight_bf16 = None

    def _build_self_attn(
        self,
        *,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
        alt_streams: Optional[List[torch.cuda.Stream]],
        compress_ratio_override: Optional[int],
    ) -> MQALayer:
        return MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )

    def _init_hyper_connections(self) -> None:
        """(Re)bind the two hyper-connections to the current mixing weights."""
        if not self.hc_pre_from_prev_sublayer:
            return
        self.attn_hc = mhc.HcSubLayer(
            self.hc_cfg,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.input_layernorm,
            self._hc_attn_tf32_parts,
            self._hc_attn_bf16_parts,
        )
        self.ffn_hc = mhc.HcSubLayer(
            self.hc_cfg,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.post_attention_layernorm,
            self._hc_ffn_tf32_parts,
            self._hc_ffn_bf16_parts,
        )

    def refresh_mhc_norm_weight_cache(self):
        # Cache bf16 norm weights so the fused path does not allocate/cast per forward.
        self._input_layernorm_weight_bf16 = (
            self.input_layernorm.weight.data.bfloat16().contiguous()
        )
        self._post_attention_layernorm_weight_bf16 = (
            self.post_attention_layernorm.weight.data.bfloat16().contiguous()
        )

        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

        # The original FP32 parameters stay intact for small rows and invariant mode.
        self._hc_attn_tf32_parts = self._hc_ffn_tf32_parts = None
        self._hc_attn_bf16_parts = self._hc_ffn_bf16_parts = None
        # Prepare HCU parts after each weight load/update, before graph capture.
        # Original FP32 parameters remain the source of truth and fallback.
        if (
            mhc._use_hcu_hc_mix_stats
            and self.hc_pre_from_prev_sublayer
            and self.hc_attn_fn.shape == (24, 20480)
            and getattr(self.config, "model_type", None) == "deepseek_v41"
            and not is_batch_invariant_mode_enabled()
        ):
            from lightop import split_bf16_hc_weight, split_tf32_hc_weight

            self._hc_attn_tf32_parts = split_tf32_hc_weight(self.hc_attn_fn.data)
            self._hc_ffn_tf32_parts = split_tf32_hc_weight(self.hc_ffn_fn.data)
            self._hc_attn_bf16_parts = split_bf16_hc_weight(self.hc_attn_fn.data)
            self._hc_ffn_bf16_parts = split_bf16_hc_weight(self.hc_ffn_fn.data)
        elif (
            self.hc_pre_from_prev_sublayer
            and (get_platform().is_sm100 or get_platform().is_sm90)
            and self.hc_attn_fn.shape == (24, 20480)
            and envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
            and getattr(self.config, "model_type", None) == "deepseek_v41"
            and not is_batch_invariant_mode_enabled()
        ):
            from sglang.kernels.ops.layernorm.mhc import split_bf16_hc_weight

            if get_platform().is_sm90:
                # Hopper's compensated projection does not require DeepGEMM.
                self._hc_attn_bf16_parts = split_bf16_hc_weight(self.hc_attn_fn.data)
                self._hc_ffn_bf16_parts = split_bf16_hc_weight(self.hc_ffn_fn.data)
            else:
                from sglang.kernels.ops.layernorm.mhc import split_tf32_hc_weight
                from sglang.srt.layers.deep_gemm_wrapper.configurer import (
                    ENABLE_JIT_DEEPGEMM,
                )

                if ENABLE_JIT_DEEPGEMM:
                    import deep_gemm

                    if callable(getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)):
                        self._hc_attn_tf32_parts = split_tf32_hc_weight(
                            self.hc_attn_fn.data
                        )
                        self._hc_ffn_tf32_parts = split_tf32_hc_weight(
                            self.hc_ffn_fn.data
                        )
                        self._hc_attn_bf16_parts = split_bf16_hc_weight(
                            self.hc_attn_fn.data
                        )
                        self._hc_ffn_bf16_parts = split_bf16_hc_weight(
                            self.hc_ffn_fn.data
                        )
        if self.hc_pre_from_prev_sublayer:
            self._init_hyper_connections()
        # The fuse gates and boundaries snapshot load-time facts; weight updates
        # re-run this, so drop them and re-resolve on the next forward.
        self.__dict__.pop("_can_fuse_attn_mhc", None)
        self.__dict__.pop("_can_fuse_ffn_mhc", None)

    def _input_norm(
        self,
        hidden_states: torch.Tensor,
        allow_aiter_quant: bool = True,
        coefficients=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """input_layernorm(hidden_states) as (the bf16 norm attention reads, the pre-quantized
        operand of its dense projections or None); coefficients is a ROCm boundary's pending
        reduce + sinkhorn."""
        if _is_hip:
            return _hip.input_norm(
                self,
                hidden_states,
                allow_aiter_quant,
                coefficients,
                _fused_rmsnorm_fp8_quant,
            )
        return self.input_layernorm(hidden_states), None

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: Optional[nn.Module] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        """If *norm* is given and the TileLang path is active, the returned
        hidden_states are already post-norm (the norm is fused into the kernel)."""

        @compile_in_capture_mode
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        shape, dtype = x.size(), x.dtype

        if _is_npu:
            return get_mhc_ops().npu_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                hc_mult=self.hc_mult,
                hc_sinkhorn_iters=self.hc_sinkhorn_iters,
                rms_norm_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
                forward_batch=forward_batch,
            )

        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=torch.float32, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=torch.float32, device=x.device
            )
            return y, post, comb, False

        if _is_xpu:
            norm_kwargs = {}
            if norm is not None:
                norm_kwargs["norm_weight"] = norm.weight.data
                norm_kwargs["norm_eps"] = norm.variance_epsilon

            post, comb, y = get_mhc_ops().mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=MHC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                **norm_kwargs,
            )
            return y, post, comb, norm is not None

        if envs.SGLANG_OPT_USE_FLASHINFER_MHC.get():
            y, post, comb = _flashinfer_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                rms_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
                sinkhorn_iters=self.hc_sinkhorn_iters,
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            if (
                _is_hcu
                and _use_aiter_tilelang_mhc
                and not _hcu_arch_supports_tilelang_mmac()
            ):
                # This HCU arch (e.g. gfx936) lacks tilelang T.gemm support, so
                # sglang's own tilelang split-k mhc_pre fatals in LayoutInference.
                # Bypass it entirely and use AITER's fully-fused mhc_pre_big_fuse,
                # which is also what earlier sglang releases dispatched to here.
                from aiter.ops.tilelang import mhc_pre_big_fuse

                post, comb, y = mhc_pre_big_fuse(
                    residual=x,
                    fn=hc_fn,
                    mhc_scale=hc_scale,
                    mhc_base=hc_base,
                    rms_eps=self.rms_norm_eps,
                    mhc_pre_eps=self.hc_eps,
                    mhc_sinkhorn_eps=self.hc_eps,
                    mhc_post_mult_value=MHC_POST_MULT_VALUE,
                    sinkhorn_repeat=self.hc_sinkhorn_iters,
                    n_splits=16,
                )
                # AITER mhc_pre_big_fuse does not fuse the decoder RMSNorm.
                return y, post.squeeze(-1), comb, False

            from sglang.kernels.ops.layernorm.mhc import mhc_pre

            norm_kwargs = {}
            if norm is not None:
                norm_kwargs["norm_weight"] = norm.weight.data
                norm_kwargs["norm_eps"] = norm.variance_epsilon

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=MHC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                **norm_kwargs,
            )
            # The HCU compatibility path inside mhc_pre dispatches to AITER's
            # pre_big_fuse_tilelang, which computes MHC pre but does not fuse
            # the decoder RMSNorm.  Keep the validated v0.5.15.post1_dev
            # contract so the caller still applies input_layernorm on HCU.
            norm_fused = norm is not None and not (_is_hcu and _use_aiter_tilelang_mhc)
            return y, post.squeeze(-1), comb, norm_fused

        if _is_hip:
            from aiter.ops.mhc import mhc_pre

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=MHC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            return y, post.squeeze(-1), comb, False

        # The deepgemm tf32 gemm wins at large M (prefill) but its fixed
        # dispatch cost dominates at small M (decode): dispatch by token count.
        if (
            envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
            and x.shape[0] >= _HC_PRENORM_DEEPGEMM_MIN_TOKENS
        ):
            from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
                tf32_hc_prenorm_gemm,
            )

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        pre, post, comb = get_mhc_ops().hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        from sglang.kernels.ops.layernorm.mhc import hc_combine

        # y is the post-norm activation fed into the MoE. Allocate it in the
        # symmetric memory pool so the downstream all-reduce uses the low-latency
        # NCCL symmetric path: the Triton inplace MoE runner writes the expert
        # output back into this buffer, so a symmetric input yields a symmetric
        # all-reduce input. Gated by is_allocation_symmetric() (mirrors the
        # TileLang path in _mhc_pre_impl / mhc_fused_post_pre).
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            y = hc_combine(x_flat, pre.squeeze(1), self.hc_mult, dtype)
        return y, post.squeeze(1), comb.squeeze(1), False

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return mhc.post(self.hc_cfg, x, residual, post, comb)

    @functools.cached_property
    def _can_fuse_ffn_mhc(self) -> bool:
        """The MoE counterpart of `_can_fuse_attn_mhc`."""
        return (
            self.mlp.tp_size == 4
            and get_moe_a2a_backend().is_none()
            and not self.mlp._shared_expert_tp1
            and not should_skip_post_experts_all_reduce(is_tp_path=True)
            and mhc.can_fuse_post(self.hc_cfg)
        )

def _scatter_tail_rows(
    tail: LateLayerTail, rows: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    # Rows outside the tail are never read (see _check_late_layer_tail_readers).
    if tail.scatter_indices is not None:
        # Prefill CUDA graph: fixed-capacity tail block; padding rows land on the
        # dummy row past the extend and are dropped by the slice.
        full = rows.new_zeros((num_tokens + 1, rows.shape[1]))
        full.index_copy_(0, tail.scatter_indices, rows)
        return full[:num_tokens]
    full = rows.new_empty((num_tokens, rows.shape[1]))
    if tail.contiguous_start is not None:
        # Bounded: a CP-padded extend (HCU) carries padding rows past the tail.
        end = tail.contiguous_start + rows.shape[0]
        full[tail.contiguous_start : end].copy_(rows)
    else:
        full[tail.token_indices] = rows[: tail.token_indices.shape[0]]
    return full

class DeepseekV4PlatformDecoderLayer(NonCudaDecoderMixin, DeepseekV4DecoderLayerBase):
    """Non-CUDA execution with platform mHC state and fusion."""

def _log_mid_chunk_decoder_skip_once(end_layer: int) -> None:
    global _logged_mid_chunk_decoder_skip
    if not _logged_mid_chunk_decoder_skip:
        _logged_mid_chunk_decoder_skip = True
        logger.info(
            "Mid-chunk prefill stops after layer %d (SGLANG_DSV41_SKIP_MID_CHUNK_DECODER)",
            end_layer - 1,
        )

def _scatter_global_tail_rows(
    tail: LateLayerTail, rows: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    # CP-gathered tail rows are in global tail order; place them in the global extend.
    full = rows.new_empty((num_tokens, *rows.shape[1:]))
    full[tail.global_token_indices] = rows
    return full

class DeepseekV4Model(NativeDeepseekV4Model):
    fall_back_to_pt_during_load = False
    # Class-level default for instances built without running __init__.
    decoder_replay_tail: Optional[DecoderReplayTail] = None

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super(NativeDeepseekV4Model, self).__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        self.hidden_size = config.hidden_size
        if self.pp_group.is_first_rank:
            embedding_quant_config = (
                quant_config
                if quant_config is not None and quant_config.get_name() == "expert_pack"
                else None
            )
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
                quant_config=embedding_quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.rms_norm_eps = config.rms_norm_eps
        # HCU retains the five-stream topology required by its dedicated DSV4
        # CP/MegaMoE path; generic HIP keeps the official opt-in behavior.
        use_stream_pool = (
            _is_hcu
            or _is_cuda
            or (
                _is_hip
                and not _is_hcu
                and (
                    envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
                    or envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
                )
            )
            or (_is_npu and envs.SGLANG_NPU_USE_MULTI_STREAM.get())
        )
        device_module = torch.get_device_module()
        num_alt_streams = 5 if (_is_cuda or _is_hcu or _is_npu) else 2
        self.alt_streams = (
            [device_module.Stream() for _ in range(num_alt_streams)]
            if use_stream_pool
            else None
        )
        # Routed-MoE input pre-quant, separate from the attention/indexer and
        # shared-expert streams.
        self.moe_routed_quant_stream = (
            device_module.Stream()
            if _is_cuda
            and config.hc_pre_from_prev_sublayer
            and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            else None
        )
        # One shared stream for all layers; every sublayer joins it before
        # reusing its residual.
        self.hc_stats_stream = (
            device_module.Stream()
            if _is_cuda
            and (get_platform().is_blackwell or get_platform().is_sm90)
            and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and config.hc_pre_from_prev_sublayer
            else None
        )
        self.engram_layout = EngramLayout.from_config(config)
        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
                engram_layout=self.engram_layout,
                hc_stats_stream=self.hc_stats_stream,
                moe_routed_quant_stream=self.moe_routed_quant_stream,
            ),
            prefix=add_prefix("layers", prefix),
            final_read=hc_boundary.FinalRead() if _is_cuda else None,
            stage_facts=functools.partial(hc_boundary.stages, config)
            if _is_cuda
            else None,
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.gemm_output_zero_allocator_size = 0
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.hc_pre_from_prev_sublayer = config.hc_pre_from_prev_sublayer
        self.hc_head_fn = self.hc_head_base = self.hc_head_scale = None
        if self.pp_group.is_last_rank and not self.hc_pre_from_prev_sublayer:
            (
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
            ) = make_hc_head_params(hc_mult, config.hidden_size)
        self.engram_hasher = None
        if self.engram_layout is not None:
            self.engram_hasher = EngramHasher.from_config(
                config,
                self.engram_layout,
                image_token_id=(
                    config.image_token_id
                    if config.model_type == "deepseek_v41"
                    and config.vision_n_layers > 0
                    else None
                ),
            )

        if not _is_cuda:
            self.use_fused_mhc_post_pre = (
                is_cross_layer_mhc_fusion_enabled()
                or _is_fused_mhc_post_pre_enabled_xpu()
            )

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size

        self.dspark_layers_to_capture: Optional[List[int]] = None

        # Decoder SWA bounded replay: layers from late_layer_start on run over each
        # request's last SWA_WINDOW extend tokens only.
        self.late_layer_start: Optional[int] = None
        # The tail starts at the last kv_source layer, whose compressed KV is first
        # written for every row (DeepseekV4AttnBackend.write_tail_source_kv).
        self.tail_writes_source_kv = False
        if get_exec().features.enable_decoder_swa_bounded_replay:
            assert config.kv_source_layer_ids, (
                "decoder SWA bounded replay needs kv_source_layer_ids"
            )
            last_source = max(config.kv_source_layer_ids)
            # ROCm starts the tail after the source layer: the gfx950 fused boundary
            # runs a whole layer in one call, and the ROCm backend carries the source
            # layer's top-k and candidates into the tail itself.
            self.tail_writes_source_kv = not _is_hip
            self.late_layer_start = (
                last_source if self.tail_writes_source_kv else last_source + 1
            )
            if self.tail_writes_source_kv:
                assert self.layers[last_source].engram is None, (
                    "the last kv_source layer must not carry an Engram"
                )
                # The tail layers read the top-k and candidates the source layer
                # publishes inside the tail; no earlier layer carries them over.
                assert (
                    last_source in config.index_source_layer_ids
                    and config.candidate_source_layer_id in (-1, last_source)
                ), (
                    "the last kv_source layer must also be the index and candidate source"
                )
            late_ratios = set(
                config.compress_ratios[last_source + 1 : config.num_hidden_layers]
            )
            assert late_ratios <= {
                0,
                1,
            }, f"late layers must not compress on their own, got ratios {late_ratios}"
            assert getattr(config, "num_hash_layers", 0) <= self.late_layer_start, (
                "decoder SWA bounded replay needs hash routing before the late layers"
            )
        # Only vision routing reads token IDs in the late layers.
        self.late_layer_vl_routing = (
            config.model_type == "deepseek_v41"
            and config.vision_n_layers > 0
            and not getattr(config, "language_model_only", False)
        )
        self.decoder_replay_tail = DecoderReplayTail.maybe_create(self)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        if x.numel() > 0:
            if _is_xpu:
                return get_mhc_ops().fused_hc_head(
                    x.contiguous(),
                    hc_fn,
                    hc_scale,
                    hc_base,
                    norm_eps=self.norm_eps,
                    hc_eps=self.hc_eps,
                )
            from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head

            return fused_hc_head(
                x.contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        return hc_head_torch(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
        )

    def _check_late_layer_tail_readers(
        self, forward_batch: ForwardBatch, tail: LateLayerTail
    ) -> None:
        # Rows outside the tail are never computed from late_layer_start on;
        # late_layer_whole_extend gives the requests that read them a whole tail.
        extend_lens = forward_batch.extend_seq_lens_cpu
        if (
            forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
            and self.dspark_layers_to_capture is None
            and list(tail.extend_seq_lens_cpu) != list(extend_lens)
        ):
            raise ValueError(
                "decoder SWA bounded replay tail misses captured hidden-state rows"
            )
        if forward_batch.return_logprob and any(
            start < n - t
            for start, n, t in zip(
                forward_batch.extend_logprob_start_lens_cpu,
                extend_lens,
                tail.extend_seq_lens_cpu,
            )
        ):
            raise ValueError(
                "decoder SWA bounded replay tail misses prompt-logprob rows"
            )

    def _forward_layers_hc_pre_from_prev(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        input_ids_global: torch.Tensor,
        capture_dspark: bool,
        dspark_aux_hidden_states: AuxHiddenStateAccumulator,
        dspark_layers_to_capture: Optional[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[LateLayerTail]]:
        assert self.pp_group.world_size == 1, "pre-mix hand-off across PP is not wired"
        hash_ids = None
        cp_extend = (
            (dsa_use_prefill_cp(forward_batch) if _is_hcu else is_cp_active(forward_batch))
            and forward_batch.forward_mode.is_extend()
        )
        if self.engram_hasher is not None:
            if cp_extend:
                # n-gram hashing needs each token's predecessors: hash the whole prompt
                # Legacy HCU round-robin metadata has total_seq_lens=0. Keep its
                # global execution padding, matching forward_batch.positions;
                # EngramHasher masks padding and commits only real requests.
                total = (
                    forward_batch.input_ids.shape[0]
                    if _is_hcu
                    else int(forward_batch.attn_cp_metadata.total_seq_lens)
                )
                hash_ids = self.engram_hasher(
                    forward_batch.input_ids[:total], forward_batch
                )
                parallel = get_parallel()
                hash_ids = hash_ids[parallel.attn_cp_rank :: parallel.attn_cp_size]
                pad_rows = hidden_states.shape[0] - hash_ids.shape[0]
                if pad_rows > 0:
                    hash_ids = torch.cat(
                        [hash_ids, hash_ids.new_zeros(pad_rows, *hash_ids.shape[1:])]
                    )
            elif (
                forward_batch.forward_mode.is_extend() and is_in_breakable_cuda_graph()
            ):
                hash_ids = bcg_deepseek_v4_engram_hash_ids(
                    self.engram_hasher, input_ids
                )
            else:
                hash_ids = self.engram_hasher(input_ids, forward_batch)
        tail = None
        end_layer = self.end_layer
        if (
            self.late_layer_start is not None
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            attn_backend = get_attn_backend()
            if getattr(attn_backend, "skip_late_layers", False):
                # A mid-chunk prefill: nothing reads the later layers' rows.
                end_layer = self.late_layer_start
                _log_mid_chunk_decoder_skip_once(end_layer)
            else:
                tail = attn_backend.tail_forward_metadata.late_layer_tail
                self._check_late_layer_tail_readers(forward_batch, tail)
        tail_in_graph = (
            tail is not None
            and tail.scatter_indices is not None
            and is_in_breakable_cuda_graph()
        )
        late_dp_counts = None
        saved_dp = None
        if (
            self.late_layer_start is not None
            and get_parallel().attn_dp_size > 1
            # The prefill graph's capture batch leaves is_extend_in_batch unset.
            and (forward_batch.is_extend_in_batch or tail_in_graph or _is_hip)
        ):
            late_dp_counts = late_layer_dp_counts(
                forward_batch,
                (
                    tail.token_indices.shape[0]
                    if tail is not None
                    else hidden_states.shape[0]
                ),
                in_graph=tail_in_graph,
            )
        use_tail_graph = (
            self.decoder_replay_tail is not None
            and tail is not None
            and self.decoder_replay_tail.plan(
                tail, num_tokens=input_ids.shape[0], tail_in_graph=tail_in_graph
            )
        )
        tail_sentinel = positions.new_zeros((1,)) if tail_in_graph else None
        saved_full = None
        # The HIP defer_post gate excludes residual readers. CUDA materializes
        # its producer update explicitly before these readers below.
        # mHC assumes full token rows per rank; LayerNorm SP needs its own path.
        assert not get_forward().sp_active
        state = mhc.HcState(hidden_states)
        if _is_cuda:
            residual_batch.start(forward_batch)
            hidden_states = residual_batch.set_written(state, forward_batch)
        for i in range(self.start_layer, end_layer):
            if _is_cuda:
                layer = self.layers[i]
                if (
                    (tail is not None and i == self.late_layer_start)
                    or layer.engram is not None
                    or (capture_dspark and i in self.dspark_layers_to_capture)
                ):
                    call = hc_boundary.LayerCall(
                        input_ids,
                        input_ids_global,
                    )
                    next_boundary = (
                        hc_boundary.HcRead(layer, True)
                        .bind(forward_batch, call)
                        .next_boundary
                    )
                    hidden_states = state = hc_boundary.materialize(
                        hidden_states, forward_batch, next=next_boundary
                    )
            if tail is not None and i == self.late_layer_start:
                if self.tail_writes_source_kv:
                    source = self.layers[i]
                    attn_backend.write_tail_source_kv(
                        layer=source.self_attn,
                        x=mhc.combine(source.attn_hc, state),
                        positions=positions,
                        forward_batch=forward_batch,
                    )
                # Decode reaches back at most SWA_WINDOW positions.
                if tail_in_graph:
                    # Eager break: re-run at replay so the backend swaps onto the
                    # live batch's tail metadata; the gathers below are captured
                    # against the persistent index buffers.
                    tail_sentinel = bcg_deepseek_v4_enter_late_layer_tail(tail_sentinel)
                    saved_full = attn_backend.prefill_graph_late_layer_saved
                else:
                    saved_full = attn_backend.enter_late_layer_tail(forward_batch)
                state = state.take_rows(tail.rows)
                if _is_cuda:
                    hidden_states = residual_batch.set_written(state, forward_batch)
                input_ids = tail.rows(input_ids)
                if not (
                    get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none()
                ):
                    input_ids_global = tail.rows(input_ids_global)
                positions = tail.positions
                if hash_ids is not None:
                    hash_ids = tail.rows(hash_ids)
                if use_tail_graph:
                    residual, pre, aux = self.decoder_replay_tail.run(
                        state=state,
                        positions=positions,
                        input_ids=input_ids,
                        input_ids_global=input_ids_global,
                        forward_batch=forward_batch,
                        capture_dspark=capture_dspark,
                    )
                    dspark_aux_hidden_states.extend(aux)
                    residual_batch.clear(forward_batch)
                    attn_backend.exit_late_layer_tail(saved_full, forward_batch)
                    return residual, pre, tail
            if late_dp_counts is not None and i == self.late_layer_start:
                saved_dp = enter_late_layer_dp(forward_batch, late_dp_counts)
                if get_moe_a2a_backend().is_none():
                    # Full-prefill IDs are not indexed by this rank's tail rows.
                    input_ids_global = (
                        _gather_late_layer_ids(input_ids, late_dp_counts, forward_batch)
                        if self.late_layer_vl_routing
                        else None
                    )
            engram = self.layers[i].engram
            if engram is not None:
                before_engram = state.residual
                hidden_states = engram(
                    state.residual,
                    hash_ids[:, engram.layer_hash_index],
                    forward_batch,
                    cp_all_tokens=cp_extend,
                )
                if (
                    self.config.model_type == "deepseek_v41"
                    and self.config.vision_n_layers > 0
                ):
                    hidden_states = torch.where(
                        (input_ids == self.config.image_token_id)[:, None, None],
                        before_engram,
                        hidden_states,
                    )
                state = state.with_residual(hidden_states)
                if _is_cuda:
                    hidden_states = residual_batch.set_written(state, forward_batch)
            if capture_dspark and i in dspark_layers_to_capture:
                # The draft head reads the attention input of its target layers.
                aux = state.residual
                if tail is not None and i < self.late_layer_start:
                    aux = tail.rows(aux)
                dspark_aux_hidden_states.append(aux.mean(dim=1))
            ctx = (
                nullcontext()
                if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                else get_global_expert_distribution_recorder().with_current_layer(i)
            )
            with ctx:
                if _is_cuda:
                    hidden_states = self.layers[i](
                        positions=positions,
                        hidden_states=hidden_states,
                        input_ids=input_ids,
                        forward_batch=forward_batch,
                        input_ids_global=input_ids_global,
                    )
                else:
                    if _is_hip and self.layers[i].hc_boundary_fused:
                        state = forward_layer_fused_boundary(
                            self,
                            i,
                            state,
                            positions=positions,
                            input_ids=input_ids,
                            forward_batch=forward_batch,
                            input_ids_global=input_ids_global,
                            capture_dspark=capture_dspark,
                        )
                        continue
                    state = self.layers[i].forward_hc_pre_from_prev(
                        positions=positions,
                        state=state,
                        input_ids=input_ids,
                        forward_batch=forward_batch,
                        input_ids_global=input_ids_global,
                    )
        if _is_cuda:
            state = residual_batch.read_output(
                hidden_states, forward_batch, hc_boundary.FinalRead(), None
            )
            residual_batch.clear(forward_batch)
        # The last run layer may have deferred its post to the next one.
        state = state.materialized(self.layers[end_layer - 1].hc_cfg)
        if end_layer < self.end_layer and capture_dspark:
            # Keep the capture list whole; these rows precede every SWA window
            # that later reads them, the DSpark draft's included.
            for i in sorted(dspark_layers_to_capture):
                if i >= end_layer:
                    dspark_aux_hidden_states.append(state.residual.mean(dim=1))
        if saved_dp is not None:
            exit_late_layer_dp(forward_batch, saved_dp)
        if saved_full is not None:
            if tail_in_graph:
                tail_sentinel = bcg_deepseek_v4_exit_late_layer_tail(tail_sentinel)
            else:
                attn_backend.exit_late_layer_tail(saved_full, forward_batch)
            return state.residual, state.pre, tail
        return state.residual, state.pre, None

    def _can_run_tbo(self, forward_batch: ForwardBatch) -> bool:
        """DSV4 prefill-only two-batch-overlap gate.

        TBO batch prep (tbo_split_seq_index / tbo_children) is populated
        model-agnostically when --enable-two-batch-overlap is set and the
        DP-attention preparer allows it (mori `normal` mode permits prefill
        TBO). We additionally restrict to prefill (EXTEND), single PP, and
        non-CP paths supported by the DSV4 op strategy.
        """
        from sglang.srt.layers.moe import is_tbo_enabled

        path_ok = not is_cp_active(forward_batch) and (
            not _is_hip
            or not get_moe_a2a_backend().is_none()
            or get_parallel().attn_dp_size > 1
        )
        return (
            is_tbo_enabled()
            and forward_batch.can_run_tbo
            and forward_batch.tbo_children is not None
            and forward_batch.global_forward_mode is not None
            # MTP target-verify also reports is_extend(); only real prefill
            # should enter the prefill TBO strategy.
            and forward_batch.global_forward_mode.is_extend_without_speculative()
            and path_ok
            and self.pp_group.world_size == 1
        )

    def _forward_layers_tbo(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.batch_overlap.operations import execute_overlapped_operations
        from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
        from sglang.srt.batch_overlap.two_batch_overlap import (
            _model_forward_filter_inputs,
            _model_forward_tbo_merge_key,
        )

        layers = [self.layers[i] for i in range(self.start_layer, self.end_layer)]
        operations_strategy = OperationsStrategy.init_new_tbo(
            layers, forward_batch.global_forward_mode
        )

        # Split the per-rank batch into the 2 ubatches (token-range slice + pad
        # to tbo_padded_len). residual is unused by the DSV4 non-fused layer ops.
        inputs_arr = [
            _model_forward_filter_inputs(
                hidden_states=hidden_states,
                residual=None,
                positions=positions,
                output_forward_batch=child,
                tbo_subbatch_index=idx,
            )
            for idx, child in enumerate(forward_batch.tbo_children)
        ]

        # Non-EP DP TP-MoE: the per-ubatch DP gather/combine (op_gather/op_combine)
        # needs each ubatch's per-rank token counts, but tbo_padded_len is computed
        # per-rank locally (not synced). All-gather both ubatches' padded lengths
        # once across DP ranks, then populate each child's global_num_tokens +
        # global_dp_buffer_len so the gatherv/reduce_scatterv buffers size correctly.
        if get_moe_a2a_backend().is_none() and get_parallel().attn_dp_size > 1:
            tp_group = get_parallel().tp_group
            world = tp_group.world_size
            children = forward_batch.tbo_children
            local_lens = torch.tensor(
                [int(c.tbo_padded_len) for c in children],
                dtype=torch.int64,
                device=hidden_states.device,
            )
            gathered = torch.empty(
                (world, local_lens.shape[0]),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            tp_group.all_gather_into_tensor(gathered, local_lens)
            gathered_cpu = gathered.tolist()
            rank = tp_group.rank_in_group
            for idx, child in enumerate(children):
                sizes = [gathered_cpu[r][idx] for r in range(world)]
                child.global_num_tokens_cpu = sizes
                child.global_num_tokens_gpu = gathered[:, idx].contiguous()
                child.global_dp_buffer_len = sum(sizes)
                # Gather the ubatch's input_ids -> global ONCE here (cached on the
                # child) instead of per-layer in op_gather_a. The hash MoE reads
                # the SAME global ids every layer, so 61x2 per-layer all_gatherv of
                # VARYING size (-> RCCL registers a new internal buffer per size ->
                # HSA_STATUS_ERROR_OUT_OF_RESOURCES) collapses to 1 per ubatch.
                local_ids = child.input_ids
                rows = sizes[rank]
                if local_ids.shape[0] < rows:
                    padded_ids = local_ids.new_zeros((rows,))
                    padded_ids[: local_ids.shape[0]] = local_ids
                elif local_ids.shape[0] > rows:
                    padded_ids = local_ids[:rows]
                else:
                    padded_ids = local_ids
                gids = torch.empty(
                    (sum(sizes),), dtype=local_ids.dtype, device=local_ids.device
                )
                tp_group.all_gatherv(padded_ids, sizes=sizes, output=gids)
                child._tbo_global_input_ids = gids

        try:
            outputs_arr = execute_overlapped_operations(
                inputs_arr=inputs_arr,
                operations_arr=[operations_strategy.operations] * 2,
                delta_stages=[0, operations_strategy.tbo_delta_stages],
            )

            return _model_forward_tbo_merge_key(
                outputs_arr[0], outputs_arr[1], "hidden_states", hidden_states.shape[0]
            )
        finally:
            if _is_cuda:
                for child in forward_batch.tbo_children:
                    residual_batch.clear(child)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if not _is_cuda:
            return self._forward_impl(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        try:
            return self._forward_impl(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        finally:
            residual_batch.clear(forward_batch)

    @torch.no_grad()
    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        cp_v2_active = is_cp_v2_active(forward_batch)
        use_prefill_cp = (
            dsa_use_prefill_cp(forward_batch)
            if _is_hcu
            else is_cp_active(forward_batch)
        )
        incoming_pd_aux_hidden_states: List[torch.Tensor] = []
        local_dspark_aux_hidden_states: List[torch.Tensor] = []
        deferred_mhc_input = None
        mhc_cp_split_done = False
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            if (
                _FUSE_MHC_REPEAT_CP_SPLIT
                and _is_hcu
                and use_prefill_cp
                and not cp_v2_active
                and is_dsa_prefill_cp_round_robin_split()
                and forward_batch.forward_mode.is_extend_without_speculative()
                and _can_defer_mhc_repeat_cp_split(
                    hidden_states,
                    forward_batch,
                    self.hc_mult,
                    get_parallel().attn_cp_size,
                    get_parallel().attn_cp_rank,
                )
            ):
                # The intervening DP gather consumes input_ids, not hidden
                # states. Defer until run_tbo decides whether full rows are needed.
                deferred_mhc_input = hidden_states
            else:
                from sglang.kernels.ops.layernorm.mhc import hc_broadcast

                hidden_states = hc_broadcast(hidden_states, self.hc_mult)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            incoming_pd_aux_hidden_states = [
                pp_proxy_tensors[key]
                for key in sorted(
                    key
                    for key in pp_proxy_tensors.tensors
                    if key.startswith("pd_aux_hidden_states_")
                )
            ]
            # Unflatten 2D PP IPC tensor back to 3D mHC shape.
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.view(
                    hidden_states.shape[0], self.hc_mult, self.hidden_size
                )

        if get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (get_global_dp_buffer_len(), 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            # Token ids are replicated within an attention-TP group. Use replicate
            # gather here to avoid summing duplicated ids when attention_tp_size > 1.
            # Clone because the MAX_LEN gather may zero its local input in place.
            dp_gather_replicate(
                input_ids_global, input_ids[:, None].clone(), forward_batch
            )
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = getattr(forward_batch, "input_ids_global", input_ids)

        attn_backend = get_attn_backend()
        if _is_npu and forward_batch.attn_cp_metadata is not None:
            attn_backend.prepare_dsv4_cp_metadata(forward_batch)
            local_positions = getattr(forward_batch, "dsv4_cp_local_positions", None)
            if (
                local_positions is not None
                and positions.shape[0] == local_positions.shape[0]
            ):
                forward_batch.positions = positions

        # PD disaggregation drives the capture set per batch; fall back to the
        # model-level set for the non-disaggregated path.
        dspark_layers_to_capture = getattr(
            forward_batch, "pd_hidden_capture_layer_ids", None
        )
        if dspark_layers_to_capture is None:
            dspark_layers_to_capture = self.dspark_layers_to_capture
        capture_dspark = dspark_layers_to_capture is not None
        # Under CP-v2 the body returns rank-local hidden/aux rows; the CP runner
        # gathers them and finishes through DeepseekV4ForCausalLM.logits_from_body_output.
        use_packed_pd_aux = capture_dspark and self.pp_group.world_size == 1
        pd_aux_hidden_states: AuxHiddenStateAccumulator = (
            AuxHiddenStatePacker(len(dspark_layers_to_capture))
            if use_packed_pd_aux
            else list(incoming_pd_aux_hidden_states)
        )
        # DSpark aux capture needs the per-layer eager loop (TBO's overlapped
        # execution cannot expose per-layer completed hidden states), so skip
        # TBO when capturing -- a perf-only downgrade, not a correctness one.
        run_tbo = self._can_run_tbo(forward_batch) and not capture_dspark
        if deferred_mhc_input is not None:
            if not run_tbo:
                hidden_states = _repeat_mhc_input_on_cp_rank(
                    deferred_mhc_input,
                    self.hc_mult,
                    get_parallel().attn_cp_size,
                    get_parallel().attn_cp_rank,
                )
                mhc_cp_split_done = True
            else:
                # TBO partitions full rows into children before their CP split.
                hidden_states = deferred_mhc_input.unsqueeze(1).repeat(
                    1, self.hc_mult, 1
                )
            deferred_mhc_input = None
        if use_prefill_cp and not run_tbo:
            if cp_v2_active:
                # Generic CP already shards model inputs in the runner.
                pass
            else:
                if self.pp_group.is_first_rank and not mhc_cp_split_done:
                    hidden_states = cp_split_and_rebuild_data(
                        forward_batch, hidden_states
                    )
                positions = cp_split_and_rebuild_position(forward_batch, positions)
                input_ids = cp_round_robin_input_ids(input_ids)
            input_ids_global = input_ids

        # Reset Compressor's per-step freqs_cis cache from any previous step.
        for _attr in ("freqs_cis_c4", "freqs_cis_c128"):
            if hasattr(forward_batch, _attr):
                delattr(forward_batch, _attr)

        if _is_npu and not run_tbo:
            # Rope cos/sin for the whole forward: one bf16 gather per rope
            # config on the current stream, before the layer loop forks the
            # KV/Q side streams. TBO children carry their own positions and
            # recompute per layer.
            prime_rope_cos_sin(
                (
                    self.layers[i].self_attn
                    for i in range(self.start_layer, self.end_layer)
                ),
                forward_batch,
                positions,
            )
        last_pre = None
        tail = None
        if self.hc_pre_from_prev_sublayer:
            assert not run_tbo, "two-batch overlap is not wired for this hc scheme"
            hidden_states, last_pre, tail = self._forward_layers_hc_pre_from_prev(
                positions,
                hidden_states,
                forward_batch,
                input_ids,
                input_ids_global,
                capture_dspark,
                pd_aux_hidden_states,
                dspark_layers_to_capture,
            )
        elif run_tbo:
            # Two-batch-overlap prefill (EP / mori). Cross-layer mHC fusion is
            # disabled here (each layer self-contained), so no trailing hc_post.
            hidden_states = self._forward_layers_tbo(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            if _is_cuda:
                residual_batch.start(forward_batch)
            use_fused = not _is_cuda and self.use_fused_mhc_post_pre
            prev_residual, prev_post, prev_comb = None, None, None
            last_layer = None
            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                last_layer = layer
                ctx = (
                    nullcontext()
                    if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    if _is_cuda:
                        hidden_states = layer(
                            positions=positions,
                            hidden_states=hidden_states,
                            forward_batch=forward_batch,
                            input_ids=input_ids,
                            input_ids_global=input_ids_global,
                        )
                    else:
                        hidden_states, prev_residual, prev_post, prev_comb = layer(
                            positions=positions,
                            hidden_states=hidden_states,
                            forward_batch=forward_batch,
                            input_ids=input_ids,
                            input_ids_global=input_ids_global,
                            prev_residual=prev_residual,
                            prev_post=prev_post,
                            prev_comb=prev_comb,
                        )
                if capture_dspark and i in dspark_layers_to_capture:
                    if _is_cuda:
                        completed = hc_boundary.materialize(
                            hidden_states, forward_batch, consume=False
                        )
                    elif use_fused:
                        completed = layer.hc_post(
                            hidden_states, prev_residual, prev_post, prev_comb
                        )
                    else:
                        completed = hidden_states
                    captured_hidden = completed.mean(dim=1)
                    pd_aux_hidden_states.append(captured_hidden)
                    local_dspark_aux_hidden_states.append(captured_hidden)
            if _is_cuda:
                hidden_states = residual_batch.read_output(
                    hidden_states, forward_batch, hc_boundary.FinalRead(), None
                )
            elif use_fused and last_layer is not None:
                hidden_states = last_layer.hc_post(
                    hidden_states, prev_residual, prev_post, prev_comb
                )

        # CP all-gather only on the last PP rank; PP IPC carries CP-split tensors.
        # CP v2 partitions the output per rank, so it needs no output re-gather,
        # and TBO never applied the CP split in the first place.
        cp_gather_output = (
            use_prefill_cp
            and not cp_v2_active
            and not run_tbo
            and self.pp_group.is_last_rank
        )

        # With PP+D-Spark, each stage projects only its own capture features.
        # The final stage receives the prior projected accumulator separately,
        # so its logits output must contain local features rather than the old
        # concatenated raw-hidden relay.
        if (
            capture_dspark
            and self.pp_group.world_size > 1
            and self.pp_group.is_last_rank
        ):
            pd_aux_hidden_states = local_dspark_aux_hidden_states

        if isinstance(pd_aux_hidden_states, AuxHiddenStatePacker):
            pd_aux_hidden = pd_aux_hidden_states.finalize()
        else:
            pd_aux_hidden = pd_aux_hidden_states

        if not self.pp_group.is_last_rank:
            # Flatten 3D mHC tensor for PP IPC.
            proxy_tensors = {"hidden_states": hidden_states.flatten(1)}
            if capture_dspark:
                for idx, aux_hidden in enumerate(pd_aux_hidden_states):
                    proxy_tensors[f"pd_aux_hidden_states_{idx}"] = (
                        aux_hidden.flatten(1) if aux_hidden.ndim == 3 else aux_hidden
                    )
                if local_dspark_aux_hidden_states:
                    proxy_tensors["dspark_aux_hidden_states"] = torch.cat(
                        local_dspark_aux_hidden_states, dim=-1
                    )
                else:
                    proxy_tensors["dspark_aux_hidden_states"] = hidden_states.new_empty(
                        hidden_states.shape[0], 0
                    )
            return PPProxyTensors(proxy_tensors)

        pre_hc_head = hidden_states.flatten(1)
        if self.hc_pre_from_prev_sublayer:
            from sglang.kernels.ops.layernorm.mhc import hc_combine

            hidden_states = hc_combine(
                pre_hc_head.float(), last_pre, self.hc_mult, hidden_states.dtype
            )
        else:
            hidden_states = self.hc_head(
                hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
            )
        hidden_states = self.norm(hidden_states)
        # HCU prefill CP left the decoder tail rank-local: gather it under its own
        # layout, which yields the tail rows in global order.
        hcu_cp_tail = (
            cp_gather_output and tail is not None and tail.cp_metadata is not None
        )
        if cp_gather_output:
            stream = torch.cuda.current_stream()
            full_cp_metadata = forward_batch.attn_cp_metadata
            if hcu_cp_tail:
                forward_batch.attn_cp_metadata = tail.cp_metadata
            pre_hc_head = cp_all_gather_rerange_output(
                pre_hc_head, self.cp_size, forward_batch, stream
            )
            hidden_states = cp_all_gather_rerange_output(
                hidden_states, self.cp_size, forward_batch, stream
            )
            if capture_dspark:
                pd_aux_hidden = (
                    [
                        cp_all_gather_rerange_output(
                            aux, self.cp_size, forward_batch, stream
                        )
                        for aux in pd_aux_hidden
                    ]
                    if isinstance(pd_aux_hidden, list)
                    else cp_all_gather_rerange_output(
                        pd_aux_hidden, self.cp_size, forward_batch, stream
                    )
                )
            forward_batch.attn_cp_metadata = full_cp_metadata

        if tail is not None and not capture_dspark:
            # The logits processor indexes rows by the full extend layout.
            if hcu_cp_tail:
                num_tokens = forward_batch.input_ids.shape[0]
                hidden_states = _scatter_global_tail_rows(
                    tail=tail, rows=hidden_states, num_tokens=num_tokens
                )
                pre_hc_head = _scatter_global_tail_rows(
                    tail=tail, rows=pre_hc_head, num_tokens=num_tokens
                )
            else:
                num_tokens = input_ids.shape[0]
                hidden_states = _scatter_tail_rows(
                    tail=tail, rows=hidden_states, num_tokens=num_tokens
                )
                pre_hc_head = _scatter_tail_rows(
                    tail=tail, rows=pre_hc_head, num_tokens=num_tokens
                )

        if capture_dspark:
            return (hidden_states, pre_hc_head), pd_aux_hidden

        return hidden_states, pre_hc_head

def _should_disable_v41_vision_for_hcu(config: DeepSeekV4Config) -> bool:
    if not (
        _is_hcu
        and config.model_type == "deepseek_v41"
        and config.vision_n_layers > 0
    ):
        return False

    parallel = get_parallel()
    return (
        get_disagg().disaggregation_mode in ("prefill", "decode")
        or parallel.attn_cp_size != 1
        or parallel.pp_group.world_size != 1
        or not (
            get_moe_a2a_backend().is_none()
            or get_moe_a2a_backend().is_deepep()
        )
    )

class DeepseekV4ForCausalLM(NativeDeepseekV4ForCausalLM):
    supports_cuda_vmm_feature_transport = True

    @classmethod
    def get_hf_to_sglang_mapper(cls, config) -> Optional[WeightsMapper]:
        # The loader hands this mapper to every quant config; other formats were
        # written against different names, so only Quark checkpoints get it.
        return _QUARK_CKPT_TO_SGLANG_MAPPER if _is_quark_checkpoint(config) else None

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super(NativeDeepseekV4ForCausalLM, self).__init__()
        # DeepseekV4 enables, by default, the CK w8a8-block GEMM (MLA proj) and the
        # batched/contiguous-load rope kernels (faster on gfx95; .
        # Module-level toggles default OFF; flipped True here for DSV4
        if _is_hip:
            from sglang.kernels.ops.attention.deepseek_v4_rope import set_batched_rope
            from sglang.srt.layers.quantization.fp8_utils import (
                set_force_ck_w8a8,
            )

            set_force_ck_w8a8(True)
            set_batched_rope(True)
            # TODO(aiter#6004): remove once ROCm/aiter#6004 is in the aiter pin
            # TODO(aiter#6029): remove once ROCm/aiter#6029 is in the aiter pin
            # TODO(aiter#6030): remove once ROCm/aiter#6030 is in the aiter pin
            from hcu_sglang.models.deepseek.current_v4.execution.aiter_intree_tune import (
                install_aiter_intree_tune,
            )

            install_aiter_intree_tune()
            if envs.SGLANG_DSV4_DENSE_MXSCALE.get():
                from sglang.srt.layers.quantization.fp8_utils import set_dense_mxscale

                set_dense_mxscale(True)
        if _should_disable_v41_vision_for_hcu(config):
            logger.info(
                "Disabling the DeepSeek-V4.1 vision tower for this HCU text "
                "serving topology; the V4.1 vision path does not support PD, "
                "CP, PP or MoE A2A."
            )
            config.vision_n_layers = 0

        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.wo_a_fp8 = wo_a_fp8_gemm_enabled(quant_config)
        self.determine_num_fused_shared_experts()
        self.vision = None
        if (
            config.model_type == "deepseek_v41"
            and config.vision_n_layers > 0
            and not getattr(config, "language_model_only", False)
        ):
            if (
                get_parallel().attn_cp_size != 1
                or get_parallel().pp_group.world_size != 1
                or not (
                    get_moe_a2a_backend().is_none()
                    or get_moe_a2a_backend().is_deepep()
                    or (_is_cuda and get_moe_a2a_backend().is_megamoe())
                )
            ):
                raise ValueError(
                    "V4.1 vision supports TP/EP/DP without CP or PP; "
                    "MoE A2A requires DeepEP or CUDA MegaMoE"
                )

            args = SimpleNamespace(**vars(config), dim=config.hidden_size)
            self.vision = ViT(args)
            self.aligner = Aligner(args)
            self.image_start = nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_parallel().pp_group
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.model.start_layer, self.model.end_layer)
                if isinstance(
                    self.model.layers[layer_id].mlp, deepseek_v2.DeepseekV2MoE
                )
            }
        )

        # Expose start_layer/end_layer for model_runner PP support
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer

        # update_weights_from_disk/_tensor/_distributed re-enter load_weights
        # mid-serving (RL refit sends many partial batches); the prewarm and
        # its barrier must only run on the first (startup) load.
        self._mhc_prewarmed_at_load = False

    @torch.inference_mode()
    def wants_prefill_autotune(self) -> bool:
        return getattr(self.config, "model_type", None) == "deepseek_v41"

    def autotune_prefill_kernels(self, num_tokens: int, *, dtype: torch.dtype) -> int:
        """Tune resident MXFP8 linears for every M bucket up to ``num_tokens``.
        The quant method is called directly, so no TP collectives run and no
        request/KV/draft state is touched; the runner owns the autotune context."""
        if getattr(self.config, "model_type", None) != "deepseek_v41":
            return 0
        seen = set()
        # The backbone excludes vision and lm_head, whose prefill shapes differ.
        for layer in self.model.modules():
            method = getattr(layer, "quant_method", None)
            if not isinstance(method, Fp8LinearMethod):
                continue
            if not (method.use_mxfp8 or method.block_fp8_as_mxfp8):
                continue
            if method.block_fp8_as_mxfp8 and not getattr(
                layer, "block_fp8_mxfp8_ready", False
            ):
                # No swizzled MXFP8 scale buffer: these kept the block-FP8 fallback.
                continue
            backend = method.mxfp8_dense_backend
            if backend is None or not backend.is_flashinfer_cutedsl():
                continue
            if method.block_fp8_as_mxfp8:
                # Small shapes and deterministic execution keep their pinned tactic.
                method.mxfp8_prefill_autotune_min_tokens = 4096
            weight = layer.weight
            scale = layer.weight_scale_inv_swizzled
            key = (
                weight.shape,
                weight.stride(),
                weight.dtype,
                scale.shape,
                scale.stride(),
                scale.dtype,
            )
            if key in seen:
                continue
            seen.add(key)
            x = torch.zeros(
                (num_tokens, weight.shape[1]),
                dtype=dtype,
                device=weight.device,
            )
            method.apply(layer, x)
            del x
        if seen:
            logger.info(
                "FlashInfer prefill autotune: %d MXFP8 weight layouts at M=%d.",
                len(seen),
                num_tokens,
            )
        return len(seen)

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def pad_input_ids(self, input_ids: array, mm_inputs) -> array:
        return pad_mm_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items):
        """Return complete spans for the shared MM cache and chunk scheduler."""

        spans = []
        device, dtype = self.image_start.device, self.image_start.dtype
        for item in items:
            item.reconstruct(device.index)
            h, w = int(item.n_vit_h), int(item.n_vit_w)
            pixels = torch.as_tensor(item.feature, device=device)
            plan = item.model_specific_data.get(GPU_PLAN_KEY)
            patches = (
                materialize_image_gpu(pixels, plan).to(dtype)
                if plan is not None
                else pixels.to(dtype)
            )
            features = self.aligner(self.vision(patches, h, w), h, w)
            r = self.config.vision_downsample_ratio
            types = image_token_types((h + r - 1) // r, (w + r - 1) // r).to(device)
            span = torch.empty(
                (len(types), self.config.hidden_size), device=device, dtype=dtype
            )
            span[types == 0] = self.image_start
            span[types == 1] = features.to(dtype)
            span[types == 2] = self.image_newline
            span[types == 3] = self.image_end
            spans.append(span)
        return spans

    def _prepare_mm_embeddings(self, input_ids, forward_batch):
        # Keep scheduler hash IDs intact: the shared embedder clamps its input in place.
        input_embeds, _ = embed_mm_inputs(
            mm_inputs_list=[
                item if item is not None else MultimodalInputs(mm_items=[])
                for item in forward_batch.mm_inputs
            ],
            extend_prefix_lens=forward_batch.extend_prefix_lens_cpu,
            extend_seq_lens=forward_batch.extend_seq_lens_cpu,
            input_ids=input_ids.clone(),
            input_embedding=self.get_input_embeddings(),
            multimodal_model=self,
        )
        forward_batch.mm_input_embeds = input_embeds
        return input_embeds

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_dspark_layers_to_capture(self, layer_ids: List[int]) -> None:
        if layer_ids is None:
            raise ValueError(
                "DSPARK requires explicit layer_ids for aux hidden capture."
            )
        local_layer_ids = [
            int(layer_id)
            for layer_id in layer_ids
            if self.model.start_layer <= int(layer_id) < self.model.end_layer
        ]
        self.capture_aux_hidden_states = bool(local_layer_ids)
        self.model.dspark_layers_to_capture = local_layer_ids or None

    @classmethod
    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        """V4 only fuses when explicitly asked to, and then the checkpoint must
        carry exactly one shared expert. Asked by the loader before any layer is
        built."""
        # Need to disable if quant precision mismatch, even if
        # --enforce-shared-experts-fusion is specified
        if quant_blocks_shared_experts_fusion(quant_config):
            return (
                "Quantization keeps shared experts at a higher precision than the "
                "routed experts, so they cannot be fused into the quantized "
                "routed-expert path."
            )
        if get_parallel().moe_ep_size > 1 and not uses_per_rank_fused_shared_slots():
            return (
                "Expert parallelism keeps only a slice of the routed experts on "
                "each rank, so the fused shared expert cannot be appended to the "
                "routed weight tensor (only DeepEP/MegaMOE per-rank shared slots "
                "support fusion under EP)."
            )
        if not get_exec().moe.enforce_shared_experts_fusion:
            return "Config does not support fused shared expert(s)."
        if hf_config.n_shared_experts != 1:
            raise ValueError(
                "DeepSeek V4 shared-experts fusion expects exactly one shared "
                f"expert, but got n_shared_experts={hf_config.n_shared_experts}."
            )
        return None

    def determine_num_fused_shared_experts(self):
        # The decision was installed by the loader; this only reads it.
        self.num_fused_shared_experts = (
            0 if is_shared_experts_fusion_disabled() else self.config.n_shared_experts
        )

    def prepare_language_model_inputs(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare full-sequence image embeddings and model IDs before CP splits.

        Scheduler hash IDs stay intact for multimodal cache keys; the language
        model uses image_token_id for Engram masking and visual MoE routing.
        """
        if (
            getattr(self, "vision", None) is not None
            and not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.mm_inputs is not None
            and any(x is not None for x in forward_batch.mm_inputs)
        ):
            if input_embeds is not None:
                raise ValueError("Cannot combine input_embeds and image inputs")
            input_embeds = self._prepare_mm_embeddings(input_ids, forward_batch)
        if getattr(self, "vision", None) is not None and not (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
        ):
            # Decode/verify IDs are already vocabulary IDs; remap prompt image
            # hashes for Engram and routing.
            input_ids = input_ids.masked_fill(
                input_ids >= MM_PAD_SHIFT_VALUE, self.config.image_token_id
            )

        if not enable_cp_v2() and self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
                if is_dsa_prefill_cp_round_robin_split():
                    attn_backend = get_attn_backend()
                    metadata = attn_backend.forward_metadata
                    core_meta = metadata.core_attn_metadata
                    core_meta.apply_cp_reindex()
                    core_meta.init_flashmla_related(is_prefill=True)
                    if metadata.indexer_metadata is not None:
                        metadata.indexer_metadata = (
                            attn_backend.init_forward_metadata_indexer(core_meta)
                        )
                    if getattr(attn_backend, "tail_forward_metadata", None) is not None:
                        # decoder SWA bounded replay: the tail was built before the split
                        convert = getattr(
                            attn_backend, "convert_late_layer_tail_to_hcu_cp", None
                        )
                        if convert is None:
                            raise NotImplementedError(
                                "decoder SWA bounded replay under HCU prefill CP is "
                                f"not wired for {type(attn_backend).__name__}"
                            )
                        convert(forward_batch)
                elif (
                    getattr(get_attn_backend(), "tail_forward_metadata", None)
                    is not None
                ):
                    raise NotImplementedError(
                        "decoder SWA bounded replay supports only the round-robin "
                        "(interleave) HCU prefill CP split"
                    )

        return input_ids, input_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        input_ids, input_embeds = self.prepare_language_model_inputs(
            input_ids, forward_batch, input_embeds
        )
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_from_body_output(input_ids, hidden_states, forward_batch)

    def logits_from_body_output(
        self,
        input_ids: torch.Tensor,
        hidden_states,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Finish a forward from the (CP-gathered) body output.

        Shared by the eager forward and the CP-v2 eager / breakable-graph
        runners so all of them apply the same DSpark and PD aux contract.
        ``hidden_states`` is ``(hidden, pre_hc_head)`` or, when aux capture is
        on, ``((hidden, pre_hc_head), aux_hidden_states)``.
        """
        aux_hidden_states = None
        pd_aux_hidden_states = None
        has_pd_hidden_capture = (
            getattr(forward_batch, "pd_hidden_capture_layer_ids", None) is not None
        )
        if has_pd_hidden_capture:
            hidden_states, pd_aux_hidden_states = hidden_states
            if self.capture_aux_hidden_states:
                aux_hidden_states = pack_aux_hidden_states(pd_aux_hidden_states)
        elif self.capture_aux_hidden_states:
            hidden_states, captured_aux_hidden_states = hidden_states
            aux_hidden_states = pack_aux_hidden_states(captured_aux_hidden_states)
            # Eager/static forward-batch wrappers can preserve the global
            # DSpark capture plan while omitting the per-request PD marker.
            # A PD prefill target still has to expose those captured layers to
            # the transfer path instead of relying on LogitsProcessor's request
            # capture mode to retain them.
            if self.model.dspark_layers_to_capture is not None:
                pd_aux_hidden_states = captured_aux_hidden_states
        hidden_states, pre_hc_head = hidden_states

        logits_metadata = forward_batch
        tail = None
        if (
            self.capture_aux_hidden_states
            and self.model.late_layer_start is not None
            and forward_batch.forward_mode.is_extend_without_speculative()
            # A mid-chunk prefill skipped the tail and returned full rows.
            and get_attn_backend().tail_forward_metadata is not None
        ):
            tail = get_attn_backend().tail_forward_metadata.late_layer_tail
            input_ids = (
                input_ids[tail.global_token_indices]
                if tail.global_token_indices is not None
                else tail.rows(input_ids)
            )
            if tail.real_rows_count is not None:
                # Fixed-capacity tail block from the prefill CUDA graph: drop padding.
                n = tail.real_rows_count
                input_ids = input_ids[:n]
                hidden_states = hidden_states[:n]
                pre_hc_head = pre_hc_head[:n]
                if aux_hidden_states is not None:
                    aux_hidden_states = [a[:n] for a in aux_hidden_states]
            logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
            logits_metadata.extend_seq_lens = tail.extend_seq_lens
            logits_metadata.extend_seq_lens_cpu = tail.extend_seq_lens_cpu
            logits_metadata.extend_logprob_start_lens_cpu = (
                # A prompt-logprob reader's tail is its whole extend, so its start
                # stands; every other start is the extend's end.
                [
                    min(start, t)
                    for start, t in zip(
                        forward_batch.extend_logprob_start_lens_cpu,
                        tail.extend_seq_lens_cpu,
                    )
                ]
                if forward_batch.return_logprob
                else tail.extend_seq_lens_cpu
            )

        output = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            logits_metadata,
            aux_hidden_states,
            hidden_states_before_norm=(
                None if aux_hidden_states is not None else pre_hc_head
            ),
        )
        if tail is not None:
            output.hidden_states_token_indices = (
                tail.global_token_indices
                if tail.global_token_indices is not None
                else (
                    tail.token_indices[: tail.real_rows_count]
                    if tail.real_rows_count is not None
                    else tail.token_indices
                )
            )
        if pd_aux_hidden_states is not None and (
            has_pd_hidden_capture or self.model.dspark_layers_to_capture is not None
        ):
            packed_pd_aux_hidden = pack_aux_hidden_states(pd_aux_hidden_states)
            output.hidden_states = (
                packed_pd_aux_hidden.flatten(1)
                if packed_pd_aux_hidden.ndim == 3
                else packed_pd_aux_hidden
            )
        return output

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        if _FP8_WO_A_UE8M0:
            from deep_gemm import transform_sf_into_required_layout

        if is_nextn:
            layers = [self.model.decoder]
        else:
            layers = [
                self.model.layers[layer_id]
                for layer_id in range(self.model.start_layer, self.model.end_layer)
            ]
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            if _wo_a_weight_scale_to_e8m0 is not None:
                # ROCm: aiter's mxscale GEMM reads uint8 e8m0 block scales, and
                # requantizes the weight when the checkpoint's scales are not
                # already powers of two. It also needs the weight row-major, so
                # check the linear method honoured keep_plain_weight_layout: a
                # preshuffled weight has the same shape, dtype and strides and
                # would only show up as garbage output.
                assert not getattr(attn.wo_a, "aiter_bpreshuffled", False), (
                    "DSV4 wo_a was B-preshuffled by the fp8 linear method; the "
                    "aiter mxscale absorb GEMM needs the row-major weight"
                )
                weight, scale = _wo_a_weight_scale_to_e8m0(
                    attn.wo_a.weight.data,
                    attn.wo_a.weight_scale_inv.data,
                    G,
                    R,
                )
                attn.wo_a.weight.data = weight.view(G * R, D)
                attn.wo_a.weight_scale_inv.data = scale
                attn.wo_a.weight_scale_inv.format_ue8m0 = True
                continue

            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            if _FP8_WO_A_UE8M0:
                attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                    raw_scale,
                    mn=R,
                    k=D,
                    recipe=(1, 128, 128),
                    num_groups=G,
                    is_sfa=False,
                )
                attn.wo_a.weight_scale_inv.format_ue8m0 = True
            else:
                attn.wo_a.weight_scale_inv.data = raw_scale.contiguous()
                attn.wo_a.weight_scale_inv.format_ue8m0 = False

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if self.wo_a_fp8:
            self._setup_fp8_wo_a_scales(is_nextn)

        for module in self.modules():
            if isinstance(module, MQALayer):
                module.setup_mega_attention()

        if is_nextn:
            return
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_id]
            self_attn = layer.self_attn
            if (
                self_attn.compress_ratio in (4, 128)
                and not self_attn.compressor.ape_converted
            ):
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()
            layer.refresh_mhc_norm_weight_cache()
        layers = self.model.layers
        for i, layer in enumerate(layers):
            if isinstance(layer, DeepseekV4PlatformDecoderLayer):
                layer.bind_next(layers[i + 1] if i + 1 < len(layers) else None)

    def precompile_kernels_after_loading(self) -> None:
        from sglang.srt.layers.moe.mega_moe import (
            build_mega_moe_shared_weights,
            should_fuse_mega_moe_shared_experts,
        )

        # The only deployments whose fused-collective gates can fire; registration
        # must happen eagerly here, before graph capture.
        if get_platform().is_blackwell and get_parallel().tp_size == 4:
            from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                register_fused_all_reduce_comm,
            )

            register_fused_all_reduce_comm()

        for module in self.modules():
            if isinstance(
                module, deepseek_v2.DeepseekV2MoE
            ) and should_fuse_mega_moe_shared_experts(module):
                module.mega_shared_l1_weights, module.mega_shared_l2_weights = (
                    build_mega_moe_shared_weights(module.shared_experts)
                )

    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str,
        is_nextn: bool = False,
        num_hidden_layers: Optional[int] = None,
    ) -> str:
        if name.startswith("vision."):
            return name.replace(".attn.wqkv.", ".attn.qkv_proj.").replace(
                ".attn.wo.", ".attn.proj."
            )
        if name.startswith(("aligner.", "image_")):
            return name
        if name.startswith("embed."):
            return "model.embed_tokens." + name.removeprefix("embed.")
        if name.startswith("head."):
            return "lm_head." + name.removeprefix("head.")
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj.weight_scale_inv"
                    elif rest == "h_proj.scale":
                        rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        if "self_attn" in name and name.endswith(".scale"):
            name = name.removesuffix(".scale") + ".weight_scale_inv"
        if ".engram.wkv." in name and name.endswith(".scale"):
            name = name.removesuffix(".scale") + ".weight_scale_inv"

        name = name.replace(".gate.tid2eid", ".topk.tid2eid")
        name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        if "mlp" in name and name.endswith(".scale"):
            name = name.removesuffix(".scale") + ".weight_scale_inv"

        return name

    def _prewarm_mhc_kernels(self) -> None:
        """One-shot MHC JIT prewarm at load time, synced across ranks.

        Runs before any forward so the compile burst stays off the serving
        path; the barrier keeps ranks from proceeding while a peer is still
        compiling. The early returns below must stay rank-uniform.
        """
        if _is_hcu:
            # HCU AITER TileLang MHC kernels are specialized lazily from real
            # request shapes. The generic bucket prewarm reuses that kernel
            # across synthetic n_splits shapes and can violate its cached input
            # contract. Preserve the validated HCU behavior.
            return
        if self._mhc_prewarmed_at_load:
            return
        self._mhc_prewarmed_at_load = True
        if _is_npu or _is_xpu or not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            return
        layer = next(
            (m for m in self.model.layers if isinstance(m, DeepseekV4DecoderLayer)),
            None,
        )
        if layer is None:
            return

        from sglang.kernels.ops.layernorm.mhc import mhc_post, prewarm_mhc_pre

        tic = time.perf_counter()
        residual = torch.zeros(
            (1, layer.hc_mult, layer.hidden_size),
            dtype=torch.bfloat16,
            device=layer.hc_attn_fn.device,
        )
        prewarm_mhc_pre(
            # Template carrying dtype/device; buckets allocate their own sizes.
            residual=residual,
            fn=layer.hc_attn_fn,
            hc_scale=layer.hc_attn_scale,
            hc_base=layer.hc_attn_base,
            rms_eps=layer.rms_norm_eps,
            hc_pre_eps=layer.hc_eps,
            hc_sinkhorn_eps=layer.hc_eps,
            hc_post_mult_value=MHC_POST_MULT_VALUE,
            sinkhorn_repeat=layer.hc_sinkhorn_iters,
            n_splits=1,
            n_splits_pre=32,
            norm_weight=layer.input_layernorm.weight.data,
            norm_eps=layer.input_layernorm.variance_epsilon,
        )
        mhc_post(
            x=residual.new_zeros((1, layer.hidden_size)),
            residual=residual,
            post_layer_mix=torch.zeros(
                (1, layer.hc_mult, 1),
                dtype=torch.float32,
                device=residual.device,
            ),
            comb_res_mix=torch.zeros(
                (1, layer.hc_mult, layer.hc_mult),
                dtype=torch.float32,
                device=residual.device,
            ),
        )
        torch.cuda.synchronize()
        compile_secs = time.perf_counter() - tic
        # Runs before init_memory_pool(); don't let transients skew pool sizing.
        torch.cuda.empty_cache()
        get_parallel().tp_group.barrier()
        logger.info(
            "DeepSeek V4 MHC prewarm at load: compile %.1fs, rank sync +%.1fs",
            compile_secs,
            time.perf_counter() - tic - compile_secs,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        # Must mirror MQALayer.__init__'s `quantize_wo_a`: dequantizing wo_a here
        # while the layer allocated an FP8 parameter (or vice versa) fails the
        # weight loader's dtype check.
        if not (self.wo_a_fp8 or use_npu_arch35_mxfp8_wo_a(self.quant_config)):
            weights = _prepare_deepseek_v4_weights(weights, self.quant_config)

        stacked_params_mapping = DEEPSEEK_V4_STACKED_PARAMS_MAPPING

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if is_wint4afp8_or_wint4a16_config(self.quant_config):
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"

        fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]] = {}
        skipped_by_group: dict[str, int] = {}
        # V4 checkpoints must load every compressor / indexer tensor.
        is_dsv41 = getattr(self.config, "model_type", None) == "deepseek_v41"

        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        def maybe_remap_compressed_tensors_scale_name(param_name: str) -> str:
            # Compressed-tensors checkpoints can expose *_scale_inv while the
            # materialized layer parameter is named *_scale. The generic
            # DeepSeek-V4 config alias does not expose this mapping itself.
            candidate_names = []
            if param_name.endswith(".weight_scale_inv"):
                candidate_names.append(
                    param_name.replace(".weight_scale_inv", ".weight_scale")
                )
            if param_name.endswith("_weight_scale_inv"):
                candidate_names.append(
                    param_name.replace("_weight_scale_inv", "_weight_scale")
                )
            if (
                _FP8_WO_A_GEMM
                and param_name.endswith(".scale")
                and not param_name.endswith(".weight_scale")
            ):
                base = param_name[: -len(".scale")]
                candidate_names.extend(
                    [base + ".weight_scale_inv", base + ".weight_scale"]
                )

            for candidate_name in candidate_names:
                if candidate_name in params_dict:
                    return candidate_name
            return param_name

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                if (
                    self.wo_a_fp8
                    and name.endswith(".wo_a.weight")
                    and loaded_weight.dtype != torch.float8_e4m3fn
                ):
                    raise ValueError(
                        f"SGLANG_OPT_FP8_WO_A_GEMM is enabled but {name} has "
                        f"dtype {loaded_weight.dtype}, expected "
                        "torch.float8_e4m3fn. This checkpoint does not provide "
                        "a supported fp8-quantized wo_a; rerun with "
                        "SGLANG_OPT_FP8_WO_A_GEMM=0."
                    )
                try:
                    use_async_loading = should_async_load(loaded_weight)
                    # EnGram tables are much larger than ordinary model weights.
                    # Loading them asynchronously keeps the full checkpoint tensors
                    # alive in pending futures and can exhaust host memory before
                    # their TP-local row shards are copied. Consume them inline so
                    # each full tensor can be released before reading the next one.
                    if (
                        _is_hcu
                        and envs.SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION.get()
                        and ".engram." in name
                    ):
                        use_async_loading = False

                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    # V4.1 checkpoint tensors with no module in the text model yet.
                    skip_group = None
                    if not is_dsv41:
                        pass
                    elif self.vision is None and name.startswith(
                        ("vision.", "aligner.", "image_")
                    ):
                        skip_group = "vision"
                    elif self.vision is None and name.endswith(
                        ".gate.e_score_correction_bias_vl"
                    ):
                        skip_group = "gate.bias_vl"
                    if skip_group is not None:
                        skipped_by_group[skip_group] = (
                            skipped_by_group.get(skip_group, 0) + 1
                        )
                        continue

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name not in params_dict:
                            name = maybe_remap_compressed_tensors_scale_name(name)
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        skip_unmaterialized_expert_param = False
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            resolved_name = name.replace(weight_name, param_name)
                            if resolved_name not in params_dict:
                                resolved_name = (
                                    maybe_remap_compressed_tensors_scale_name(
                                        resolved_name
                                    )
                                )
                            if resolved_name not in params_dict:
                                skip_unmaterialized_expert_param = True
                                continue
                            param = params_dict[resolved_name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    resolved_name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(resolved_name)
                            break
                        else:
                            if skip_unmaterialized_expert_param:
                                continue
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            if (
                                name == "model.norm.weight"
                                and not self.pp_group.is_last_rank
                            ):
                                continue
                            if (
                                name.startswith("model.hc_head_")
                                or name == "lm_head.weight"
                            ) and not self.pp_group.is_last_rank:
                                continue
                            elif (
                                COMPRESSOR_PART in name
                                and ".wkv_gate." not in name
                                and (name.rsplit(".", 2)[0] + ".wkv_gate.weight")
                                in params_dict
                            ):
                                # Split-projection modules load per parameter instead.
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        _clone_if_runai_streamed_tensor(loaded_weight),
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            elif (
                                fuse_wqa_wkv
                                and ".compressor." not in name
                                and ".engram." not in name
                                and (
                                    name.endswith(".wq_a.weight")
                                    or name.endswith(".wq_a.weight_scale_inv")
                                    or name.endswith(".wq_a.weight_scale")
                                    or name.endswith(".wkv.weight")
                                    or name.endswith(".wkv.weight_scale_inv")
                                    or name.endswith(".wkv.weight_scale")
                                    or name.endswith(".wq_a.qweight")
                                    or name.endswith(".wkv.qweight")
                                    or name.endswith(".wq_a.qweight_type")
                                    or name.endswith(".wkv.qweight_type")
                                )
                            ):
                                is_q = ".wq_a." in name
                                param_name = name.replace(
                                    ".wq_a." if is_q else ".wkv.", ".wqkv_a."
                                )
                                cache_key = param_name
                                bucket = cache_wqkv_a_weight.setdefault(cache_key, {})
                                shard_key = "q" if is_q else "kv"
                                assert shard_key not in bucket, (
                                    f"duplicate shard {shard_key} for {param_name}"
                                )
                                bucket[shard_key] = _clone_if_runai_streamed_tensor(
                                    loaded_weight
                                )
                                if len(bucket) == 2:
                                    fused_weight = _fuse_deepseek_v4_wqkv_a_pair(
                                        param_name, bucket
                                    )
                                    param_name = (
                                        maybe_remap_compressed_tensors_scale_name(
                                            param_name
                                        )
                                    )
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_wqkv_a_weight.pop(cache_key)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    name = maybe_remap_compressed_tensors_scale_name(
                                        name
                                    )
                                if name not in params_dict:
                                    if not name.startswith("mtp"):
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    context = f"name={name!r} loaded_weight.shape={loaded_weight.shape}"
                    if hasattr(e, "add_note"):
                        e.add_note(context)
                    raise

            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        assert len(cache_wqkv_a_weight) == 0, cache_wqkv_a_weight.keys()
        if skipped_by_group:
            log_info_on_rank0(
                logger,
                "Skipped checkpoint tensors not wired yet: "
                + ", ".join(f"{k}={v}" for k, v in sorted(skipped_by_group.items())),
            )
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = [
            "attn_mqa.k_scale",
            "attn_mqa.v_scale",
            "blockscale_swizzled",
        ]
        if not self.pp_group.is_first_rank:
            skipped_checking_patterns.append("embed_tokens")
        if not self.pp_group.is_last_rank:
            skipped_checking_patterns.append("model.norm.")
            skipped_checking_patterns.extend(["lm_head", "hc_head_"])
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if unloaded_params:
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

        if not is_nextn:
            for i, layer in enumerate(self.model.layers):
                if getattr(layer, "engram", None) is not None:
                    layer.engram.embed.finish_load(label=f"layer {i}")
            from sglang.kernels.ops.gemm.bf16_fp32 import prewarm_auto_bf16_fp32

            prewarm_auto_bf16_fp32()
            self._prewarm_mhc_kernels()

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        # Hot weight reload (RL workflows). Use the device-agnostic module
        # accessor so this works on both CUDA/HIP and NPU.
        torch.get_device_module().empty_cache()
        torch.get_device_module().synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )

DeepseekV4DecoderLayer = DeepseekV4PlatformDecoderLayer
