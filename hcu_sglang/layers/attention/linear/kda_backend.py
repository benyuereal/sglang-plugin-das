from __future__ import annotations

import types
import dis
import builtins
from typing import Optional, Tuple, Union
import sglang.srt.layers.attention.linear.kda_backend as K
import sglang.srt.layers.attention.hybrid_linear_attn_backend as H
import sglang.srt.layers.attention.mamba.mamba2_metadata as MD

globals().update({name: value for name, value in vars(K).items() if not name.startswith("__")})
globals().update({name: value for name, value in vars(H).items() if not name.startswith("__")})
from hcu_sglang.utils.detection import is_hcu
from sglang.srt.utils import get_bool_env_var
from hcu_sglang.layers.attention.linear.kda_state_indices import prepare_fused_replay_state_indices

_hcu_causal_conv1d_fn = None
_hcu_causal_conv1d_update = None
_use_hcu_cc1d = is_hcu() and get_bool_env_var(
    "SGLANG_KDA_USE_CAUSAL_CONV1D_HCU", default="true"
)
if _use_hcu_cc1d:
    from causal_conv1d import (
        causal_conv1d_fn_hcu as _hcu_causal_conv1d_fn,
        causal_conv1d_update as _hcu_causal_conv1d_update,
    )


def _run_causal_conv1d_fn(
    x,
    weight,
    bias,
    *,
    conv_states,
    query_start_loc,
    seq_lens_cpu,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
):
    """HCU uses the operator-library varlen causal conv; other platforms keep
    the sglang triton implementation."""
    if _hcu_causal_conv1d_fn is not None:
        return _hcu_causal_conv1d_fn(
            x,
            weight,
            bias,
            initial_states=conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            seq_lens_cpu=seq_lens_cpu,
            activation=activation,
        )
    return causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        seq_lens_cpu=seq_lens_cpu,
        activation=activation,
    )

def _run_causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    *,
    activation="silu",
    conv_state_indices=None,
):
    """HCU uses the operator-library single-step causal conv; other platforms
    keep the sglang triton implementation (or its NPU/CPU overrides)."""
    if _hcu_causal_conv1d_update is not None:
        return _hcu_causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
        )
    return causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
    )

class _DispatcherMethods:
    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
        verify_backend: LinearAttnKernelBackend,
    ):
        self.verify_backend = verify_backend
        self.prefill_backend = prefill_backend
        triton_kernel = TritonKDAKernel()
        self.triton_kernel = triton_kernel
        helion_kernel = None
        if (
            decode_backend.is_helion()
            or prefill_backend.is_helion()
            or verify_backend.is_helion()
        ):
            if not is_cuda():
                raise ValueError("KDA Helion backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_helion import (
                HelionKDAKernel,
            )

            helion_kernel = HelionKDAKernel(
                triton_fallback=triton_kernel,
                enable_decode=decode_backend.is_helion(),
                enable_prefill=prefill_backend.is_helion(),
            )

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_helion():
            assert helion_kernel is not None
            self.decode_kernel = helion_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            self.decode_kernel = CuteDSLKDAKernel()
        elif decode_backend.is_flashinfer():
            # FlashInfer recurrent_kda: SM100 decode + MTP (target_verify).
            if not is_cuda():
                raise ValueError("KDA FlashInfer backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
                FlashInferKDAKernel,
            )

            self.decode_kernel = FlashInferKDAKernel()
        else:
            raise ValueError(
                f"Unsupported KDA decode backend: {decode_backend}. "
                "KDA supports 'triton', 'helion', 'cutedsl', or 'flashinfer'."
            )

        # target_verify kernel, selected via --linear-attn-verify-backend (defaults
        # to follow decode: flashinfer -> recurrent_kda, else triton).
        #   triton: fused chain + tree (retrieve_parent_token) verify; the reference
        #     the KDA correctness tests assert against.
        #   flashinfer: recurrent_kda (SM100, chain only); reuses the decode kernel
        #     when decode is also flashinfer.
        #   helion: recurrent chain + tree verify with per-candidate snapshots.
        #   nv_cutedsl: fused Kimi-K3/DSpARK dense verify.
        if verify_backend.is_triton():
            self.verify_kernel = triton_kernel
        elif verify_backend.is_helion():
            assert helion_kernel is not None
            self.verify_kernel = helion_kernel
        elif verify_backend.is_flashinfer():
            if decode_backend.is_flashinfer():
                self.verify_kernel = self.decode_kernel
            else:
                if not is_cuda():
                    raise ValueError("KDA FlashInfer verify backend requires CUDA")
                from sglang.srt.layers.attention.linear.kernels.kda_flashinfer import (
                    FlashInferKDAKernel,
                )

                self.verify_kernel = FlashInferKDAKernel()
        elif verify_backend.is_nv_cutedsl():
            self.verify_kernel = triton_kernel
        elif verify_backend.is_custom():
            # Future custom KDA verify kernel plugs in here.
            raise NotImplementedError(
                "--linear-attn-verify-backend custom: no custom KDA verify kernel "
                "is registered yet."
            )
        else:
            raise ValueError(
                f"Unsupported KDA verify backend: {verify_backend}. "
                "KDA verify supports 'triton', 'helion', 'nv_cutedsl', or 'flashinfer'."
            )

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_helion():
            assert helion_kernel is not None
            self.extend_kernel = helion_kernel
        elif prefill_backend.is_flashkda():
            from sglang.srt.layers.attention.linear.kernels.kda_flashkda import (
                FlashKDAKernel,
            )

            self.extend_kernel = FlashKDAKernel()
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("KDA FlashInfer prefill backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (
                FlashInferKDAPrefillKernel,
            )

            self.extend_kernel = FlashInferKDAPrefillKernel()
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("KDA CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import (
                CuteDSLKDAKernel,
            )

            cutedsl_kernel = CuteDSLKDAKernel()
            if cutedsl_kernel.supports_prefill:
                # SM100 chunk prefill pipeline.
                self.extend_kernel = cutedsl_kernel
            else:
                # CuTe DSL prefill kernels need SM100 (Blackwell); on older GPUs
                # fall back to the Triton chunk kernel.
                self.extend_kernel = triton_kernel
                rank0_log(
                    "KDA cutedsl prefill needs SM100; falling back to Triton extend."
                )
        elif prefill_backend.is_ptx_kda():
            if not is_cuda():
                raise ValueError("PTX KDA prefill backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_ptx import PtxKDAKernel

            ptx_kda_kernel = PtxKDAKernel()
            if ptx_kda_kernel.supports_prefill:
                self.extend_kernel = ptx_kda_kernel
            else:
                self.extend_kernel = triton_kernel
                rank0_log(
                    "PTX KDA prefill needs SM100 or SM103; falling back to Triton "
                    "extend."
                )
        elif prefill_backend.is_nvidia_kda():
            if not is_cuda():
                raise ValueError("NVIDIA KDA prefill backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.kda_nvidia import (
                NvidiaKDAKernel,
            )

            nvidia_kda_kernel = NvidiaKDAKernel()
            if nvidia_kda_kernel.supports_prefill:
                self.extend_kernel = nvidia_kda_kernel
            else:
                self.extend_kernel = triton_kernel
                rank0_log(
                    "NVIDIA KDA prefill needs SM100; falling back to Triton extend."
                )
        else:
            raise ValueError(
                f"Unsupported KDA prefill backend: {prefill_backend}. "
                "KDA supports 'triton', 'helion', 'flashinfer', 'flashkda', 'cutedsl', "
                "'nvidia_kda', or 'ptx_kda' (cutedsl/nvidia_kda prefill need "
                "SM100, ptx_kda SM100 or SM103)."
            )

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )
        self.supports_packed_decode_safe_gate = getattr(
            self.decode_kernel, "supports_packed_decode_safe_gate", False
        )

        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__}, "
            f"verify_impl={getattr(self.verify_kernel, 'verify_impl_name', 'native')} "
            f"extend={self.extend_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

class _KdaMethods:
    def __init__(self, model_runner: ModelRunner):
        super(KDAAttnBackend, self).__init__(model_runner)
        self.prefill_graph_max_seqs = envs.SGLANG_KDA_PREFILL_GRAPH_MAX_SEQS.get()
        # Needed by the extra_buffer track path: _init_track_conv_indices reads
        # conv_states_shape[-1] as the conv window length (kernel_size - 1).
        # The KDA pool stores conv states as [kernel-1, dim] — transposed vs
        # Mamba2/GDN's [dim, kernel-1] — so expose the transposed shape here.
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0]
            .transpose(-1, -2)
            .shape
        )
        self.use_hcu_kda = self.req_to_token_pool.mamba_pool.use_hcu_kda
        self.hcu_kernel = None
        if self.use_hcu_kda:
            from hcu_sglang.layers.attention.linear.kda_hcu import HcuKDAKernel

            self.hcu_kernel = HcuKDAKernel()
            self.supports_ragged_verify_graph = False

        backends = model_runner.linear_attn_backends
        decode_backend = backends.decode
        prefill_backend = backends.prefill
        verify_backend = backends.verify
        # KDA FlashInfer target_verify (recurrent_kda) is chain-only (no tree-ancestor
        # traversal). Reject EAGLE tree verify (topk > 1) early at setup, keyed on the
        # verify backend (not decode). The kernel keeps a per-call
        # retrieve_parent_token backstop that also covers ngram tree.
        speculative_topk = get_spec().speculative_eagle_topk or 1
        if verify_backend.is_flashinfer() and speculative_topk > 1:
            raise ValueError(
                "KDA FlashInfer speculative decoding only supports topk=1 "
                "(EAGLE tree verify / retrieve_parent_token is unsupported)."
            )
        if self.use_hcu_kda:
            if not all(
                backend.is_triton()
                for backend in (decode_backend, prefill_backend, verify_backend)
            ):
                raise ValueError(
                    "HCU KDA uses the source Triton decode, prefill and verify kernels"
                )
            self.kernel_dispatcher = None
        else:
            self.kernel_dispatcher = KDAKernelDispatcher(
                decode_backend, prefill_backend, verify_backend
            )
        if prefill_backend.is_flashinfer():
            _validate_flashinfer_kda_prefill(
                chunk_size=self.mamba_chunk_size,
                enable_two_batch_overlap=get_exec().overlap.enable_two_batch_overlap,
                prefill_cuda_graph_backend=get_exec().graph.cuda_graph_config.prefill.backend,
            )
        # One-shot; emitted at the first fused-decode interception below.
        self._fused_override_notice = (
            "K3 fused KDA decode engaged: --linear-attn-decode-backend "
            f"{decode_backend} only picks the fallback kernel for shapes "
            "the fused kernel does not cover."
        )
        self._fused_chain_verify_fn = None
        if (
            not self.use_hcu_kda
            and envs.SGLANG_OPT_FUSED_KDA_VERIFY.get()
            and verify_backend.is_triton()
            and self.kernel_dispatcher.verify_kernel.supports_fused_chain_verify
        ):
            from sglang.kernels.ops.attention.fla.fused_kda_conv_recurrent_verify import (
                fused_kda_conv_gating_verify,
            )

            self._fused_chain_verify_fn = fused_kda_conv_gating_verify
            rank0_log("KDA fused chain-verify kernel enabled (topk==1 path).")
        # Per-request row index into the speculative `intermediate_ssm` scratch,
        # used by the MTP / target_verify path (mirrors GDNAttnBackend). Sized
        # past the pool for attn_tp-padded warmup/MLP-sync batches (see helper).
        self.verify_intermediate_state_indices = (
            build_verify_intermediate_state_indices(
                self.req_to_token_pool.size,
                model_runner.device,
            )
        )
        # Fused-accept spec path (flashinfer recurrent_kda): the next verify
        # seeds itself in-kernel from the accepted checkpoint slot
        # (num_accepted_tokens), so the per-round SSM commit scatter is skipped.
        # accept_lens_pool holds last round's accept length per mamba slot;
        # extend stages fresh requests with 1 (read slot 0). Its presence is the
        # signal that switches the post-verify commit to conv-only.
        if not self.use_hcu_kda and self._can_fuse_accept_state(verify_backend):
            self.accept_lens_pool = torch.ones(
                self.req_to_token_pool.size + 1,
                dtype=torch.int32,
                device=model_runner.device,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # An eager forward never runs on a graph bucket's tables.
        self.prefill_graph_metadata = None
        super(KDAAttnBackend, self).init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            if self.forward_metadata.mamba_track_mask_indices is None:
                self.forward_metadata.mamba_track_mask_indices = (
                    forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                )
            self.forward_metadata.conv_states_mask_indices = (
                self._translate_mamba_indices(forward_batch.mamba_track_indices)[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )
            if (
                self.kernel_dispatcher.extend_kernel.uses_state_checkpoints
                and forward_batch.forward_mode.is_extend_without_speculative()
                and self.forward_metadata.track_ssm_h_src is not None
                and self.forward_metadata.track_ssm_h_src.numel() > 0
            ):
                from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (
                    build_flashinfer_kda_checkpoint_plan,
                )

                assert self._has_cpu_prefill_track_metadata(forward_batch), (
                    "FlashInfer KDA checkpoints require host prefill tracking metadata"
                )
                build_flashinfer_kda_checkpoint_plan(
                    forward_batch,
                    self.forward_metadata,
                    self.device,
                    self.mamba_chunk_size,
                )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        if self.use_hcu_kda:
            return self._hcu_forward_decode(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        # ReplaySSM is mostly a GDN bandwidth optimization. It remains wired for
        # KDA correctness paths, but packed decode is faster for KDA today.
        replayssm_write_pos = getattr(
            self.forward_metadata, "replayssm_write_pos", None
        )
        replayssm_force_flush = getattr(
            self.forward_metadata, "replayssm_force_flush", None
        )
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        deferred_f_b = bool(getattr(layer, "_k3_deferred_f_b", False))
        if replayssm_d is None and deferred_f_b and is_gfx95_supported():
            fused_static = getattr(layer, "_k3_hip_fused_decode_args", None)
            fused_backend = getattr(layer, "_k3_hip_fused_decode_backend", "")
            onorm_gate = getattr(layer, "_k3_onorm_gate", None)
            if fused_static is not None and onorm_gate is not None:
                f_b_weight, norm_weight, norm_eps, a_log = fused_static
                conv_state_view = conv_states.transpose(-1, -2)
                output_gate = onorm_gate.view(
                    onorm_gate.shape[0], layer.num_v_heads, layer.head_v_dim
                )
                out = mixed_qkv.new_empty(
                    (1, mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim)
                )
                if fused_backend == "aiter" and kda_fused_decode_aiter_hip.covered(
                    a,
                    f_b_weight,
                    mixed_qkv,
                    b,
                    conv_state_view,
                    ssm_states,
                    cache_indices,
                    output_gate,
                    norm_weight,
                ):
                    core_attn_out = kda_fused_decode_aiter_hip.run(
                        f_a=a,
                        f_b_weight=f_b_weight,
                        mixed_qkv=mixed_qkv,
                        conv_weight=layer.conv_weights,
                        conv_state=conv_state_view,
                        raw_beta=b,
                        A_log=a_log,
                        dt_bias=layer.dt_bias,
                        lower_bound=float(layer.lower_bound),
                        state=ssm_states,
                        state_indices=cache_indices,
                        output_gate=output_gate,
                        norm_weight=norm_weight,
                        norm_eps=norm_eps,
                        out=out,
                    )
                else:
                    core_attn_out = None
                    if not getattr(KDAAttnBackend, "_hip_fused_reject_logged", False):
                        KDAAttnBackend._hip_fused_reject_logged = True
                        rank0_log(
                            "K3 HIP fused KDA rejected: "
                            f"backend={fused_backend}, "
                            f"f_a={tuple(a.shape)}/{a.dtype}/{a.stride()}, "
                            f"mixed={tuple(mixed_qkv.shape)}/{mixed_qkv.dtype}/"
                            f"{mixed_qkv.stride()}, beta={tuple(b.shape)}/{b.dtype}/"
                            f"{b.stride()}, conv={tuple(conv_state_view.shape)}/"
                            f"{conv_state_view.dtype}/{conv_state_view.stride()}, "
                            f"state={tuple(ssm_states.shape)}/{ssm_states.dtype}/"
                            f"{ssm_states.stride()}, indices={cache_indices.dtype}/"
                            f"{cache_indices.stride()}, gate={tuple(output_gate.shape)}/"
                            f"{output_gate.dtype}/{output_gate.stride()}, "
                            f"norm={norm_weight.dtype}/{norm_weight.stride()}"
                        )

                if core_attn_out is not None:
                    layer._k3_onorm_consumed = True
                    self._track_mamba_state_decode(
                        forward_batch,
                        conv_states,
                        ssm_states,
                        cache_indices,
                        layer.layer_id,
                    )
                    return core_attn_out

            # The model deferred f_b only after publishing static fallback
            # weights. Materialize the original gate before entering the
            # unchanged conv + packed-KDA fallback chain.
            from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

            if fused_static is None:
                raise RuntimeError("K3 deferred f_b is missing fallback weights")
            a = kimi_k3_tiny_gemm(
                a,
                fused_static[0].view(
                    layer.num_v_heads * layer.head_v_dim, layer.head_v_dim
                ),
            )

        # Fully fused decode step: conv1d update + delta-rule recurrence +
        # gated RMSNorm in one kernel. Engages only when the model handed off
        # the output-norm gate for this forward (attempt-and-verify stash,
        # see kimi_k3.py) and the shapes are covered; the model applies the
        # norm itself whenever the stash is left unconsumed.
        if replayssm_d is None:
            fused_static = getattr(layer, "_k3_fused_decode_args", None)
            onorm_gate = getattr(layer, "_k3_onorm_gate", None)
            if (
                fused_static is not None
                and onorm_gate is not None
                and mixed_qkv.shape[0] == cache_indices.shape[0]
                and b.ndim == 3
                and kda_fused_decode.covered(
                    mixed_qkv,
                    a,
                    b[0],
                    conv_states,
                    ssm_states,
                    cache_indices,
                    onorm_gate,
                )
            ):
                if self._fused_override_notice is not None:
                    rank0_log(self._fused_override_notice)
                    self._fused_override_notice = None
                w_q_t, w_k_t, w_v_t, conv_bias, a_log, onorm_w, onorm_eps = fused_static
                core_attn_out = kda_fused_decode.kda_fused_decode(
                    mixed_qkv,
                    a,
                    b[0],
                    conv_states,
                    w_q_t,
                    w_k_t,
                    w_v_t,
                    conv_bias,
                    a_log,
                    layer.dt_bias,
                    onorm_gate,
                    onorm_w,
                    ssm_states,
                    cache_indices,
                    scale=layer.head_k_dim**-0.5,
                    onorm_eps=onorm_eps,
                    lower_bound=layer.lower_bound,
                )
                layer._k3_onorm_consumed = True
                self._track_mamba_state_decode(
                    forward_batch,
                    conv_states,
                    ssm_states,
                    cache_indices,
                    layer.layer_id,
                )
                return core_attn_out
            elif fused_static is not None and onorm_gate is not None:
                # One-shot diagnostics: the model offered the handoff but the
                # runtime shapes were rejected (decode stays on the unfused
                # chain, which is correct but slower).
                if not getattr(KDAAttnBackend, "_fused_reject_logged", False):
                    KDAAttnBackend._fused_reject_logged = True
                    rank0_log(
                        "KDA fused decode rejected by covered(): "
                        f"mixed_qkv {tuple(mixed_qkv.shape)}/{mixed_qkv.dtype} "
                        f"stride {mixed_qkv.stride()}, "
                        f"conv_states {tuple(conv_states.shape)} "
                        f"stride {conv_states.stride()}, "
                        f"ssm_states {tuple(ssm_states.shape)}/{ssm_states.dtype}, "
                        f"b {tuple(b.shape)}, indices {cache_indices.dtype}"
                    )

        qkv = _run_causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        # The packed kernel assumes one token per request.
        if self.kernel_dispatcher.supports_packed_decode and (
            getattr(layer, "lower_bound", None) is None
            or self.kernel_dispatcher.supports_packed_decode_safe_gate
        ):
            assert qkv.shape[0] == cache_indices.shape[0], (
                "KDA packed decode requires one token per sequence (T=1): "
                f"got {qkv.shape[0]} tokens for {cache_indices.shape[0]} requests."
            )
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                lower_bound=layer.lower_bound,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
            )
            return core_attn_out

        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        core_attn_out = self.kernel_dispatcher.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            lower_bound=layer.lower_bound,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        # MTP / speculative-decode verify is a multi-token-per-seq path with
        # per-step state checkpointing + central rollback; handled separately.
        if self.use_hcu_kda:
            return self._hcu_forward_extend(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )

        if forward_batch.forward_mode.is_target_verify():
            return self._forward_target_verify(layer, forward_batch, mixed_qkv, a, b)

        if self.prefill_graph_metadata is not None:
            # Breakable prefill CUDA graph bucket: host-free chain on static
            # tables refreshed before each replay.
            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
            return kda_extend_in_graph(
                layer=layer,
                meta=self.prefill_graph_metadata,
                conv_pool=layer_cache.conv[0],
                ssm_states=layer_cache.temporal,
                kernel_dispatcher=self.kernel_dispatcher,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
            )

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0].transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        # Normal extend path
        if forward_batch.extend_prefix_lens is None:
            raise RuntimeError(
                "extend_prefix_lens cannot be None in non-TARGET_VERIFY mode."
            )

        physical_num_tokens = mixed_qkv.shape[0]
        logical_num_tokens = self.forward_metadata.logical_num_tokens
        if logical_num_tokens is None:
            logical_num_tokens = int(query_start_loc[-1])
        if logical_num_tokens < physical_num_tokens:
            mixed_qkv = mixed_qkv[:logical_num_tokens]
            a = a[:, :logical_num_tokens]
            b = b[:, :logical_num_tokens]

        if self.forward_metadata.has_mamba_track_mask:
            # Snapshot the conv sliding window at the last track-aligned chunk
            # boundary into the ping-pong track slots (the prefix-cache restore
            # source). The KDA pool stores conv states as [kernel-1, dim], so
            # rows of the raw [tokens, dim] pre-conv input index in directly
            # (GDN needs a transpose here; KDA does not).
            mamba_cache_params.conv[0][
                self.forward_metadata.conv_states_mask_indices
            ] = mixed_qkv[self.forward_metadata.track_conv_indices]

        has_initial_state = forward_batch.extend_prefix_lens > 0
        if _hcu_causal_conv1d_fn is not None:
            # The HCU library accepts each Q/K/V group separately; preserve the
            # main packed path on other devices.
            splits = [layer.q_dim, layer.k_dim, layer.v_dim]
            q, k, v = mixed_qkv.transpose(0, 1).split(splits, dim=0)
            q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
                splits, dim=0
            )
            q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)
            if layer.bias is not None:
                q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None
            q = _run_causal_conv1d_fn(
                q,
                q_conv_weight,
                q_bias,
                activation="silu",
                conv_states=q_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)
            k = _run_causal_conv1d_fn(
                k,
                k_conv_weight,
                k_bias,
                activation="silu",
                conv_states=k_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)
            v = _run_causal_conv1d_fn(
                v,
                v_conv_weight,
                v_bias,
                activation="silu",
                conv_states=v_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)
        else:
            qkv = self._convolve_prefill(
                layer, forward_batch, mixed_qkv, conv_states, cache_indices
            )
            q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)

        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        if a.ndim == 3:
            a = a.unflatten(-1, (-1, layer.head_k_dim))

        track_ssm = self.forward_metadata.has_mamba_track_mask
        track_chunk_idx = self.forward_metadata.track_chunk_idx
        h_track_buf = None
        if (
            track_ssm
            and track_chunk_idx is not None
            # Same rows as track_ssm_h_batch_src, but known without a GPU sync.
            and self.forward_metadata.track_ssm_h_src.numel() > 0
        ):
            # fp32 scratch the kernel snapshots the tracked chunk-boundary
            # states into (rows follow the batch; untracked rows stay unread).
            # A kernel that does not declare support would leave the buffer
            # unwritten and corrupt prefix-cache restores — fail loudly here.
            # Check the kernel the dispatcher will actually run (safe-gate
            # reroute included), not just the configured one.
            extend_kernel = self.kernel_dispatcher.effective_extend_kernel(
                layer.lower_bound, q.shape[1]
            )
            assert extend_kernel.supports_track_state_snapshot, (
                f"{type(extend_kernel).__name__} cannot write the fp32 track "
                f"snapshot required by the mamba track path; use "
                f"--linear-attn-prefill-backend triton or "
                f"--mamba-radix-cache-strategy no_buffer"
            )
            h_track_buf = torch.empty(
                (track_chunk_idx.shape[0], *ssm_states.shape[1:]),
                dtype=torch.float32,
                device=ssm_states.device,
            )
        core_attn_out = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            lower_bound=layer.lower_bound,
            beta_is_raw=True,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_prefix_lens=forward_batch.extend_prefix_lens,
            layer_id=layer.layer_id,
            # draft_extend_v2 must stay rollback-able, so kernels that commit state
            # in place (e.g. FlashKDA) must not run for it.
            is_spec_decode=forward_batch.forward_mode.is_draft_extend_v2(),
            return_intermediate_states=track_ssm,
            # Which global chunk rows of h the track snapshot will read; lets
            # kernels that cannot materialize per-chunk states (NVIDIA KDA) take the
            # fast path when the snapshot only needs the final state.
            track_ssm_h_src=(
                self.forward_metadata.track_ssm_h_src if track_ssm else None
            ),
            track_state=h_track_buf,
            track_chunk_idx=(track_chunk_idx if h_track_buf is not None else None),
            state_checkpoint_cu_starts=self.forward_metadata.state_checkpoint_cu_starts,
            num_state_checkpoints=self.forward_metadata.num_state_checkpoints,
            state_checkpoint_every_n_tokens=(
                self.forward_metadata.state_checkpoint_every_n_tokens
            ),
            state_checkpoint_indices=self.forward_metadata.state_checkpoint_indices,
            track_ssm_h_batch_src=self.forward_metadata.track_ssm_h_batch_src,
        )
        if track_ssm:
            # Snapshot the SSM state at the last track-aligned chunk boundary
            # from the kernel's per-chunk states (h) / final states into the
            # ping-pong track slots (see _init_track_ssm_indices).
            core_attn_out, h = core_attn_out
            self._track_mamba_state_extend(
                forward_batch,
                h,
                ssm_states,
                self.forward_metadata,
                h_track_buf=h_track_buf,
            )

        if logical_num_tokens < physical_num_tokens:
            pad = core_attn_out.new_zeros(
                (1, physical_num_tokens - logical_num_tokens)
                + tuple(core_attn_out.shape[2:])
            )
            core_attn_out = torch.cat((core_attn_out, pad), dim=1)

        if (
            self.accept_lens_pool is not None
            and not forward_batch.forward_mode.is_draft_extend_v2()
        ):
            # Fused-accept staging: the extend kernel just wrote this request's
            # committed state; copy it into scratch slot 0 and reset the accept
            # length to 1 so the first verify reads slot 0. Runs once per KDA
            # layer (the nat write is idempotent; the scratch copy is per-layer).
            slots = cache_indices.to(torch.int64)
            intermediate_ssm = mamba_cache_params.intermediate_ssm
            intermediate_ssm[slots, 0] = ssm_states[slots].to(intermediate_ssm.dtype)
            self.accept_lens_pool[slots] = 1

        return core_attn_out

    def _hcu_forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(-1, -2),
            layer.conv_weights,
            layer.bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        output = self.hcu_kernel.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            beta_scale=getattr(layer, "beta_scale", 1.0),
            lower_bound=(
                getattr(layer, "lower_bound", None)
                if getattr(layer, "safe_gate", True)
                else None
            ),
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )
        return output

    def _hcu_forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        # Under input_scattered or dp-attention, the model passes [N_pad, ...]
        # tensors but query_start_loc still describes only [0, n_valid). The
        # Triton kernels (causal_conv1d / chunk_kda) only schedule thread blocks
        # for rows covered by query_start_loc, so output rows [n_valid, N_pad)
        # are never written and keep whatever bit pattern torch.empty() left in
        # GPU memory — often NaN/Inf. Strip to valid rows here, run the kernels
        # on a smaller tensor, then re-pad core_attn_out with zeros so the
        # caller sees the [N_pad, ...] shape it expects.
        n_total = mixed_qkv.shape[0]
        n_valid = (
            (
                sum(forward_batch.extend_seq_lens_cpu)
                if forward_batch.extend_seq_lens_cpu is not None
                else forward_batch.extend_num_tokens
            )
            if not forward_batch.forward_mode.is_target_verify()
            else None
        )
        needs_repad = n_valid is not None and n_valid < n_total
        if needs_repad:
            mixed_qkv = mixed_qkv[:n_valid]

        if n_valid is not None:
            # Main's linear-attention wrapper may already trim mixed_qkv while
            # leaving the [1, N, ...] gates padded. Align their token dimension
            # even when mixed_qkv no longer needs trimming here.
            num_tokens = mixed_qkv.shape[0]
            a = a[:, :num_tokens] if a.ndim in (3, 4) else a[:num_tokens]
            b = b[:, :num_tokens] if b.ndim == 3 else b[:num_tokens]

        forward_metadata = self.forward_metadata
        # Match hybrid_linear_attn_backend._forward_metadata: read the CURRENT
        # forward_mode. DP max-length padding rewrites TARGET_VERIFY into
        # EXTEND and reshapes the batch into single-token rows, and the
        # metadata layer already prepared the EXTEND-shape query_start_loc
        # ([0,1,2,...,bs]) for it. Taking the target_verify branch on that
        # padded batch would divide seq_len by draft_token_num and reshape
        # the flat token stream into a tree layout that no longer matches
        # what the caller passes in.
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0].transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        splits = [layer.q_dim, layer.k_dim, layer.v_dim]

        if is_target_verify:
            draft_token_num = forward_batch.spec_info.draft_token_num
            seq_len = mixed_qkv.shape[0]
            batch_size = seq_len // draft_token_num
            intermediate_state_indices = self.verify_intermediate_state_indices[
                :batch_size
            ]
            mixed_qkv = mixed_qkv.view(batch_size, draft_token_num, -1).transpose(1, 2)
            mixed_qkv = (
                causal_conv1d_update(
                    mixed_qkv,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    activation="silu",
                    conv_state_indices=cache_indices[:batch_size],
                    intermediate_conv_window=mamba_cache_params.intermediate_conv_window[
                        0
                    ].transpose(-1, -2),
                    intermediate_state_indices=intermediate_state_indices,
                    retrieve_next_token=retrieve_next_token,
                    retrieve_next_sibling=retrieve_next_sibling,
                    retrieve_parent_token=retrieve_parent_token,
                )
                .transpose(1, 2)
                .reshape(seq_len, -1)
            )
            q, k, v = mixed_qkv.split(splits, dim=-1)
        else:
            has_initial_state = forward_batch.extend_prefix_lens > 0
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            q, k, v = mixed_qkv.split(splits, dim=0)
            q_conv_weight, k_conv_weight, v_conv_weight = layer.conv_weights.split(
                splits, dim=0
            )
            q_conv_state, k_conv_state, v_conv_state = conv_states.split(splits, dim=-2)
            if layer.bias is not None:
                q_bias, k_bias, v_bias = layer.bias.split(splits, dim=0)
            else:
                q_bias, k_bias, v_bias = None, None, None

            q = causal_conv1d_fn(
                q,
                q_conv_weight,
                q_bias,
                activation="silu",
                conv_states=q_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k,
                k_conv_weight,
                k_bias,
                activation="silu",
                conv_states=k_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v,
                v_conv_weight,
                v_bias,
                activation="silu",
                conv_states=v_conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)

        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)  # n (h d) -> 1 n h d
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)  # n (h d) -> 1 n h d

        if is_target_verify:
            mamba_pool = self.req_to_token_pool.mamba_pool
            replayssm_on = mamba_pool.use_hcu_kda_replayssm
            if not replayssm_on:
                assert mamba_cache_params.intermediate_ssm is not None
            replayssm_kwargs = {}
            if replayssm_on:
                replayssm_kwargs = dict(
                    cache_replayssm_inputs=True,
                    replayssm_rawv=mamba_cache_params.replayssm_rawv,
                    replayssm_rawk=mamba_cache_params.replayssm_rawk,
                    replayssm_g=mamba_cache_params.replayssm_g,
                    replayssm_beta=mamba_cache_params.replayssm_beta,
                )
            core_attn_out = self.hcu_kernel.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=mamba_cache_params.intermediate_ssm,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
                beta_scale=getattr(layer, "beta_scale", 1.0),
                lower_bound=(
                    getattr(layer, "lower_bound", None)
                    if getattr(layer, "safe_gate", True)
                    else None
                ),
                **replayssm_kwargs,
            )
        else:
            # Main models may supply raw flattened gate/beta; GLM already supplies
            # [1, T, H, K] gates and activated beta on extend.
            if a.ndim != 4:
                a = a.reshape(1, q.shape[1], -1, layer.head_k_dim)
                b = b.reshape(1, q.shape[1], -1).float().sigmoid() * getattr(
                    layer, "beta_scale", 1.0
                )
            core_attn_out = self.hcu_kernel.extend(
                q=q,
                k=k,
                v=v,
                g=a,
                beta=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                beta_scale=getattr(layer, "beta_scale", 1.0),
                lower_bound=(
                    getattr(layer, "lower_bound", None)
                    if getattr(layer, "safe_gate", True)
                    else None
                ),
                return_intermediate_state=forward_metadata.has_mamba_track_mask,
            )

            if forward_metadata.has_mamba_track_mask:
                core_attn_out, h = core_attn_out
                if h is not None:
                    self._track_mamba_state_extend(
                        forward_batch, h, ssm_states, forward_metadata
                    )

        if needs_repad:
            # core_attn_out comes back from the kernel as [1, n_valid, h, d]
            # (token dim is dim 1 due to the unsqueeze(0) above). Re-pad along
            # the token dim so the caller's `.squeeze(0).flatten(-2)` gives
            # [n_total, h*d] as expected.
            full = core_attn_out.new_zeros(
                (
                    core_attn_out.shape[0],
                    n_total,
                    *core_attn_out.shape[2:],
                )
            )
            full[:, :n_valid] = core_attn_out
            core_attn_out = full

        return core_attn_out

class _BaseMethods:
    def __init__(self, model_runner: ModelRunner):
        self.validate_mis_support(model_runner.server_args)
        super(MambaAttnBackendBase, self).__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.topk = get_spec().speculative_eagle_topk or 0
        self.is_draft_worker = model_runner.is_draft_worker
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.enable_unified_memory = get_memory().enable_unified_memory
        # model_config must not be touched here: backend selection reads the
        # linear_attn_backends stamp first, and that guard test constructs
        # backends on runners without a real model_config.
        self._model_runner = model_runner
        self._mamba_chunk_size: Optional[int] = None
        pool = self.req_to_token_pool
        self._fused_state_indices_ok = (
            torch.device(self.device).type == "cuda"
            and isinstance(pool, HybridReqToTokenPool)
            and pool.mamba_translate_is_fusable
        )
        # Keep the original Triton fusion unless HIP C++ is explicitly enabled.
        self._fused_replay_state_indices = fused_replay_state_indices
        if (
            self._fused_state_indices_ok
            and pool.mamba_v2p_table is None
            and get_bool_env_var(
                "SGLANG_USE_FUSED_REPLAY_STATE_INDICES", default="false"
            )
        ):
            # Prepare before graph capture/replay; never build on a live step.
            self._fused_replay_state_indices = prepare_fused_replay_state_indices(
                self.device
            )
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        # Static (max_bs,) track-dest buffer captured by pointer, refreshed in-place
        # each replay; the captured track-save reads this, not the InputBuffer slot.
        self.mamba_track_indices_buf = None
        # Per-bs static write-cursor / force-flush buffers for cuda-graph; None
        # unless --enable-linear-replayssm is set.
        self.replayssm_write_pos_list = None
        self.replayssm_force_flush_list = None
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None
        # Constant (== 1) for mamba-like backends; hoisted so the replay path
        # skips the per-cycle method dispatch.
        self._graph_seq_len_fill_value = self.get_cuda_graph_seq_len_fill_value()

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        logical_mode = forward_batch.logical_forward_mode

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_chunk_idx = None
        track_ssm_h_dst = None
        track_ssm_h_batch_src = None
        track_ssm_final_src = None
        track_ssm_final_dst = None
        track_ssm_seq_idx = None
        track_ssm_end_locs = None
        track_ssm_recompute_dst = None
        logical_num_tokens = None
        track_mask_indices = None

        # Padded rows carry req-pool row 0, which maps to the reserved mamba
        # padding slot 0 in both the static and the unified pool.
        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )
        mamba_cache_indices = self._translate_mamba_indices(mamba_cache_indices)
        if forward_batch.mamba_track_indices is not None:
            forward_batch.mamba_track_indices = self._translate_mamba_indices(
                forward_batch.mamba_track_indices
            )
        # Resolve the tracked-row selection once per forward
        cpu_track_metadata = self._has_cpu_prefill_track_metadata(forward_batch)
        if cpu_track_metadata:
            rows = [
                i
                for i, track in enumerate(forward_batch.mamba_prefill_track_mask_cpu)
                if track
            ]
            has_mamba_track_mask = bool(rows)
            track_mask_indices = self._track_indices_to_device(rows) if rows else None
        else:
            has_mamba_track_mask = bool(
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            )
        replayssm_write_pos = None
        replayssm_force_flush = None
        if logical_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
            # The ring cursor is a per-slot decode counter shared by all GDN layers;
            # manage it once here (snapshot, hand to layers, advance mod L), not per-layer.
            # Gate on the linear_replayssm FLAG, not on cursor-tensor presence: the
            # spec-verify ring (--enable-linear-replayssm-spec) shares the write_pos
            # allocation but owns it exclusively via commit_gdn_replayssm_spec
            # (advance-by-accept-count once per verify step). Advancing it here as
            # well inserts one phantom/stale ring entry per step and cumulatively
            # poisons the reconstruction (degenerate repetition at 10k+ tokens).
            mamba_pool = getattr(self.req_to_token_pool, "mamba_pool", None)
            write_pos_buf = (
                mamba_pool.replayssm_write_pos
                if mamba_pool is not None and mamba_pool.enable_linear_replayssm
                else None
            )
            if write_pos_buf is not None:
                slots = mamba_cache_indices.to(torch.long)
                # Padded rows point at the reserved padding slot 0; the clamp keeps
                # a -1 sentinel in bounds (the kernel zeroes rows with state_idx < 0).
                safe_slots = slots.clamp(min=0)
                replayssm_write_pos = write_pos_buf[safe_slots].clone()
                L = mamba_pool.linear_replayssm_cache_len
                # KDA has no radix coordination: flush only on the natural write_pos
                # == L-1 wrap. GDN adds the radix-aligned force-flush below.
                is_kda = getattr(mamba_pool, "replayssm_is_kda", False)
                # Force-flush on the radix track's seq_lens % mamba_track_interval
                # == 0 boundary so the ring folds into temporal[slot] when read.
                if not is_kda:
                    force_flush_bool = self._replayssm_track_flush_mask(
                        forward_batch.seq_lens_cpu, bs
                    )
                    replayssm_force_flush = force_flush_bool.to(
                        device=self.device, dtype=torch.int32
                    )
                # Advance only valid slots, scattered over unique slots (dup-index
                # race; padded rows share slot 0); a forced flush -> next write_pos 0.
                valid_mask = slots >= 0
                valid_slots = slots[valid_mask]
                if valid_slots.numel() > 0:
                    flushed = replayssm_write_pos == (L - 1)
                    if replayssm_force_flush is not None:
                        flushed = flushed | (replayssm_force_flush != 0)
                    next_pos = torch.where(
                        flushed,
                        torch.zeros_like(replayssm_write_pos),
                        (replayssm_write_pos + 1) % L,
                    )
                    # Dedup: rows sharing a slot share write_pos/flush, so the
                    # scattered value is identical regardless of which row wins.
                    uniq_slots, inv = torch.unique(valid_slots, return_inverse=True)
                    next_for_valid = next_pos[valid_mask]
                    new_vals = torch.empty(
                        uniq_slots.shape[0],
                        dtype=write_pos_buf.dtype,
                        device=write_pos_buf.device,
                    )
                    new_vals[inv] = next_for_valid.to(write_pos_buf.dtype)
                    write_pos_buf[uniq_slots] = new_vals
        elif logical_mode.is_extend(include_draft_extend_v2=True):
            has_extend_meta = (
                forward_batch.extend_start_loc is not None
                and forward_batch.extend_seq_lens is not None
            )
            if forward_batch.forward_mode.is_draft_extend_v2() and not has_extend_meta:
                # Draft-extend-v2 may omit linear metadata when the draft runs only
                # full-attention layers.
                query_start_loc = None
            elif forward_batch.forward_mode.is_target_verify():
                ragged_layout = forward_batch.spec_info.ragged_verify_layout
                if ragged_layout is not None:
                    # Compact ragged verify: variable per-request verify lens.
                    query_start_loc = ragged_layout.qo_indptr_device
                else:
                    query_start_loc = torch.arange(
                        0,
                        forward_batch.input_ids.shape[0] + 1,
                        step=forward_batch.spec_info.draft_token_num,
                        dtype=torch.int32,
                        device=forward_batch.input_ids.device,
                    )

                if self.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrieve_next_token
                    retrieve_next_sibling = (
                        forward_batch.spec_info.retrieve_next_sibling
                    )
                    # None during dummy run
                    if retrieve_next_token is not None:
                        retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.extend_seq_lens_cpu is not None
                    and len(forward_batch.extend_seq_lens_cpu) == bs
                    and forward_batch.tbo_parent_token_range is None
                ):
                    logical_num_tokens = sum(forward_batch.extend_seq_lens_cpu)
                else:
                    logical_num_tokens = int(query_start_loc[-1])
                if has_mamba_track_mask:
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch, track_mask_indices
                    )

                    (
                        track_chunk_idx,
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_h_batch_src,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                        track_ssm_seq_idx,
                        track_ssm_end_locs,
                        track_ssm_recompute_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            logical_num_tokens=logical_num_tokens,
            mamba_track_mask_indices=track_mask_indices,
            mamba_cache_indices=mamba_cache_indices,
            # Physical track destinations (None when tracking off); cuda-graph
            # supplies this via the static backend buffer in _replay_metadata.
            mamba_track_indices=getattr(forward_batch, "mamba_track_indices", None),
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_h_batch_src=track_ssm_h_batch_src,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            track_chunk_idx=track_chunk_idx,
            track_ssm_seq_idx=track_ssm_seq_idx,
            track_ssm_end_locs=track_ssm_end_locs,
            track_ssm_recompute_dst=track_ssm_recompute_dst,
            has_mamba_track_mask=has_mamba_track_mask,
            replayssm_write_pos=replayssm_write_pos,
            replayssm_force_flush=replayssm_force_flush,
        )

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
        in_capture: bool = False,
        mamba_track_indices: Optional[torch.Tensor] = None,
    ):
        if num_padding is None:
            if seq_lens_cpu is None:
                num_padding = 0
            else:
                num_padding = torch.count_nonzero(
                    seq_lens_cpu == self._graph_seq_len_fill_value
                )
        num_padding = int(num_padding)
        if self._fused_state_indices_ok and self.replayssm_write_pos_list is None:
            # Single-launch fast path: mapping gather + padding sentinel + store
            # into the static buffer, plus zeroing padded req_pool_indices rows —
            # bit-identical to the reference chain below.
            mamba_indices = self._fused_replay_state_indices(
                req_pool_indices=req_pool_indices,
                mamba_index_mapping=self.req_to_token_pool.req_index_to_mamba_index_mapping,
                out_state_indices=self.state_indices_list[bs - 1],
                valid_bs=bs - int(num_padding),
                total_bs=bs,
                v2p=self.req_to_token_pool.mamba_v2p_table,
            )
        else:
            # Make sure forward metadata is correctly handled for padding reqs
            req_pool_indices[bs - num_padding :] = 0
            mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
            # Translate using the LIVE v2p table BEFORE the padding sentinel below;
            # captured Mamba kernels read state_indices_list as PHYSICAL ids.
            mamba_indices = self._translate_mamba_indices(mamba_indices)
            mamba_indices[bs - num_padding :] = -1
            self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if _validate_mamba_replay_state_indices and not in_capture:
            # This pre-replay diagnostic intentionally syncs to reject malformed
            # live or padded indices before a captured state update uses them.
            valid_bs = bs - int(num_padding)
            validate_replay_state_indices_cpu(
                mamba_indices.detach().cpu(),
                valid_bs=valid_bs,
                total_bs=bs,
                num_state_slots=self.req_to_token_pool.mamba_pool.size + 1,
                pad_slot_id=self.pad_slot_id,
            )
        # Refresh the static track-dest buffer in-place (translated); the captured
        # track-save reads it, leaving the handed-in InputBuffer slot read-only.
        # Hand out only the refreshed [:bs] prefix — Mamba2's track-save slices
        # [-num_decodes:], which on the full max_bs buffer binds the stale tail.
        track_buf = None
        if mamba_track_indices is not None:
            assert len(mamba_track_indices) >= bs, (
                f"{len(mamba_track_indices)=} < {bs=}"
            )
            track_buf = self.mamba_track_indices_buf[:bs]
            track_buf.copy_(self._translate_mamba_indices(mamba_track_indices[:bs]))
        # Refresh the static write cursor in-place (mirrors the eager
        # snapshot-then-advance). Skip the advance during capture: dummy slots
        # would corrupt real ring positions.
        replayssm_write_pos = None
        replayssm_force_flush = None
        if self.replayssm_write_pos_list is not None:
            mamba_pool = self.req_to_token_pool.mamba_pool
            write_pos_buf = mamba_pool.replayssm_write_pos
            static_wp = self.replayssm_write_pos_list[bs - 1]
            static_ff = self.replayssm_force_flush_list[bs - 1]
            # Hand the full captured per-bs buffers to the kernel; it indexes per row.
            replayssm_write_pos = static_wp
            replayssm_force_flush = static_ff
            if write_pos_buf is not None:
                # this replay's per-row physical slots (padded rows == -1)
                slots = mamba_indices.to(torch.long)
                safe_slots = slots.clamp(min=0)
                # Snapshot this step's cursor into the captured buffer in-place
                # (copy_, never reassign the object).
                static_wp[: len(mamba_indices)].copy_(write_pos_buf[safe_slots])
                # Refresh the force-flush buffer in-place from this step's seq_lens
                # (same condition as the radix track). Zeroed during capture.
                force_flush_dev = None
                # KDA: no radix coordination -> leave zeroed so the advance is a pure wrap.
                is_kda = getattr(mamba_pool, "replayssm_is_kda", False)
                if (
                    not is_kda
                    and forward_mode.is_decode_or_idle()
                    and seq_lens_cpu is not None
                ):
                    ff_mask = self._replayssm_track_flush_mask(seq_lens_cpu, bs)
                    force_flush_dev = ff_mask.to(device=self.device, dtype=torch.int32)
                    static_ff.copy_(force_flush_dev)
                else:
                    static_ff.zero_()
                # Defense in depth: the decode-ring advance is only meaningful for
                # decode/idle forwards (mirrors the eager path's gating). A
                # TARGET_VERIFY replay must never advance the cursor.
                if not in_capture and forward_mode.is_decode_or_idle():
                    L = mamba_pool.linear_replayssm_cache_len
                    # Advance only valid (non-padded) slots; a forced flush empties
                    # the ring -> next write_pos 0, like the natural L-1 wrap.
                    valid_mask = slots >= 0
                    valid_slots = slots[valid_mask]
                    if valid_slots.numel() > 0:
                        cur_pos = write_pos_buf[safe_slots]
                        flushed = cur_pos == (L - 1)
                        if force_flush_dev is not None:
                            flushed = flushed | (force_flush_dev != 0)
                        next_pos = torch.where(
                            flushed,
                            torch.zeros_like(cur_pos),
                            (cur_pos + 1) % L,
                        )
                        # Dedup; rows sharing a slot share write_pos+flush.
                        uniq_slots, inv = torch.unique(valid_slots, return_inverse=True)
                        next_for_valid = next_pos[valid_mask]
                        new_vals = torch.empty(
                            uniq_slots.shape[0],
                            dtype=write_pos_buf.dtype,
                            device=write_pos_buf.device,
                        )
                        new_vals[inv] = next_for_valid.to(write_pos_buf.dtype)
                        write_pos_buf[uniq_slots] = new_vals
        is_target_verify = forward_mode.is_target_verify()
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            ragged_layout = (
                spec_info.ragged_verify_layout if spec_info is not None else None
            )
            if ragged_layout is not None:
                # Mamba kernels index dense [bs, N] scratch, so they need the
                # capped variant (see padded_to_bucket). Padding rows carry
                # mamba slot -1 and are skipped.
                if ragged_layout.bs != bs or ragged_layout.cap is None:
                    ragged_layout = ragged_layout.padded_to_bucket(
                        padded_bs=bs, cap=spec_info.draft_token_num
                    )
                self.query_start_loc_list[bs - 1].copy_(ragged_layout.qo_indptr_device)
            elif num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        qsl_buf = self.query_start_loc_list[bs - 1]

        if is_target_verify and self.topk > 1:
            if (
                spec_info is not None
                and getattr(spec_info, "retrieve_next_token", None) is not None
            ):
                bs_without_pad = spec_info.retrieve_next_token.shape[0]
                self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_token
                )
                self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_sibling
                )
            return ForwardMetadata(
                query_start_loc=qsl_buf,
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
        else:
            return ForwardMetadata(
                query_start_loc=qsl_buf,
                mamba_cache_indices=self.state_indices_list[bs - 1],
                mamba_track_indices=track_buf,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )

class _HybridMethods:
    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend.get_indexer_metadata(layer_id, forward_batch)
        return None

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cpu_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                out_cache_loc,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For linear attention
        b: Optional[torch.Tensor] = None,  # For linear attention
        **kwargs,
    ):
        is_linear_attn = not self._is_full_attn(layer, kwargs.get("layer_id"))

        # For linear attention, dispatch on the logical mode to match the
        # built metadata.
        dispatch_mode = (
            forward_batch.logical_forward_mode
            if is_linear_attn
            else forward_batch.forward_mode
        )

        if forward_batch.forward_mode.is_idle():
            if is_linear_attn:
                return mixed_qkv.new_empty(
                    mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif dispatch_mode.is_decode():
            return self.forward_decode(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )
        else:
            return self.forward_extend(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
        req_pool_indices: Optional[torch.Tensor] = None,
    ):
        """Commit accepted Mamba states; PP rows identify the delayed batch."""
        request_number = last_correct_step_indices.shape[0]
        src_indices_raw = None
        if pp_spec_stable_rows_enabled() and req_pool_indices is not None:
            src_indices_raw = req_pool_indices[:request_number]

        # `mamba_track_indices` is VIRTUAL; the scatter writes physical views.
        if mamba_track_indices is not None:
            mamba_track_indices = self.linear_attn_backend._translate_mamba_indices(
                mamba_track_indices
            )

        if src_indices_raw is not None:
            state_indices_tensor = self.linear_attn_backend._translate_mamba_indices(
                self.linear_attn_backend.req_to_token_pool.get_mamba_indices(
                    req_pool_indices[:request_number]
                )
            )
        else:
            state_indices_tensor = (
                self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                    :request_number
                ]
            )

        req_pool = self.linear_attn_backend.req_to_token_pool
        mamba_caches = req_pool.get_speculative_mamba2_params_all_layers()

        # ReplaySSM-KDA: the accepted drafts live in the per-slot ring (written
        # during verify); no intermediate_ssm is allocated. Replay the accepted
        # prefix into `temporal` instead of scattering an intermediate state.
        # dspark/dflash call this method directly; the generic spec_utils commit
        # handles replayssm before reaching here (returns early), so this branch is
        # only hit by the direct callers. Chain layout only (topk <= 1), so
        # accept_lens == last_correct_step_indices + 1.
        mamba_pool = req_pool.mamba_pool
        if getattr(mamba_pool, "replayssm_is_kda", False):
            from hcu_sglang.layers.attention.linear.kda_replayssm import (
                commit_kda_replayssm_after_verify,
            )

            commit_kda_replayssm_after_verify(
                spec_state=mamba_caches,
                use_hcu_kda=mamba_pool.use_hcu_kda,
                state_batch_indices=state_indices_tensor,
                accept_lens=last_correct_step_indices + 1,
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                src_indices_raw=src_indices_raw,
                null_block_id=-1,
            )
            return

        # KDA fused-accept: the next verify seeds itself in-kernel from the
        # accepted checkpoint slot (recurrent_kda's num_accepted_tokens), so the
        # SSM state never round-trips through `temporal` and only the conv
        # windows still need the accept rollback. Recording this round's accept
        # length is what selects that seed next round; chain layout only (see
        # above), so accept_lens == last_correct_step_indices + 1. The pool
        # exists only where KDAAttnBackend found the contract satisfied, which
        # includes mamba radix tracking being off.
        accept_lens_pool = self.linear_attn_backend.accept_lens_pool
        if accept_lens_pool is not None:
            assert mamba_track_indices is None, "fused-accept runs with radix off"
            for conv_states, intermediate_conv_window in zip(
                mamba_caches.conv, mamba_caches.intermediate_conv_window
            ):
                fused_conv_window_scatter_with_mask(
                    conv_states,
                    intermediate_conv_window,
                    state_indices_tensor,
                    last_correct_step_indices,
                    src_indices_raw,
                )
            accept_lens_pool[state_indices_tensor.to(torch.int64)] = (
                last_correct_step_indices.to(torch.int32) + 1
            )
            return

        _linear = getattr(self, "linear_attn_backend", None)
        if _linear is not None:
            from hcu_sglang.layers.attention.linear import gdn_bplus

            gdn_bplus.compensate_from_stash(
                _linear,
                state_indices_tensor,
                last_correct_step_indices,
                mamba_caches,
            )
            # Handoff: verify/compensate already wrote the committed rows, so
            # only the pointer moves and the 6 GiB ssm copy is skipped.
            if gdn_bplus.handoff_enabled() and src_indices_raw is None:
                # Host int: .item() here would serialize the CPU behind the
                # verify every decode step. The forward's own record is
                # capture-time only, so prefer the live width.
                _st = gdn_bplus.state(_linear)
                T = int(_st.live_T or _st.capture_T)
                _cp_ok = T > 1 and gdn_bplus.commit_prev(
                    _linear,
                    state_indices_tensor,
                    last_correct_step_indices,
                    T,
                    int(mamba_caches.intermediate_ssm.shape[2]),
                )
                if _cp_ok:
                    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
                        _conv_multi_eligible,
                        fused_conv_window_scatter_multi,
                    )

                    pairs = list(
                        zip(mamba_caches.conv, mamba_caches.intermediate_conv_window)
                    )
                    if pairs:
                        if _conv_multi_eligible(pairs):
                            fused_conv_window_scatter_multi(
                                pairs,
                                state_indices_tensor,
                                last_correct_step_indices,
                                None,
                                None,
                            )
                        else:
                            for conv_states, intermediate_conv_window in pairs:
                                fused_conv_window_scatter_with_mask(
                                    conv_states,
                                    intermediate_conv_window,
                                    state_indices_tensor,
                                    last_correct_step_indices,
                                    None,
                                )
                    self._update_ple_state_after_mtp_verify(
                        state_indices_tensor,
                        last_correct_step_indices,
                        mamba_track_indices,
                        mamba_steps_to_track,
                        src_indices_raw,
                    )
                    return
        scatter_mamba_states_after_mtp_verify(
            mamba_caches,
            state_indices_tensor,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
            src_indices_raw=src_indices_raw,
        )

        self._update_ple_state_after_mtp_verify(
            state_indices_tensor,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
            src_indices_raw,
        )

class _MetadataMethods:
    def prepare_mixed(
        cls,
        forward_metadata: ForwardMetadata,
        chunk_size: int,
        forward_batch: ForwardBatch,
    ) -> "Mamba2Metadata":
        """This path cannot run with CUDA graph, as it contains extend requests."""
        if (
            forward_batch.extend_num_tokens is None
            or forward_batch.logical_forward_mode.is_decode()
        ):
            draft_token_num = (
                forward_batch.spec_info.draft_token_num
                if forward_batch.spec_info is not None
                else 1
            )
            return cls.prepare_decode(
                forward_metadata,
                forward_batch.seq_lens,
                is_target_verify=forward_batch.forward_mode.is_target_verify(),
                draft_token_num=draft_token_num,
            )
        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        if extend_seq_lens_cpu is None:
            num_prefills = len(forward_batch.extend_seq_lens)
        else:
            num_prefills = len(extend_seq_lens_cpu)
        if extend_seq_lens_cpu is not None:
            num_prefill_tokens = int(sum(extend_seq_lens_cpu))
        else:
            num_prefill_tokens = int(forward_batch.extend_num_tokens)
        num_decodes = len(forward_batch.seq_lens) - num_prefills
        context_lens_tensor = forward_batch.extend_prefix_lens
        assert context_lens_tensor is not None
        has_initial_states = context_lens_tensor > 0
        prep_initial_states = torch.any(has_initial_states[:num_prefills]).item()

        query_start_loc = forward_metadata.query_start_loc[: num_prefills + 1]
        _seq_idx_output_size = (
            num_prefill_tokens if extend_seq_lens_cpu is not None else None
        )
        seq_idx = torch.repeat_interleave(
            torch.arange(
                num_prefills, dtype=torch.int32, device=query_start_loc.device
            ),
            query_start_loc.diff(),
            output_size=_seq_idx_output_size,
        )
        seq_idx.unsqueeze_(0)

        # We compute metadata for chunked prefill once at the top level model
        # forward and reuse them in mamba layers. If not needed, they will be
        # ignored inside mamba kernels.
        chunk_offsets, chunk_indices = None, None
        if prep_initial_states:
            chunk_indices, chunk_offsets = (
                cls._query_start_loc_to_chunk_indices_offsets(
                    query_start_loc, chunk_size, num_prefill_tokens
                )
            )

        draft_token_num = (
            getattr(forward_batch.spec_info, "draft_token_num", 1)
            if forward_batch.spec_info is not None
            else 1
        )
        # Resolve the tracked-row selection once per forward
        mamba_track_mask_indices = None
        conv_states_mask_indices = None
        if forward_metadata.has_mamba_track_mask:
            mamba_track_mask_indices = forward_batch.mamba_track_mask.nonzero(
                as_tuple=True
            )[0]
            conv_states_mask_indices = forward_batch.mamba_track_indices[
                mamba_track_mask_indices
            ]
        return Mamba2Metadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=forward_metadata.mamba_cache_indices,
            mamba_track_indices=forward_metadata.mamba_track_indices,
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
            track_conv_indices=forward_metadata.track_conv_indices,
            track_ssm_h_src=forward_metadata.track_ssm_h_src,
            track_ssm_h_dst=forward_metadata.track_ssm_h_dst,
            track_ssm_final_src=forward_metadata.track_ssm_final_src,
            track_ssm_final_dst=forward_metadata.track_ssm_final_dst,
            track_ssm_seq_idx=forward_metadata.track_ssm_seq_idx,
            track_ssm_end_locs=forward_metadata.track_ssm_end_locs,
            track_ssm_recompute_dst=forward_metadata.track_ssm_recompute_dst,
            has_mamba_track_mask=forward_metadata.has_mamba_track_mask,
            mamba_track_mask_indices=mamba_track_mask_indices,
            conv_states_mask_indices=conv_states_mask_indices,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decodes=num_decodes,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
            mixed_metadata=cls.MixedMetadata(
                has_initial_states=has_initial_states,
                prep_initial_states=prep_initial_states,
                chunk_size=chunk_size,
                seq_idx=seq_idx,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
            ),
        )
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
        raise ImportError("KDA runtime namespace lacks " + str(sorted(missing)))
    fn = types.FunctionType(raw.__code__, vars(module), name, raw.__defaults__, raw.__closure__)
    fn.__kwdefaults__ = raw.__kwdefaults__
    fn.__annotations__ = raw.__annotations__
    fn.__qualname__ = owner.__qualname__ + "." + name
    for decorator in reversed(decorators):
        if decorator == "classmethod":
            fn = classmethod(fn)
        elif decorator == "staticmethod":
            fn = staticmethod(fn)
        elif decorator == "property":
            fn = property(fn)
        else:
            raise RuntimeError("Unknown KDA descriptor: " + decorator)
    return fn


def _track_skip_enabled() -> bool:
    """Radix-off serving never consumes the per-step mamba track checkpoints."""
    return __import__("os").environ.get("SGLANG_GDN_TRACK_SKIP") == "1"


def _around_set_mamba_track_indices(original, batch, track_positions=None):
    if not _track_skip_enabled():
        return original(batch, track_positions)
    if track_positions is None:
        track_positions = [
            (
                req.kv.mamba_next_track_idx
                if req.kv.mamba_next_track_idx is not None
                else 0
            )
            for req in batch.reqs
        ]
    # Keep the host-side list (finished-request bookkeeping reads it) but skip
    # the pinned tensor + device gather: with --disable-radix-cache nothing
    # consumes the track checkpoints every step.
    batch.mamba_track_buffer_indices = list(track_positions)
    batch.mamba_track_indices = None
    return None


def register_hooks():
    global _REGISTERED
    if not is_hcu():
        return
    if _REGISTERED:
        from hcu_sglang import speculative_kda
        speculative_kda.register_hooks()
        return
    K._hcu_causal_conv1d_fn = _hcu_causal_conv1d_fn
    K._hcu_causal_conv1d_update = _hcu_causal_conv1d_update
    K._run_causal_conv1d_fn = _run_causal_conv1d_fn
    K._run_causal_conv1d_update = _run_causal_conv1d_update
    for name, decorators in [('__init__', [])]:
        setattr(K.KDAKernelDispatcher, name, _bind(vars(_DispatcherMethods)[name], K, K.KDAKernelDispatcher, name, decorators))
    for name, decorators in [('__init__', []), ('init_forward_metadata', []), ('forward_decode', []), ('forward_extend', []), ('_hcu_forward_decode', []), ('_hcu_forward_extend', [])]:
        setattr(K.KDAAttnBackend, name, _bind(vars(_KdaMethods)[name], K, K.KDAAttnBackend, name, decorators))
    for name, decorators in [('__init__', []), ('_forward_metadata', []), ('init_forward_metadata_capture_cpu_graph', []), ('_replay_metadata', [])]:
        setattr(H.MambaAttnBackendBase, name, _bind(vars(_BaseMethods)[name], H, H.MambaAttnBackendBase, name, decorators))
    for name, decorators in [('get_indexer_metadata', []), ('init_forward_metadata_capture_cpu_graph', []), ('forward', []), ('update_mamba_state_after_mtp_verify', [])]:
        setattr(H.HybridLinearAttnBackend, name, _bind(vars(_HybridMethods)[name], H, H.HybridLinearAttnBackend, name, decorators))
    for name, decorators in [('prepare_mixed', ['classmethod'])]:
        setattr(MD.Mamba2Metadata, name, _bind(vars(_MetadataMethods)[name], MD, MD.Mamba2Metadata, name, decorators))
    from hcu_sglang import speculative_kda
    speculative_kda.register_hooks()
    if _track_skip_enabled():
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType

        HookRegistry.register(
            "sglang.srt.managers.schedule_batch.set_mamba_track_indices_from_reqs",
            _around_set_mamba_track_indices,
            HookType.AROUND,
        )
    _REGISTERED = True
