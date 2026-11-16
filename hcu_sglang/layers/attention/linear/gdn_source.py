"""HCU overlay on sglang's GDN backend.

The upstream module is imported and its namespace copied in wholesale, so
everything this file does not redefine is upstream code. Only these upstream
names are redefined, each for one reason:

- ``forward_decode``      decode step: B+ commit/handoff, the HIP recurrent
                          entry, and the HCU causal-conv path.
- ``forward_extend``      prefill/verify: B+ seed and stash, the HIP verify
                          entry, and the HCU conv.
- ``_convolve_prefill``   HCU causal_conv1d instead of the Triton conv.
- ``conv_update``         the same swap for the single-token update.

When a change here is general rather than HCU-specific it belongs upstream:
see flash-attention-das ``fix/varlen-layout`` for the pattern.
"""

from __future__ import annotations
import sglang.srt.layers.attention.linear.gdn_backend as upstream
from hcu_sglang.layers.attention.linear import gdn_commit, gdn_recurrent
from hcu_sglang.utils.detection import is_hcu
from sglang.srt.utils import get_bool_env_var

globals().update({name: value for name, value in vars(upstream).items()
                  if not name.startswith("__")})
_is_hcu = is_hcu()
_use_hcu_causal_conv1d = False
if _is_hcu and get_bool_env_var("SGLANG_USE_CAUSAL_CONV1D"):
    try:
        from causal_conv1d import causal_conv1d_fn_hcu
        from causal_conv1d.causal_conv1d_interface import (
            causal_conv1d_update as causal_conv1d_update_hcu,
        )

        _use_hcu_causal_conv1d = True
        rank0_log("Using HCU causal_conv1d for GDN decode/prefill.")
    except ImportError as exc:
        rank0_log(
            "SGLANG_USE_CAUSAL_CONV1D=1 but HCU causal_conv1d import failed "
            f"({exc}); falling back to Triton."
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
    if _is_hip and isinstance(mixed_qkv, torch.Tensor) and mixed_qkv.shape[0] == 0:
        return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

    layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states = layer_cache.conv[0]
    ssm_states = layer_cache.temporal
    query_start_loc = self.forward_metadata.query_start_loc
    cache_indices = self.forward_metadata.mamba_cache_indices
    # Per-layer ring slices and the per-row write cursor; all None unless
    # --enable-linear-replayssm, so the legacy dispatch is byte-identical.
    replayssm_write_pos = self.forward_metadata.replayssm_write_pos
    # Per-row force-flush at radix track boundaries; None unless
    # --enable-linear-replayssm.
    replayssm_force_flush = self.forward_metadata.replayssm_force_flush
    replayssm_d = layer_cache.replayssm_d
    replayssm_k = layer_cache.replayssm_k
    replayssm_g = layer_cache.replayssm_g

    return_z = False
    conv_already_applied = False
    if isinstance(mixed_qkv, tuple):
        mixed_qkv, z, b, a, conv_already_applied = _decode_projected_input(
            self,
            layer=layer,
            mixed_qkv=mixed_qkv,
            conv_states=conv_states,
            cache_indices=cache_indices,
        )
        return_z = True
    else:
        assert isinstance(mixed_qkv, torch.Tensor)

    if not conv_already_applied:
        mixed_qkv = _decode_conv(
            self,
            layer=layer,
            mixed_qkv=mixed_qkv,
            conv_states=conv_states,
            cache_indices=cache_indices,
        )

    core_attn_out = _decode_hip_core(
        self,
        layer=layer,
        forward_batch=forward_batch,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        replayssm_d=replayssm_d,
        replayssm_k=replayssm_k,
        replayssm_g=replayssm_g,
        replayssm_write_pos=replayssm_write_pos,
        replayssm_force_flush=replayssm_force_flush,
    )
    self._track_mamba_state_decode(
        forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
    )

    return (core_attn_out, z) if return_z else core_attn_out


def _decode_projected_input(
    self, *, layer, mixed_qkv, conv_states, cache_indices
):
    if len(mixed_qkv) != 2:
        raise ValueError(
            "Fused GDN decode projection input must be "
            "(projected_qkvz, projected_ba)"
        )
    projected_qkvz, projected_ba = mixed_qkv
    return _decode_fused_proj_conv(
        self,
        layer=layer,
        projected_qkvz=projected_qkvz,
        projected_ba=projected_ba,
        conv_states=conv_states,
        cache_indices=cache_indices,
    )


def _decode_fused_proj_conv(
    self,
    *,
    layer,
    projected_qkvz,
    projected_ba,
    conv_states,
    cache_indices,
):
    fused, eligibility_reason = _decode_fusion_intro(
        self,
        layer=layer,
        projected_qkvz=projected_qkvz,
        projected_ba=projected_ba,
        conv_states=conv_states,
        cache_indices=cache_indices,
    )
    if fused is None:
        mixed_qkv, z, b, a = _decode_split_proj(
            self,
            layer=layer,
            projected_qkvz=projected_qkvz,
            projected_ba=projected_ba,
            eligibility_reason=eligibility_reason,
        )
        return mixed_qkv, z, b, a, False

    qkv_dim, fused_backend = fused
    # Compare real activations against the direct-Triton update on a
    # compact state copy, leaving the live cache to the candidate.
    verify_real_tensors = (
        _fused_decode_verify_real_tensors
        and layer.layer_id not in _fused_decode_real_tensor_verified_layers
    )
    reference_values = None
    if verify_real_tensors:
        reference_values = _fused_decode_reference(
            self,
            layer=layer,
            projected_qkvz=projected_qkvz,
            projected_ba=projected_ba,
            conv_states=conv_states,
            cache_indices=cache_indices,
        )

    mixed_qkv, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(
        projected_qkvz,
        projected_ba,
        conv_states,
        layer.conv_weights,
        layer.bias,
        cache_indices,
        qkv_dim=qkv_dim,
        v_dim=layer.v_dim,
        num_v_heads=layer.num_v_heads,
        head_v_dim=layer.head_v_dim,
        activation=layer.activation,
    )
    if verify_real_tensors:
        _report_fused_decode_parity(
            self,
            layer=layer,
            fused_backend=fused_backend,
            outputs=(mixed_qkv, z, b, a),
            conv_states=conv_states,
            cache_indices=cache_indices,
            reference_values=reference_values,
        )
    return mixed_qkv, z, b, a, True


def _decode_fusion_intro(
    self,
    *,
    layer,
    projected_qkvz,
    projected_ba,
    conv_states,
    cache_indices,
):
    global _fused_decode_proj_conv_logged

    eligible, eligibility_reason = (
        can_use_fused_qkvzba_causal_conv1d_update_contiguous(
            projected_qkvz,
            projected_ba,
            conv_states,
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
        )
    )
    if eligible and not _use_hcu_causal_conv1d:
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        fused_backend = "triton_direct_oracle_exact"
        if not _fused_decode_proj_conv_logged:
            rank0_log("Using fused GDN decode QKVZ/BA unpack + indexed Conv1D.")
            _fused_decode_proj_conv_logged = True
        if (
            _fused_decode_log_layer_hits or _fused_decode_verify_real_tensors
        ) and layer.layer_id not in _fused_decode_proj_conv_layers_logged:
            rank0_log(
                "GDN_FUSED_DECODE_BACKEND "
                f"layer_id={layer.layer_id} backend={fused_backend} "
                f"batch={projected_qkvz.shape[0]} "
                f"qkv_dim={qkv_dim} state_shape={tuple(conv_states.shape)} "
                f"state_indices_dtype={cache_indices.dtype}"
            )
            _fused_decode_proj_conv_layers_logged.add(layer.layer_id)
        return (qkv_dim, fused_backend), eligibility_reason
    return None, eligibility_reason


def _fused_decode_reference(
    self,
    *,
    layer,
    projected_qkvz,
    projected_ba,
    conv_states,
    cache_indices,
):
    if bool(torch.any(cache_indices < 0).item()):
        raise AssertionError(
            "Real-tensor GDN fusion verification requires "
            "non-padding cache indices"
        )
    ref_indices = torch.arange(
        cache_indices.numel(),
        device=cache_indices.device,
        dtype=torch.int32,
    )
    ref_state = torch.index_select(
        conv_states, 0, cache_indices.to(torch.int64)
    )
    ref_mixed_qkv, ref_z, ref_b, ref_a = (
        fused_qkvzba_split_reshape_cat_contiguous(
            projected_qkvz,
            projected_ba,
            layer.num_q_heads,
            layer.num_v_heads,
            layer.head_q_dim,
            layer.head_v_dim,
        )
    )
    ref_mixed_qkv = causal_conv1d_update(
        ref_mixed_qkv,
        ref_state,
        layer.conv_weights,
        layer.bias,
        layer.activation,
        conv_state_indices=ref_indices,
    )
    return ref_mixed_qkv, ref_z, ref_b, ref_a, ref_state


def _report_fused_decode_parity(
    self,
    *,
    layer,
    fused_backend,
    outputs,
    conv_states,
    cache_indices,
    reference_values,
):
    mixed_qkv, z, b, a = outputs
    ref_mixed_qkv, ref_z, ref_b, ref_a, ref_state = reference_values
    candidate_state = torch.index_select(
        conv_states, 0, cache_indices.to(torch.int64)
    )
    named_pairs = (
        ("qkv", mixed_qkv, ref_mixed_qkv),
        ("z", z, ref_z),
        ("b", b, ref_b),
        ("a", a, ref_a),
        ("state", candidate_state, ref_state),
    )
    report = []
    mismatch = False
    for tensor_name, candidate, reference in named_pairs:
        diff = (candidate.float() - reference.float()).abs()
        nonzero = int(torch.count_nonzero(diff).item())
        mismatch |= nonzero != 0
        report.append(
            f"{tensor_name}_nonzero={nonzero}/"
            f"{diff.numel()} {tensor_name}_max="
            f"{diff.max().item()}"
        )
    rank0_log(
        "GDN_FUSED_REAL_TENSOR_PARITY "
        f"layer_id={layer.layer_id} backend={fused_backend} "
        + " ".join(report)
    )
    _fused_decode_real_tensor_verified_layers.add(layer.layer_id)
    if mismatch:
        raise AssertionError(
            "GDN fused real-tensor parity failed at "
            f"layer_id={layer.layer_id}; " + " ".join(report)
        )


def _decode_split_proj(
    self, *, layer, projected_qkvz, projected_ba, eligibility_reason
):
    global _fused_decode_proj_conv_fallback_logged

    if not _fused_decode_proj_conv_fallback_logged:
        rank0_log(
            "Falling back from fused GDN decode projection/Conv1D: "
            f"{eligibility_reason}"
        )
        _fused_decode_proj_conv_fallback_logged = True
    return fused_qkvzba_split_reshape_cat_contiguous(
        projected_qkvz,
        projected_ba,
        layer.num_q_heads,
        layer.num_v_heads,
        layer.head_q_dim,
        layer.head_v_dim,
    )


def _decode_conv(self, *, layer, mixed_qkv, conv_states, cache_indices):
    return conv_update(
        mixed_qkv,
        conv_states,
        layer.conv_weights,
        layer.bias,
        layer.activation,
        conv_state_indices=cache_indices,
    )


def _decode_hip_core(
    self,
    *,
    layer,
    forward_batch,
    mixed_qkv,
    a,
    b,
    ssm_states,
    cache_indices,
    query_start_loc,
    replayssm_d,
    replayssm_k,
    replayssm_g,
    replayssm_write_pos,
    replayssm_force_flush,
):
    # Skip split + reshape + separate gating kernel by consuming
    # the packed mixed_qkv directly in a single fused Triton kernel.
    if self.kernel_dispatcher.supports_packed_decode:
        return self.kernel_dispatcher.packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            scale=layer.head_k_dim**-0.5,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=layer.num_v_heads,
            head_v_dim=layer.head_v_dim,
            replayssm_d=replayssm_d,
            replayssm_k=replayssm_k,
            replayssm_g=replayssm_g,
            replayssm_write_pos=replayssm_write_pos,
            replayssm_force_flush=replayssm_force_flush,
        )

    query, key, value = torch.split(
        mixed_qkv,
        [layer.q_dim, layer.k_dim, layer.v_dim],
        dim=-1,
    )
    # Reshape from [bs, h*d] to [1, bs, h, d]
    bs = forward_batch.batch_size
    query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
    key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
    value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

    return self.kernel_dispatcher.decode(
        q=query,
        k=key,
        v=value,
        a=a,
        b=b,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )


def _convolve_prefill(
    original,
    self,
    layer: RadixLinearAttention,
    forward_batch: ForwardBatch,
    mixed_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """HCU conv1d for prefill; the upstream implementation otherwise.

    The HCU library takes the conv state as ``initial_states`` and has no block
    table, so a pooled (paged) conv state stays on the upstream call.
    """
    if not _use_hcu_causal_conv1d:
        return original(self, layer, forward_batch, mixed_qkv, conv_states,
                        cache_indices)
    if self.forward_metadata.conv_block_table is not None:
        return original(self, layer, forward_batch, mixed_qkv, conv_states,
                        cache_indices)
    return causal_conv1d_fn_hcu(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation=layer.activation,
        initial_states=conv_states,
        has_initial_state=forward_batch.extend_prefix_lens > 0,
        cache_indices=cache_indices,
        query_start_loc=self.forward_metadata.query_start_loc,
        seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
    ).transpose(0, 1)[: mixed_qkv.shape[0]]


_HIP_LOG = {"done": False}


def _gdn_hip_kernel_enabled() -> bool:
    return _is_hcu and gdn_recurrent.enabled_kernel()


def _gdn_recurrent_enabled() -> bool:
    return _is_hcu and gdn_commit.enabled()


_CONV_HIP_LOG = {"done": False}


def conv_update(x, conv_state, weight, bias=None, activation=None, **kwargs):
    """Triton causal_conv1d_update, or its HIP twin under SGLANG_GDN_HIP_CONV=1.

    The HIP kernel is bitwise equal for the bf16 / width-4 / spec-verify shapes
    and returns None for anything else, so the Triton path stays the fallback.
    """
    if _is_hcu:
        try:
            from hcu_sglang.layers.attention.linear import gdn_recurrent

            if gdn_recurrent.conv_enabled():
                out = gdn_recurrent.conv_update(
                    x, conv_state, weight, bias, activation, **kwargs
                )
                if out is not None:
                    return out
                if not _CONV_HIP_LOG["done"]:
                    _CONV_HIP_LOG["done"] = True
                    rank0_log(
                        "SGLANG_GDN_HIP_CONV=1 but this conv call is outside "
                        "the HIP kernel "
                        f"({gdn_recurrent.conv_last_reason()}); using Triton."
                    )
        except Exception as exc:  # noqa: BLE001 - never fail a forward here
            if not _CONV_HIP_LOG["done"]:
                _CONV_HIP_LOG["done"] = True
                rank0_log(f"SGLANG_GDN_HIP_CONV path disabled: {exc}")
    return causal_conv1d_update(x, conv_state, weight, bias, activation, **kwargs)


def forward_extend(
    self,
    layer: RadixLinearAttention,
    forward_batch: ForwardBatch,
    mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    a: torch.Tensor,
    b: torch.Tensor,
    **kwargs,
):
    assert isinstance(mixed_qkv, torch.Tensor)
    seq_len = mixed_qkv.shape[0]

    if _is_hip and seq_len == 0:
        return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

    is_target_verify = forward_batch.forward_mode.is_target_verify()
    forward_metadata = self.forward_metadata

    query_start_loc = forward_metadata.query_start_loc
    cache_indices = forward_metadata.mamba_cache_indices
    retrieve_next_token = forward_metadata.retrieve_next_token
    retrieve_next_sibling = forward_metadata.retrieve_next_sibling
    retrieve_parent_token = forward_metadata.retrieve_parent_token

    mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states = mamba_cache_params.conv[0]
    ssm_states = mamba_cache_params.temporal
    if self.mis_metadata is not None:
        if is_target_verify:
            raise ValueError("GDN MIS does not support target verify")
        return self._forward_extend_mis(
            layer=layer,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            conv_states=conv_states,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
        )

    verify = None
    if is_target_verify:
        verify = _target_verify_context(
            self,
            forward_batch=forward_batch,
            mamba_cache_params=mamba_cache_params,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_len=seq_len,
        )
    return _extend_forward(
        self,
        layer=layer,
        forward_batch=forward_batch,
        forward_metadata=forward_metadata,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        seq_len=seq_len,
        is_target_verify=is_target_verify,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token,
        mamba_cache_params=mamba_cache_params,
        conv_states=conv_states,
        ssm_states=ssm_states,
        verify=verify,
        kwargs=kwargs,
    )


def _target_verify_context(
    self,
    *,
    forward_batch,
    mamba_cache_params,
    cache_indices,
    query_start_loc,
    seq_len,
):
    assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
    intermediate_state_cache = mamba_cache_params.intermediate_ssm
    intermediate_conv_window_cache = (
        mamba_cache_params.intermediate_conv_window[0]
    )
    # The mask is only read by the PP stable-rows path; computing it per
    # layer otherwise costs one elementwise launch per GDN layer per step.
    stable_rows = pp_spec_stable_rows_enabled()
    valid_rows = (
        cache_indices[: query_start_loc.shape[0] - 1] >= 0
        if stable_rows
        else None
    )
    intermediate_state_indices = select_verify_intermediate_state_indices(
        self.verify_intermediate_state_indices,
        forward_batch.req_pool_indices,
        valid_rows,
        self.req_to_token_pool.size,
    )
    mamba_pool = self.req_to_token_pool.mamba_pool
    use_replayssm_fold = (
        mamba_cache_params.replayssm_rawv is not None
        and getattr(mamba_pool, "replayssm_spec_fold", False)
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    )
    use_replayssm_spec = (
        mamba_cache_params.replayssm_d is not None
        and getattr(mamba_pool, "replayssm_cache_base", None) is not None
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    )
    batch_size = seq_len // forward_batch.spec_info.draft_token_num
    draft_token_num = forward_batch.spec_info.draft_token_num
    return {
        "intermediate_state_cache": intermediate_state_cache,
        "intermediate_conv_window_cache": intermediate_conv_window_cache,
        "intermediate_state_indices": intermediate_state_indices,
        "mamba_pool": mamba_pool,
        "use_replayssm_fold": use_replayssm_fold,
        "use_replayssm_spec": use_replayssm_spec,
        "batch_size": batch_size,
        "draft_token_num": draft_token_num,
    }


def _maybe_gather_envelope_states(
    self,
    *,
    is_target_verify,
    cache_indices,
    conv_states,
    ssm_states,
):
    # Page-major envelope: the prefill kernels (CUDA causal_conv1d_fwd,
    # chunk_gated_delta_rule) write state back in place assuming a contiguous
    # slot layout, so a strided envelope pool would silently drop the write.
    # Run them on contiguous per-sequence copies (identity-indexed) and scatter
    # the result back; CPU kernels write through cache_indices natively.
    # TODO(ch-wan): make the prefill conv and chunk_gated_delta_rule kernels
    # honor the pool's real slot stride + int64 indexing so this round-trip
    # can go away.
    needs_state_gather = (
        self.requires_contiguous_prefill_state
        and (not is_target_verify)
        and (not is_cpu())
        and (not conv_states.is_contiguous() or not ssm_states.is_contiguous())
    )
    if needs_state_gather:
        conv_states_contig = conv_states[cache_indices].contiguous()
        ssm_states_contig = ssm_states[cache_indices].contiguous()
        state_cache_indices = torch.arange(
            cache_indices.shape[0],
            device=cache_indices.device,
            dtype=cache_indices.dtype,
        )
    else:
        conv_states_contig = conv_states
        ssm_states_contig = ssm_states
        state_cache_indices = cache_indices
    return (
        conv_states_contig,
        ssm_states_contig,
        state_cache_indices,
        needs_state_gather,
    )


def _extend_conv(
    self,
    *,
    layer,
    forward_batch,
    forward_metadata,
    mixed_qkv,
    seq_len,
    conv_states,
    cache_indices,
    gather,
    is_target_verify,
    verify,
    retrieve_next_token,
    retrieve_next_sibling,
    retrieve_parent_token,
):
    conv_states_contig = gather[0]
    state_cache_indices = gather[2]
    if is_target_verify:
        batch_size = verify["batch_size"]
        draft_token_num = verify["draft_token_num"]
        intermediate_conv_window_cache = verify["intermediate_conv_window_cache"]
        intermediate_state_indices = verify["intermediate_state_indices"]
        mixed_qkv_reshaped = mixed_qkv.view(
            batch_size, draft_token_num, -1
        ).transpose(1, 2)
        mixed_qkv_processed = conv_update(
            mixed_qkv_reshaped,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices[:batch_size],
            intermediate_conv_window=intermediate_conv_window_cache,
            intermediate_state_indices=intermediate_state_indices[:batch_size],
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
        )
        return mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)

    if forward_metadata.has_mamba_track_mask:
        mixed_qkv_to_track = mixed_qkv[
            forward_metadata.track_conv_indices
        ].transpose(1, 2)
        conv_states[forward_metadata.conv_states_mask_indices] = (
            mixed_qkv_to_track
        )

    return self._convolve_prefill(
        layer,
        forward_batch,
        mixed_qkv,
        conv_states_contig,
        state_cache_indices,
    )


def _split_extend_qkv(self, *, layer, mixed_qkv):
    actual_seq_len = mixed_qkv.shape[0]
    qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
    if (
        (is_cuda() or is_hip() or is_xpu())
        and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM
        and not self._use_strided_target_verify_qkv
    ):
        return fused_qkv_split_gdn_prefill(
            mixed_qkv,
            layer.num_q_heads,
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_q_dim,
            layer.head_k_dim,
            layer.head_v_dim,
        )
    query, key, value = torch.split(
        mixed_qkv,
        [layer.q_dim, layer.k_dim, layer.v_dim],
        dim=-1,
    )
    query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
    key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
    value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)
    return query, key, value


def _target_verify(
    self,
    *,
    layer,
    qkv,
    a,
    b,
    forward_batch,
    mamba_cache_params,
    ssm_states,
    cache_indices,
    query_start_loc,
    retrieve_parent_token,
    verify,
):
    query, key, value = qkv
    use_replayssm_fold = verify["use_replayssm_fold"]
    use_replayssm_spec = verify["use_replayssm_spec"]
    mamba_pool = verify["mamba_pool"]
    if use_replayssm_fold:
        core_attn_out = self._replayssm_fold_target_verify(
            layer=layer,
            query=query,
            key=key,
            value=value,
            a=a,
            b=b,
            layer_cache=mamba_cache_params,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            retrieve_parent_token=retrieve_parent_token,
        )
    elif use_replayssm_spec:
        core_attn_out = self._replayssm_target_verify(
            layer=layer,
            query=query,
            key=key,
            value=value,
            a=a,
            b=b,
            mamba_pool=mamba_pool,
            layer_cache=mamba_cache_params,
            cache_indices=cache_indices,
            replay_indices=forward_batch.req_pool_indices,
            query_start_loc=query_start_loc,
            draft_token_num=forward_batch.spec_info.draft_token_num,
        )
    else:
        core_attn_out = _recurrent_verify(
            self,
            layer=layer,
            qkv=qkv,
            a=a,
            b=b,
            mamba_cache_params=mamba_cache_params,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            retrieve_parent_token=retrieve_parent_token,
            verify=verify,
        )
    return core_attn_out


def _recurrent_verify(
    self,
    *,
    layer,
    qkv,
    a,
    b,
    mamba_cache_params,
    ssm_states,
    cache_indices,
    query_start_loc,
    retrieve_parent_token,
    verify,
):
    intermediate_state_cache = verify["intermediate_state_cache"]
    intermediate_state_indices = verify["intermediate_state_indices"]
    batch_size = verify["batch_size"]
    draft_token_num = verify["draft_token_num"]
    # The recurrent fallback needs the per-draft snapshots, which the pool
    # gates OFF under --enable-linear-replayssm-spec (the same flag that makes
    # use_replayssm_spec true above), so this branch is unreachable with a
    # None buffer by construction -- keep it loud rather than silently frozen.
    assert intermediate_state_cache is not None, (
        "recurrent target_verify fallback requires intermediate_ssm, "
        "which is not allocated under --enable-linear-replayssm-spec"
    )
    if _gdn_recurrent_enabled():
        hip_out = _recurrent_verify_commit(
            self,
            layer=layer,
            qkv=qkv,
            a=a,
            b=b,
            mamba_cache_params=mamba_cache_params,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_state_cache=intermediate_state_cache,
            intermediate_state_indices=intermediate_state_indices,
            batch_size=batch_size,
            draft_token_num=draft_token_num,
        )
        if hip_out is not None:
            return hip_out
    if _gdn_hip_kernel_enabled():
        return _recurrent_verify_hip(
            self,
            layer=layer,
            qkv=qkv,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_state_cache=intermediate_state_cache,
            intermediate_state_indices=intermediate_state_indices,
            batch_size=batch_size,
            draft_token_num=draft_token_num,
        )
    return _recurrent_verify_fallback(
        self,
        layer=layer,
        qkv=qkv,
        a=a,
        b=b,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        intermediate_state_cache=intermediate_state_cache,
        intermediate_state_indices=intermediate_state_indices,
        retrieve_parent_token=retrieve_parent_token,
        draft_token_num=draft_token_num,
    )


def _recurrent_verify_commit(
    self,
    *,
    layer,
    qkv,
    a,
    b,
    mamba_cache_params,
    ssm_states,
    cache_indices,
    query_start_loc,
    intermediate_state_cache,
    intermediate_state_indices,
    batch_size,
    draft_token_num,
):
    from hcu_sglang.layers.attention.linear import (
        gdn_commit,
        gdn_recurrent,
    )

    query, key, value = qkv
    n = batch_size
    if not gdn_commit.stash(
        self, layer, query, key, value, a, b, draft_token_num
    ):
        return None
    flat = intermediate_state_cache.reshape(
        -1, *intermediate_state_cache.shape[2:]
    )
    snap_idx, h0_src, h0_idx, prev_arg, store_arg, store_row_arg = (
        _commit_snapshot_args(
            self,
            layer_id=layer.layer_id,
            cache_indices=cache_indices,
            mamba_cache_params=mamba_cache_params,
            ssm_states=ssm_states,
            intermediate_state_cache=intermediate_state_cache,
            intermediate_state_indices=intermediate_state_indices,
            flat=flat,
            n=n,
            draft_token_num=draft_token_num,
        )
    )
    gdn_commit.state(self).snap_idx = snap_idx
    hip_out = query.new_empty(
        (1, query.shape[1], value.shape[2], value.shape[3])
    )
    gdn_recurrent.verify(
        query,
        key,
        value,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        query_start_loc[: n + 1],
        h0_src,
        h0_idx,
        flat,
        snap_idx,
        int(intermediate_state_cache.shape[1]),
        int(draft_token_num),
        hip_out,
        prev_step=prev_arg,
        store=store_arg,
        store_row=store_row_arg,
    )
    return hip_out


def _commit_snapshot_args(
    self,
    *,
    layer_id,
    cache_indices,
    mamba_cache_params,
    ssm_states,
    intermediate_state_cache,
    intermediate_state_indices,
    flat,
    n,
    draft_token_num,
):
    store_arg = store_row_arg = None
    if gdn_commit.handoff_enabled():
        # Slot-indexed scratch rows: the committed row then has a stable
        # address, so commit only moves an int32 pointer. Row stride is the
        # allocated cache_steps, not the runtime draft-token count (adaptive
        # steps vary it).
        cache_steps = int(intermediate_state_cache.shape[1])
        slots = cache_indices[:n].to(torch.int32)
        snap_idx = slots
        gdn_commit.begin_verify_step(self, layer_id)
        sh = gdn_commit.shared_step_values(
            self,
            slots,
            mamba_cache_params,
            cache_steps,
            int(draft_token_num),
        )
        h0_src = flat
        h0_idx = sh["rows"][cache_steps]["h0"]
        store_arg = sh["store"]
        store_row_arg = sh["rows"][cache_steps]["store_row"]
        prev_arg = gdn_commit.state(self).prev
    else:
        snap_idx = intermediate_state_indices[:n].to(torch.int32)
        h0_src = ssm_states
        h0_idx = cache_indices[:n].to(torch.int32)
        prev_arg = None
    return snap_idx, h0_src, h0_idx, prev_arg, store_arg, store_row_arg


def _recurrent_verify_hip(
    self,
    *,
    layer,
    qkv,
    a,
    b,
    ssm_states,
    cache_indices,
    query_start_loc,
    intermediate_state_cache,
    intermediate_state_indices,
    batch_size,
    draft_token_num,
):
    from hcu_sglang.layers.attention.linear import gdn_recurrent

    query, key, value = qkv
    if not _HIP_LOG["done"]:
        rank0_log("GDN HIP kernel-only verify path active")
        _HIP_LOG["done"] = True

    n = batch_size
    flat = intermediate_state_cache.reshape(
        -1, *intermediate_state_cache.shape[2:]
    )
    hip_out = query.new_empty(
        (1, query.shape[1], value.shape[2], value.shape[3])
    )
    gdn_recurrent.verify_full(
        query,
        key,
        value,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        query_start_loc[: n + 1],
        ssm_states,
        cache_indices[:n].to(torch.int32),
        flat,
        intermediate_state_indices[:n].to(torch.int32),
        int(intermediate_state_cache.shape[1]),
        int(draft_token_num),
        hip_out,
    )
    return hip_out


def _recurrent_verify_fallback(
    self,
    *,
    layer,
    qkv,
    a,
    b,
    ssm_states,
    cache_indices,
    query_start_loc,
    intermediate_state_cache,
    intermediate_state_indices,
    retrieve_parent_token,
    draft_token_num,
):
    query, key, value = qkv
    return self.kernel_dispatcher.target_verify(
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        q=query,
        k=key,
        v=value,
        a=a,
        b=b,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        intermediate_states_buffer=intermediate_state_cache,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=draft_token_num,
        retrieve_parent_token=retrieve_parent_token,
        stable_rows=pp_spec_stable_rows_enabled(),
    )


def _extend_prefill(
    self,
    *,
    layer,
    forward_batch,
    forward_metadata,
    mamba_cache_params,
    qkv,
    a,
    b,
    conv_states,
    ssm_states,
    cache_indices,
    query_start_loc,
    gather,
    kwargs,
):
    (
        conv_states_contig,
        ssm_states_contig,
        state_cache_indices,
        needs_state_gather,
    ) = gather
    core_attn_out, h, h_track_buf = _extend_prefill_kernel(
        self,
        layer=layer,
        forward_batch=forward_batch,
        forward_metadata=forward_metadata,
        qkv=qkv,
        a=a,
        b=b,
        conv_states=conv_states,
        conv_states_contig=conv_states_contig,
        ssm_states=ssm_states,
        ssm_states_contig=ssm_states_contig,
        cache_indices=cache_indices,
        state_cache_indices=state_cache_indices,
        query_start_loc=query_start_loc,
        needs_state_gather=needs_state_gather,
        kwargs=kwargs,
    )

    if forward_metadata.has_mamba_track_mask:
        self._track_mamba_state_extend(
            forward_batch,
            h,
            ssm_states,
            forward_metadata,
            h_track_buf=h_track_buf,
        )

    from hcu_sglang.layers.attention.linear import gdn_commit

    # Real prefill only: the MTP draft extend also takes this branch, and
    # seeding there copies the whole state (6 GiB per draft step) for slots
    # that already have a live handoff row.
    inter = getattr(mamba_cache_params, "intermediate_ssm", None)
    if inter is not None and forward_batch.forward_mode.is_extend_without_speculative():
        gdn_commit.seed_prev(
            self, cache_indices, mamba_cache_params, int(inter.shape[1])
        )

    return core_attn_out


def _extend_prefill_kernel(
    self,
    *,
    layer,
    forward_batch,
    forward_metadata,
    qkv,
    a,
    b,
    conv_states,
    conv_states_contig,
    ssm_states,
    ssm_states_contig,
    cache_indices,
    state_cache_indices,
    query_start_loc,
    needs_state_gather,
    kwargs,
):
    query, key, value = qkv
    g, beta = self._prefill_gates(layer, a, b)
    h_track_buf = _extend_track_buffer(
        self,
        forward_metadata=forward_metadata,
        cache_indices=cache_indices,
        ssm_states=ssm_states,
    )
    core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        ssm_states=ssm_states_contig,
        cache_indices=state_cache_indices,
        query_start_loc=query_start_loc,
        state_checkpoint_cu_starts=(
            forward_metadata.state_checkpoint_cu_starts
        ),
        num_state_checkpoints=forward_metadata.num_state_checkpoints,
        state_checkpoint_every_n_tokens=(
            forward_metadata.state_checkpoint_every_n_tokens
        ),
        output=kwargs.get("linear_attn_output"),
        layer_id=layer.layer_id,
        extend_prefix_lens=forward_batch.extend_prefix_lens,
        track_state=h_track_buf,
        track_chunk_idx=(
            forward_metadata.track_chunk_idx
            if h_track_buf is not None
            else None
        ),
    )

    if is_npu() and last_recurrent_state is not None:
        last_recurrent_state = last_recurrent_state.to(
            ssm_states.dtype, copy=False
        )
        ssm_states[cache_indices] = last_recurrent_state

    if needs_state_gather:
        # Scatter the in-place-updated contiguous copies back to the
        # strided envelope pool (advanced indexing handles the strides).
        conv_states[cache_indices] = conv_states_contig
        ssm_states[cache_indices] = ssm_states_contig

    return core_attn_out, h, h_track_buf


def _extend_track_buffer(
    self,
    *,
    forward_metadata,
    cache_indices,
    ssm_states,
):
    h_track_buf = None
    if (
        forward_metadata.has_mamba_track_mask
        and forward_metadata.track_ssm_h_src.numel() > 0
        and self.kernel_dispatcher.extend_kernel.supports_track_state_snapshot
    ):
        # Snapshot-capable kernels can supply just the selected FP32
        # checkpoint instead of materializing all per-chunk states.
        assert forward_metadata.track_chunk_idx is not None
        h_track_buf = torch.empty(
            (len(cache_indices), *ssm_states.shape[1:]),
            dtype=torch.float32,
            device=ssm_states.device,
        )
    return h_track_buf


def _extend_forward(
    self,
    *,
    layer,
    forward_batch,
    forward_metadata,
    mixed_qkv,
    a,
    b,
    seq_len,
    is_target_verify,
    query_start_loc,
    cache_indices,
    retrieve_next_token,
    retrieve_next_sibling,
    retrieve_parent_token,
    mamba_cache_params,
    conv_states,
    ssm_states,
    verify,
    kwargs,
):
    gather = _maybe_gather_envelope_states(
        self,
        is_target_verify=is_target_verify,
        cache_indices=cache_indices,
        conv_states=conv_states,
        ssm_states=ssm_states,
    )
    mixed_qkv = _extend_conv(
        self,
        layer=layer,
        forward_batch=forward_batch,
        forward_metadata=forward_metadata,
        mixed_qkv=mixed_qkv,
        seq_len=seq_len,
        conv_states=conv_states,
        cache_indices=cache_indices,
        gather=gather,
        is_target_verify=is_target_verify,
        verify=verify,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_parent_token=retrieve_parent_token,
    )
    qkv = _split_extend_qkv(self, layer=layer, mixed_qkv=mixed_qkv)

    if is_target_verify:
        return _target_verify(
            self,
            layer=layer,
            qkv=qkv,
            a=a,
            b=b,
            forward_batch=forward_batch,
            mamba_cache_params=mamba_cache_params,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            retrieve_parent_token=retrieve_parent_token,
            verify=verify,
        )
    return _extend_prefill(
        self,
        layer=layer,
        forward_batch=forward_batch,
        forward_metadata=forward_metadata,
        mamba_cache_params=mamba_cache_params,
        qkv=qkv,
        a=a,
        b=b,
        conv_states=conv_states,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
        gather=gather,
        kwargs=kwargs,
    )
