from __future__ import annotations
import sglang.srt.layers.attention.linear.gdn_backend as upstream
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
    global _fused_decode_proj_conv_fallback_logged
    global _fused_decode_proj_conv_logged
    global _fused_decode_proj_conv_layers_logged
    global _fused_decode_real_tensor_verified_layers

    if _is_hip and isinstance(mixed_qkv, torch.Tensor) and mixed_qkv.shape[0] == 0:
        return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

    layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states = layer_cache.conv[0]
    ssm_states = layer_cache.temporal
    query_start_loc = self.forward_metadata.query_start_loc
    cache_indices = self.forward_metadata.mamba_cache_indices
    # GDN ReplaySSM (slice 1a): per-layer ring slices + the once-per-forward
    # per-row write cursor. All None unless --enable-linear-replayssm, so the
    # legacy dispatch below is byte-identical when the flag is off.
    replayssm_write_pos = self.forward_metadata.replayssm_write_pos
    # GDN ReplaySSM (slice 2b): per-row force-flush at radix track
    # boundaries (None unless --enable-linear-replayssm). When present the
    # kernel folds the ring into temporal[slot] on the snapshot steps.
    replayssm_force_flush = self.forward_metadata.replayssm_force_flush
    replayssm_d = layer_cache.replayssm_d
    replayssm_k = layer_cache.replayssm_k
    replayssm_g = layer_cache.replayssm_g

    return_z = False
    conv_already_applied = False
    if isinstance(mixed_qkv, tuple):
        if len(mixed_qkv) != 2:
            raise ValueError(
                "Fused GDN decode projection input must be "
                "(projected_qkvz, projected_ba)"
            )
        projected_qkvz, projected_ba = mixed_qkv
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

            # Compare real activations against the direct-Triton update on a
            # compact state copy, leaving the live cache to the candidate.
            verify_real_tensors = (
                _fused_decode_verify_real_tensors
                and layer.layer_id not in _fused_decode_real_tensor_verified_layers
            )
            if verify_real_tensors:
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
            conv_already_applied = True
        else:
            if not _fused_decode_proj_conv_fallback_logged:
                rank0_log(
                    "Falling back from fused GDN decode projection/Conv1D: "
                    f"{eligibility_reason}"
                )
                _fused_decode_proj_conv_fallback_logged = True
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                projected_qkvz,
                projected_ba,
                layer.num_q_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_v_dim,
            )
        return_z = True
    else:
        assert isinstance(mixed_qkv, torch.Tensor)

    if not conv_already_applied:
        conv_update = (
            causal_conv1d_update_hcu
            if _use_hcu_causal_conv1d
            else causal_conv1d_update
        )
        mixed_qkv = conv_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

    # Skip split + reshape + separate gating kernel by consuming
    # the packed mixed_qkv directly in a single fused Triton kernel.
    if self.kernel_dispatcher.supports_packed_decode:
        core_attn_out = self.kernel_dispatcher.packed_decode(
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
        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )
        return (core_attn_out, z) if return_z else core_attn_out

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

    core_attn_out = self.kernel_dispatcher.decode(
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

    self._track_mamba_state_decode(
        forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
    )

    return (core_attn_out, z) if return_z else core_attn_out

def _convolve_prefill(
    self,
    layer: RadixLinearAttention,
    forward_batch: ForwardBatch,
    mixed_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    block_table = self.forward_metadata.conv_block_table
    use_hcu_conv = _use_hcu_causal_conv1d and block_table is None
    conv_fn = causal_conv1d_fn_hcu if use_hcu_conv else causal_conv1d_fn
    conv_kwargs = {"initial_states" if use_hcu_conv else "conv_states": conv_states}
    if block_table is not None:
        conv_kwargs["block_table"] = block_table
    return conv_fn(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation=layer.activation,
        has_initial_state=forward_batch.extend_prefix_lens > 0,
        cache_indices=cache_indices,
        query_start_loc=self.forward_metadata.query_start_loc,
        seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        **conv_kwargs,
    ).transpose(0, 1)[: mixed_qkv.shape[0]]


_HIP_LOG = {"done": False}


def _gdn_hip_kernel_enabled() -> bool:
    if not _is_hcu:
        return False
    try:
        from hcu_sglang.layers.attention.linear import gdn_hip_bplus

        return gdn_hip_bplus.enabled_kernel()
    except Exception:
        return False


def _gdn_hip_bplus_enabled() -> bool:
    if not _is_hcu:
        return False
    try:
        from hcu_sglang.layers.attention.linear import gdn_bplus

        return gdn_bplus.enabled()
    except Exception:
        return False


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
    if is_target_verify:
        assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
        intermediate_state_cache = mamba_cache_params.intermediate_ssm
        intermediate_conv_window_cache = (
            mamba_cache_params.intermediate_conv_window[0]
        )
        intermediate_state_indices = select_verify_intermediate_state_indices(
            self.verify_intermediate_state_indices,
            forward_batch.req_pool_indices,
            cache_indices[: query_start_loc.shape[0] - 1] >= 0,
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

    # Page-major envelope: the prefill kernels (CUDA causal_conv1d_fwd,
    # chunk_gated_delta_rule) write state back in place assuming a contiguous
    # slot layout, so they silently drop the write to the strided envelope
    # pool. Run them on contiguous per-sequence copies (identity-indexed) and
    # scatter the result back. No-op for the default contiguous pool.
    # CPU kernels (causal_conv1d_fwd_cpu, chunk_gated_delta_rule_cpu) use
    # proper indexed writes and handle non-contiguous pools directly via
    # cache_indices, so the gather/scatter round-trip is unnecessary on CPU.
    # TODO(ch-wan): drop these .contiguous() copies by making the prefill conv
    # and chunk_gated_delta_rule kernels honor the pool's real slot stride +
    # int64 indexing, like packed_decode / causal_conv1d_update already do.
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

    if is_target_verify:
        batch_size = seq_len // forward_batch.spec_info.draft_token_num
        draft_token_num = forward_batch.spec_info.draft_token_num
        mixed_qkv_reshaped = mixed_qkv.view(
            batch_size, draft_token_num, -1
        ).transpose(1, 2)
        mixed_qkv_processed = causal_conv1d_update(
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
        mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
    else:
        if forward_metadata.has_mamba_track_mask:
            mixed_qkv_to_track = mixed_qkv[
                forward_metadata.track_conv_indices
            ].transpose(1, 2)
            conv_states[forward_metadata.conv_states_mask_indices] = (
                mixed_qkv_to_track
            )

        mixed_qkv = self._convolve_prefill(
            layer,
            forward_batch,
            mixed_qkv,
            conv_states_contig,
            state_cache_indices,
        )

    actual_seq_len = mixed_qkv.shape[0]
    qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
    if (
        (is_cuda() or is_hip() or is_xpu())
        and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM
        and not self._use_strided_target_verify_qkv
    ):
        query, key, value = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            layer.num_q_heads,
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_q_dim,
            layer.head_k_dim,
            layer.head_v_dim,
        )
    else:
        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

    if is_target_verify:
        # ReplaySSM verify protocols: fold-every-commit (ring-write during
        # verify, fold on commit), circular ring, or the snapshotting
        # fallback when neither ring is allocated.
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
            # The recurrent fallback needs the per-draft snapshots, which
            # the pool gates OFF under --enable-linear-replayssm-spec (the
            # same flag that makes `use_replayssm_spec` true above), so
            # this branch is unreachable with a None buffer by
            # construction -- keep it loud rather than silently frozen.
            assert intermediate_state_cache is not None, (
                "recurrent target_verify fallback requires intermediate_ssm, "
                "which is not allocated under --enable-linear-replayssm-spec"
            )
            if _gdn_hip_bplus_enabled():
                from hcu_sglang.layers.attention.linear import (
                    gdn_bplus,
                    gdn_hip_bplus,
                )

                n = batch_size
                if gdn_bplus.stash(
                    self, layer, query, key, value, a, b, draft_token_num
                ):
                    flat = intermediate_state_cache.reshape(
                        -1, *intermediate_state_cache.shape[2:]
                    )
                    if gdn_bplus.handoff_enabled():
                        # Slot-indexed scratch rows: the committed row then has a
                        # stable address, so commit only moves an int32 pointer.
                        # Row stride is the allocated cache_steps, not the
                        # runtime draft-token count (adaptive steps vary it).
                        cache_steps = int(intermediate_state_cache.shape[1])
                        slots = cache_indices[:n].to(torch.int32)
                        snap_idx = slots
                        prev = gdn_bplus.ensure_prev(
                            self, slots, mamba_cache_params, cache_steps
                        )
                        gdn_bplus.remember_step(self, slots, prev, cache_steps)
                        h0_src = flat
                        h0_idx = gdn_bplus.h0_rows(slots, prev, cache_steps)
                        prev_arg = getattr(self, "_gdn_hip_prev")
                    else:
                        snap_idx = intermediate_state_indices[:n].to(torch.int32)
                        h0_src = ssm_states
                        h0_idx = cache_indices[:n].to(torch.int32)
                        prev_arg = None
                    self._gdn_hip_snap_idx = snap_idx
                    hip_out = query.new_empty(
                        (1, query.shape[1], value.shape[2], value.shape[3])
                    )
                    gdn_hip_bplus.verify(
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
                    )
                    return hip_out
            if _gdn_hip_kernel_enabled():
                from hcu_sglang.layers.attention.linear import gdn_hip_bplus

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
                gdn_hip_bplus.verify_full(
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
            core_attn_out = self.kernel_dispatcher.target_verify(
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
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
                stable_rows=pp_spec_stable_rows_enabled(),
            )
    else:
        g, beta = self._prefill_gates(layer, a, b)
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

        if forward_metadata.has_mamba_track_mask:
            self._track_mamba_state_extend(
                forward_batch,
                h,
                ssm_states,
                forward_metadata,
                h_track_buf=h_track_buf,
            )

        from hcu_sglang.layers.attention.linear import gdn_bplus

        inter = getattr(mamba_cache_params, "intermediate_ssm", None)
        if inter is not None:
            gdn_bplus.seed_prev(
                self, cache_indices, mamba_cache_params, int(inter.shape[1])
            )

    return core_attn_out
