"""HCU overlay on sglang's GDN backend.

Everything not redefined here is upstream code. Ours: ``forward_decode``
(B+ commit, HIP recurrent, HCU conv), ``forward_extend`` (verify kernel
entries; prefill stays upstream) and ``_convolve_prefill``; the decode conv
update itself is ``gdn_conv.update``. General fixes belong upstream.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import sglang.srt.layers.attention.linear.gdn_backend as upstream
from hcu_sglang.layers.attention.linear import (
    gdn_commit,
    gdn_recurrent,
)
from hcu_sglang.utils.detection import is_hcu
from sglang.srt.utils import get_bool_env_var

globals().update({name: value for name, value in vars(upstream).items()
                  if not name.startswith("__")})
_is_hcu = is_hcu()
logger = logging.getLogger(__name__)
_GDN_BACKEND_MODULE = "sglang.srt.layers.attention.linear.gdn_backend"
# Resolve lazily through the hook registry: gdn_source may be imported after the
# REPLACE hooks are installed, when GDNAttnBackend.forward_extend is already the
# wrapper. A module-level alias would capture that wrapper and recurse.
_UPSTREAM_FORWARD_EXTEND = []


def _upstream_forward_extend(self, *args, **kwargs):
    if not _UPSTREAM_FORWARD_EXTEND:
        from hcu_sglang.utils.module_patch import get_original_attr

        _UPSTREAM_FORWARD_EXTEND.append(
            get_original_attr(_GDN_BACKEND_MODULE, "GDNAttnBackend.forward_extend")
        )
    return _UPSTREAM_FORWARD_EXTEND[0](self, *args, **kwargs)
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

from hcu_sglang.layers.attention.linear import gdn_conv

gdn_conv.configure(
    triton=causal_conv1d_update,
    hcu=causal_conv1d_update_hcu if _use_hcu_causal_conv1d else None,
    log=rank0_log,
)
conv_update = gdn_conv.update

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
    original,
    self,
    layer: RadixLinearAttention,
    forward_batch: ForwardBatch,
    mixed_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """HCU conv1d for prefill; the upstream implementation otherwise."""
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
_OVERLAY_LOG = {"done": False}

def forward_extend(
    self,
    layer: RadixLinearAttention,
    forward_batch: ForwardBatch,
    mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    a: torch.Tensor,
    b: torch.Tensor,
    **kwargs,
):
    """Prefill stays upstream; target verify takes the B+/HIP entries."""
    hip_out = _verify_attempt(self, layer, forward_batch, mixed_qkv, a, b)
    if hip_out is not None:
        return hip_out
    return _upstream_forward_extend(
        self, layer, forward_batch, mixed_qkv, a, b, **kwargs
    )


class _VerifyPlan(NamedTuple):
    """The per-forward shape the overlay needs, resolved once per layer."""

    seq_len: int
    draft_token_num: int
    batch_size: int
    meta: object
    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    mamba_cache: object
    conv_states: torch.Tensor
    ssm_states: torch.Tensor
    intermediate_state_cache: torch.Tensor
    intermediate_indices: torch.Tensor


def _verify_plan(self, layer, forward_batch, mixed_qkv):
    """The runtime shape of our verify, or None when this forward is not ours."""
    if not forward_batch.forward_mode.is_target_verify():
        return None
    if not gdn_recurrent.overlay_enabled():
        return None
    if self.mis_metadata is not None or not isinstance(mixed_qkv, torch.Tensor):
        return None
    if mixed_qkv.shape[0] == 0:
        return None
    mamba_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    if (mamba_cache.replayssm_d is not None
            or mamba_cache.replayssm_rawv is not None):
        return None
    intermediate_state_cache = mamba_cache.intermediate_ssm
    assert intermediate_state_cache is not None, (
        "recurrent target_verify requires intermediate_ssm, "
        "which is not allocated under --enable-linear-replayssm-spec"
    )
    meta = self.forward_metadata
    cache_indices = meta.mamba_cache_indices
    query_start_loc = meta.query_start_loc
    # The mask is only read by the PP stable-rows path; computing it per
    # layer otherwise costs one elementwise launch per GDN layer per step.
    intermediate_indices = select_verify_intermediate_state_indices(
        self.verify_intermediate_state_indices,
        forward_batch.req_pool_indices,
        cache_indices[: query_start_loc.shape[0] - 1] >= 0
        if pp_spec_stable_rows_enabled()
        else None,
        self.req_to_token_pool.size,
    )
    seq_len = int(mixed_qkv.shape[0])
    draft_token_num = int(forward_batch.spec_info.draft_token_num)
    return _VerifyPlan(
        seq_len=seq_len,
        draft_token_num=draft_token_num,
        batch_size=seq_len // draft_token_num,
        meta=meta,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        mamba_cache=mamba_cache,
        conv_states=mamba_cache.conv[0],
        ssm_states=mamba_cache.temporal,
        intermediate_state_cache=intermediate_state_cache,
        intermediate_indices=intermediate_indices,
    )


def _split_verify_qkv(self, mixed_qkv, layer, seq_len):
    """The fused split when the head layout allows it, torch.split otherwise."""
    if (layer.q_dim + layer.k_dim + layer.v_dim <= MAX_FUSED_QKV_SPLIT_DIM
            and not self._use_strided_target_verify_qkv):
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
        mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
    )
    return (
        query.view(1, seq_len, layer.num_q_heads, layer.head_q_dim),
        key.view(1, seq_len, layer.num_k_heads, layer.head_k_dim),
        value.view(1, seq_len, layer.num_v_heads, layer.head_v_dim),
    )


def _kernel_verify(plan, layer, query, key, value, a, b):
    """The full-snapshot kernel, used when the B+ commit declines."""
    if not _HIP_LOG["done"]:
        rank0_log("GDN HIP verify fallback: full-snapshot kernel")
        _HIP_LOG["done"] = True
    cache = plan.intermediate_state_cache
    flat = cache.reshape(-1, *cache.shape[2:])
    out = query.new_empty((1, query.shape[1], value.shape[2], value.shape[3]))
    gdn_recurrent.verify_full(
        query,
        key,
        value,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        plan.query_start_loc[: plan.batch_size + 1],
        plan.ssm_states,
        plan.cache_indices[: plan.batch_size].to(torch.int32),
        flat,
        plan.intermediate_indices[: plan.batch_size].to(torch.int32),
        int(cache.shape[1]),
        plan.draft_token_num,
        out,
    )
    return out


def _verify_from_plan(self, plan, layer, mixed_qkv, a, b):
    if not _OVERLAY_LOG["done"]:
        _OVERLAY_LOG["done"] = True
        logger.info(
            "GDN overlay verify: SGLANG_GDN_HIP=1 "
            "(prenorm, B+ plain, conv=HIP); decode/prefill stay on the dispatcher"
        )
    mixed_qkv = gdn_conv.update_required(
        mixed_qkv.view(plan.batch_size, plan.draft_token_num, -1).transpose(1, 2),
        plan.conv_states,
        layer.conv_weights,
        layer.bias,
        layer.activation,
        conv_state_indices=plan.cache_indices[: plan.batch_size],
        intermediate_conv_window=plan.mamba_cache.intermediate_conv_window[0],
        intermediate_state_indices=plan.intermediate_indices[: plan.batch_size],
        retrieve_next_token=plan.meta.retrieve_next_token,
        retrieve_next_sibling=plan.meta.retrieve_next_sibling,
        retrieve_parent_token=plan.meta.retrieve_parent_token,
    ).transpose(1, 2).view(plan.seq_len, -1)
    query, key, value = _split_verify_qkv(self, mixed_qkv, layer, plan.seq_len)
    out = gdn_commit.verify_step(
        backend=self,
        layer=layer,
        q=query,
        k=key,
        v=value,
        a=a,
        b=b,
        draft_token_num=plan.draft_token_num,
        n=plan.batch_size,
        snap_cache=plan.intermediate_state_cache,
        cache_indices=plan.cache_indices,
        intermediate_indices=plan.intermediate_indices,
        mamba_cache=plan.mamba_cache,
        ss_states=plan.ssm_states,
        query_start_loc=plan.query_start_loc,
    )
    if out is not None:
        return out
    return _kernel_verify(plan, layer, query, key, value, a, b)


def _verify_attempt(self, layer, forward_batch, mixed_qkv, a, b):
    """The overlaid verify result, or None when this forward is not ours."""
    plan = _verify_plan(self, layer, forward_batch, mixed_qkv)
    if plan is None:
        return None
    return _verify_from_plan(self, plan, layer, mixed_qkv, a, b)
