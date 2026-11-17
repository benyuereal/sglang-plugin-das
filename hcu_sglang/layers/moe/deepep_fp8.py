# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations
import functools
from typing import Callable, Optional

import torch
from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
from sglang.srt.utils import get_bool_env_var
from hcu_sglang.layers.moe.deepep_current import (
    DeepEPLLDispatchOutput,
    DeepEPNormalDispatchOutput,
)
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.layers.moe import deepep_current as D
from sglang.srt.runtime_context import get_disagg
from hcu_sglang.layers.moe.deepep_ep_kernels import ep_scatter, ep_gather, build_m_indices_triton
from sglang.srt.utils.offloader import get_offloader
from sglang.srt.layers.moe.moe_runner.deep_gemm import copy_list_to_gpu_no_ce
from lightop import moe as lightop_op
from lightop.activation import fuse_silu_mul_fp8_quant, fuse_silu_mul_fp8_quant_ep
from deepgemm import m_grouped_fp8_gemm_nt_contiguous
from compressed_tensors.quantization import QuantizationStrategy

_is_hcu = is_hcu()
_use_deepgemm_moe = get_bool_env_var("SGLANG_USE_DEEPGEMM_MOE")
_use_fp8_w8a8_moe = get_bool_env_var("SGLANG_USE_FP8_W8A8_MOE")
_use_lightop_ep_moe_align = get_bool_env_var("SGLANG_USE_LIGHTOP_EP_MOE_ALIGN", "true")
_use_lightop_ep_scatter = get_bool_env_var("SGLANG_USE_LIGHTOP_EP_SCATTER", "true")
_use_lightop_ep_gather = get_bool_env_var("SGLANG_USE_LIGHTOP_EP_GATHER", "true")
def _get_deepgemm_shuffle_unique() -> tuple[int, str]:
    """Return the DeepGEMM INT8 unique-token policy for the PD topology."""

    try:
        mode = get_disagg().disaggregation_mode
        if mode in ("prefill", "decode"):
            return 0, mode
    except Exception:
        pass
    return 1, "ifb"


def get_deepgemm_marlin_weight_pack_fn(
    weight_dtype: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the DeepGEMM MoE weight packer for the P/D topology.

    Decode uses the masked layout; prefill and IFB use the contiguous layout.
    IFB shares one packed buffer between both GEMMs: INT8 uses
    ``shuffle_unique=1``, and FP8 packers produce the same layout per arch.
    """
    if not _is_hcu:
        raise RuntimeError("DeepGEMM marlin weight packing requires the HCU runtime")
    shuffle_unique, mode = _get_deepgemm_shuffle_unique()
    use_masked = mode == "decode"

    if weight_dtype == "fp8":
        from deepgemm import marlin_fp8_contiguous_weight, marlin_fp8_masked_weight

        return marlin_fp8_masked_weight if use_masked else marlin_fp8_contiguous_weight

    if weight_dtype == "int8":
        try:
            from deepgemm import (
                marlin_int8_contiguous_weight,
                marlin_int8_masked_weight,
            )
        except ImportError:
            # Older DeepGEMM builds expose only the marlin_i8_* names.
            from deepgemm import (
                marlin_i8_contiguous_weight as marlin_int8_contiguous_weight,
            )
            from deepgemm import marlin_i8_masked_weight as marlin_int8_masked_weight

        pack_fn = (
            marlin_int8_masked_weight if use_masked else marlin_int8_contiguous_weight
        )
        return functools.partial(pack_fn, shuffle_unique=shuffle_unique)

    raise ValueError(
        f"Unsupported DeepGEMM weight dtype {weight_dtype!r}; expected 'fp8' or 'int8'"
    )


def _use_hcu_legacy_channel_fp8_deepep(layer) -> bool:
    """Keep DSV4-style channel-FP8 experts on the DeepEPMoE HCU kernels.

    That path (LightOp ep_scatter/ep_gather, deepgemm grouped FP8 GEMMs and the
    fused SiLU(+swiglu_limit)/FP8 quant) is the validated DSV4.1 MoE path.
    MoeRunner(DEEP_GEMM) stays for what only it implements: the swiglu_oai
    activation (gemm1_alpha / gemm1_clamp_limit) and the local shared group.
    """
    scheme = getattr(layer, "scheme", None)
    config = getattr(layer, "moe_runner_config", None)
    return (
        _is_hcu
        and _use_deepgemm_moe
        and _use_fp8_w8a8_moe
        and type(scheme).__name__ == "CompressedTensorsW8A8Fp8MoE"
        and getattr(getattr(scheme, "weight_quant", None), "strategy", None)
        == QuantizationStrategy.CHANNEL
        and config is not None
        and config.activation == "silu"
        and config.gemm1_alpha is None
        and config.gemm1_clamp_limit is None
        and not getattr(config, "local_shared_experts_without_dispatch", False)
    )


def _can_use_lightop_ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: Optional[torch.Tensor],
    recv_topk: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    *,
    counts_are_aligned: bool,
    scale_ue8m0: bool = False,
) -> bool:
    # lightop's ep_scatter is a byte scatter: int8 and FP8 activations share
    # the same copy path as long as scales are per-token fp32.
    return (
        _use_lightop_ep_scatter
        and recv_x.element_size() == 1
        and output_tensor.element_size() == 1
        and recv_x_scale is not None
        and recv_x_scale.dtype == torch.float32
        and output_tensor_scale.dtype == torch.float32
        and (recv_x_scale.dim() == 1 or recv_x_scale.shape[-1] == 1)
        and (output_tensor_scale.dim() == 1 or output_tensor_scale.shape[-1] == 1)
        and recv_topk.dtype == torch.int64
        and counts_are_aligned
        and not scale_ue8m0
    )


def _can_use_lightop_ep_moe_align(
    topk_ids: torch.Tensor,
    device: torch.device,
    num_experts: int,
    total_elements: Optional[int],
    block_size: int,
    counts_are_aligned: bool,
) -> bool:
    return (
        _use_lightop_ep_moe_align
        and hasattr(lightop_op, "ep_build_m_indices")
        and counts_are_aligned
        and total_elements is not None
        and total_elements > 0
        and total_elements % block_size == 0
        and topk_ids.device == device
        and topk_ids.dtype == torch.int64
        and topk_ids.is_contiguous()
        and 0 < num_experts <= 1024
        and block_size > 0
    )


@torch.no_grad()
def _build_m_indices_with_optional_lightop(
    topk_ids: torch.Tensor,
    device: torch.device,
    num_experts: int,
    *,
    total_elements: Optional[int] = None,
    block_size: int = 256,
    counts_are_aligned: bool = False,
) -> torch.Tensor:
    if _can_use_lightop_ep_moe_align(
        topk_ids,
        device,
        num_experts,
        total_elements,
        block_size,
        counts_are_aligned,
    ):
        m_indices = torch.full((total_elements,), -1, device=device, dtype=torch.int32)
        lightop_op.ep_build_m_indices(topk_ids, m_indices, num_experts, block_size)
        return m_indices

    return build_m_indices_triton(topk_ids, device, num_experts)


def _can_use_lightop_ep_gather(
    input_tensor: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
) -> bool:
    return (
        _use_lightop_ep_gather
        and hasattr(lightop_op, "ep_gather")
        and input_tensor.dim() == 2
        and output_tensor.dim() == 2
        and input_tensor.shape[1] == output_tensor.shape[1]
        and input_tensor.dtype == output_tensor.dtype
        and input_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and topk_ids.dtype == torch.int64
        and topk_weights.dtype == torch.float32
        and input_index.dtype == torch.int32
        and topk_ids.shape == topk_weights.shape
        and topk_ids.shape == input_index.shape
        and topk_ids.shape[1] <= 16
        and input_tensor.is_contiguous()
        and output_tensor.is_contiguous()
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
        and input_index.is_contiguous()
    )


@torch.no_grad()
def _ep_gather_with_optional_lightop(
    input_tensor: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    if _can_use_lightop_ep_gather(
        input_tensor, topk_ids, topk_weights, input_index, output_tensor
    ):
        lightop_op.ep_gather(
            input_tensor, topk_ids, topk_weights, input_index, None, output_tensor
        )
        return

    ep_gather(input_tensor, topk_ids, topk_weights, input_index, output_tensor)


@torch.no_grad()
def _ep_scatter_with_optional_lightop(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    all_tokens: int,
    *,
    counts_are_aligned: bool,
    scale_ue8m0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_num_expert = num_recv_tokens_per_expert.shape[0]

    if _can_use_lightop_ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk,
        output_tensor,
        output_tensor_scale,
        counts_are_aligned=counts_are_aligned,
        scale_ue8m0=scale_ue8m0,
    ):
        output_index = torch.full(
            recv_topk.shape, -1, device=recv_topk.device, dtype=torch.int32
        )
        m_indices = torch.full(
            (all_tokens,), -1, device=recv_x.device, dtype=torch.int32
        )
        lightop_op.ep_scatter(
            recv_x,
            recv_x_scale,
            recv_topk,
            None,
            num_recv_tokens_per_expert,
            output_tensor,
            output_tensor_scale,
            m_indices,
            output_index,
            local_num_expert,
            256,
        )
        return m_indices, output_index

    output_index = torch.full(
        recv_topk.shape, -1, device=recv_topk.device, dtype=torch.int32
    )
    expert_start_loc = torch.zeros_like(num_recv_tokens_per_expert)
    m_indices = _build_m_indices_with_optional_lightop(
        recv_topk,
        recv_x.device,
        local_num_expert,
        total_elements=all_tokens if counts_are_aligned else None,
        block_size=256,
        counts_are_aligned=counts_are_aligned,
    )
    ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk,
        num_recv_tokens_per_expert,
        num_recv_tokens_per_expert,
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
        scale_ue8m0=scale_ue8m0,
    )
    return m_indices, output_index


@staticmethod
def _register_runtime_buffer(
    layer: torch.nn.Module, name: str, value: torch.Tensor
) -> None:
    if name in layer._buffers:
        layer._buffers[name] = value
        layer._non_persistent_buffers_set.add(name)
        return

    if hasattr(layer, name):
        delattr(layer, name)
    layer.register_buffer(name, value, persistent=False)


def _prepare_dsv4_channel_fp8_deepgemm_weights(
    self, layer: torch.nn.Module
) -> None:
    if getattr(layer, "_dsv4_channel_fp8_deepgemm_repacked", False):
        return

    w13 = layer.w13_weight
    w2 = layer.w2_weight

    pack_fn = get_deepgemm_marlin_weight_pack_fn("fp8")

    with torch.no_grad():
        w13_deepgemm = pack_fn(w13).detach()
        w2_deepgemm = pack_fn(w2).detach()

    self._register_runtime_buffer(layer, "w13_weight_deepgemm", w13_deepgemm)
    self._register_runtime_buffer(layer, "w2_weight_deepgemm", w2_deepgemm)
    layer._dsv4_channel_fp8_deepgemm_repacked = True

    # Save original weight shapes before deletion — needed by
    # DeepEPMoE.forward_groupgemm_w8a8_fp8_contiguous for N dimension
    layer._dsv4_w13_weight_shape = tuple(w13.shape)
    layer._dsv4_w2_weight_shape = tuple(w2.shape)

    # Clean up weights that won't be used
    del layer.w13_weight
    del layer.w2_weight
    torch.cuda.empty_cache()


def forward_groupgemm_w8a8_fp8_contiguous(
    self,
    dispatch_output: DeepEPNormalDispatchOutput,
):
    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    assert self.quant_method is not None
    assert self.moe_runner_config.activation == "silu"
    if num_recv_tokens_per_expert is None:
        return hidden_states.bfloat16()

    all_tokens = sum(num_recv_tokens_per_expert)
    if all_tokens <= 0:
        return hidden_states.bfloat16()

    M, K = hidden_states.size()
    w13_shape = getattr(self, "_dsv4_w13_weight_shape", None)
    if w13_shape is not None:
        N = w13_shape[1]
    else:
        N = self.w13_weight.size(1)
    w13_weight_fp8 = (
        self.w13_weight_deepgemm,
        # self.w13_weight,
        self.w13_weight_scale,
    )

    w2_weight_fp8 = (
        self.w2_weight_deepgemm,
        # self.w2_weight,
        self.w2_weight_scale,
    )

    hidden_states_shape = hidden_states.shape
    hidden_states_device = hidden_states.device
    input_tensor = [
        torch.empty(
            (all_tokens, K),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        ),
        (
            torch.empty(
                (all_tokens, hidden_states_scale.shape[-1]),
                device=hidden_states.device,
                dtype=torch.float32,
            )
        ),
    ]
    if get_offloader().forbid_copy_engine_usage:
        num_recv_tokens_per_expert_gpu = copy_list_to_gpu_no_ce(
            num_recv_tokens_per_expert
        )
    else:
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
    counts_are_aligned = all(
        count % 256 == 0 for count in num_recv_tokens_per_expert
    )
    m_indices, output_index = _ep_scatter_with_optional_lightop(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        num_recv_tokens_per_expert_gpu,
        input_tensor[0],
        input_tensor[1],
        all_tokens,
        counts_are_aligned=counts_are_aligned,
    )

    gateup_output_factory = torch.empty if counts_are_aligned else torch.zeros
    gateup_output = gateup_output_factory(
        (all_tokens, N),
        device=hidden_states_device,
        dtype=torch.bfloat16,
    )

    m_grouped_fp8_gemm_nt_contiguous(
        (input_tensor[0], input_tensor[1].squeeze(-1)),
        (w13_weight_fp8[0], w13_weight_fp8[1].squeeze(-1)),
        gateup_output,
        m_indices,
    )
    del input_tensor

    swiglu_limit = self.moe_runner_config.swiglu_limit
    if swiglu_limit is None:
        q_a2_all, q_a2_scale = fuse_silu_mul_fp8_quant(
            gateup_output,
            fp8type=0,
        )
    else:
        q_a2_all, q_a2_scale = fuse_silu_mul_fp8_quant(
            gateup_output,
            fp8type=0,
            limit=swiglu_limit,
        )
    del gateup_output

    down_output = torch.empty(
        (all_tokens, K),
        device=hidden_states_device,
        dtype=torch.bfloat16,
    )

    m_grouped_fp8_gemm_nt_contiguous(
        (q_a2_all, q_a2_scale.squeeze(-1)),
        (w2_weight_fp8[0], w2_weight_fp8[1].squeeze(-1)),
        down_output,
        m_indices,
    )

    gather_out = torch.empty(
        hidden_states_shape,
        device=hidden_states_device,
        dtype=torch.bfloat16,
    )

    _ep_gather_with_optional_lightop(
        down_output, topk_ids, topk_weights, output_index, gather_out
    )
    del down_output

    return gather_out


def forward_groupgemm_w8a8_fp8_masked(
    self,
    dispatch_output: DeepEPLLDispatchOutput,
):

    hidden_states, hidden_states_scale, topk_ids, _, masked_m, expected_m = (
        dispatch_output
    )
    # This HCU low-latency DeepGEMM path predates the MoeRunner refactor.
    # Its overlap state is therefore stored directly on DeepEPMoE by
    # FusedMoE.set_overlap_args(), and no ``self.runner`` is constructed.
    down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = getattr(
        self, "down_gemm_overlap_args", None
    )
    meta_overlap_args: Optional[dict] = getattr(self, "meta_overlap_args", None)
    assert self.moe_runner_config.activation == "silu"
    # base shapes
    num_groups, m, k = hidden_states.size()
    expected_m = min(m, expected_m)

    # ---- weights & scales ----
    # w13_weight = self.w13_weight
    w13_weight = self.w13_weight_deepgemm
    w13_scales = self.w13_weight_scale
    # w2_weight = self.w2_weight
    w2_weight = self.w2_weight_deepgemm
    w2_scales = self.w2_weight_scale

    n1 = w13_scales.size(1)
    gateup_output = torch.empty(
        (num_groups, m, n1), device=hidden_states.device, dtype=torch.bfloat16
    )

    from deepgemm import m_grouped_fp8_gemm_nt_masked

    m_grouped_fp8_gemm_nt_masked(
        (hidden_states, hidden_states_scale.contiguous()),
        (w13_weight, w13_scales),
        gateup_output,
        masked_m,
        expected_m,
    )

    q_a2_all, q_a2_scale = fuse_silu_mul_fp8_quant_ep(
        input=gateup_output, fp8type=0, tokens_per_expert=masked_m,
        limit=self.moe_runner_config.swiglu_limit,
    )
    # The first-stage BF16 activation is no longer needed after quantization.
    # Releasing it here lowers peak memory during low-latency graph capture.
    del gateup_output

    # ---- second GEMM ----
    n2 = w2_scales.size(1)
    down_output = torch.empty(
        (num_groups, m, n2), device=q_a2_all.device, dtype=torch.bfloat16
    )

    enable_overlap = down_gemm_overlap_args is not None
    if enable_overlap:
        raise RuntimeError(
            "DeepGEMM FP8 does not support down-GEMM completion "
            "signals; disable single-batch combine/down-GEMM overlap"
        )

    m_grouped_fp8_gemm_nt_masked(
        (q_a2_all, q_a2_scale.contiguous()),
        (w2_weight, w2_scales),
        down_output,
        masked_m,
        expected_m,
        enable_overlap=enable_overlap,
        signal=down_gemm_overlap_args.signal if enable_overlap else None,
    )

    if meta_overlap_args is not None:
        meta_overlap_args["block_m"] = 64
        meta_overlap_args["threshold"] = 32

    return down_output

def around_ep_init(original, layer, *args, **kwargs):
    result = original(layer, *args, **kwargs)
    if _is_hcu:
        import inspect
        from hcu_sglang.layers.moe.deepep_int8 import configure_layer
        bound = inspect.signature(original).bind(layer, *args, **kwargs)
        configure_layer(layer, bound.arguments.get("quant_config"))
    if _is_hcu and getattr(getattr(layer, "quant_method", None), "hcu_w4a8_hipc", False):
        from sglang.srt.layers.moe.utils import get_deepep_mode
        layer.deprecate_flag = False
        layer.deepep_mode = get_deepep_mode()
    if _use_hcu_legacy_channel_fp8_deepep(layer):
        layer.deprecate_flag = False
        layer.use_fp8_w8a8 = True
        layer.use_block_quant = False
        layer.use_w4afp8 = False
        layer.use_w4a8_marlin = False
        layer.use_w8a8_marlin = False
        layer.use_bf16_marlin = False
        layer.dispatcher.set_quant_config({**(getattr(layer.dispatcher, "quant_config", None) or {}), "hcu_legacy_groupgemm": True})
    return result


def around_process_weights(original, scheme, layer):
    result = original(scheme, layer)
    if _use_hcu_legacy_channel_fp8_deepep(layer):
        _prepare_dsv4_channel_fp8_deepgemm_weights(scheme, layer)
    return result


def around_run_moe_core(original, layer, dispatch_output, *args, **kwargs):
    if _is_hcu and getattr(getattr(layer, "quant_method", None), "hcu_w4a8_hipc", False):
        from hcu_sglang.layers.moe.ct_w4a8 import run_moe_core
        return run_moe_core(layer, dispatch_output)
    if _is_hcu and getattr(layer, "_hcu_slimquant_w4a8_hipc", False):
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            from hcu_sglang.layers.moe.deepep_w4a8 import run_masked_moe_core

            return run_masked_moe_core(layer, dispatch_output)
    if _is_hcu and getattr(layer, "_hcu_slimquant_w4a8_contiguous_hipc", False):
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker

        if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
            from hcu_sglang.layers.moe.deepep_w4a8 import run_contiguous_moe_core

            return run_contiguous_moe_core(layer, dispatch_output)
    if _is_hcu and getattr(layer, "use_int8_w8a8_deepgemm", False):
        from hcu_sglang.layers.moe.deepep_int8 import run_moe_core
        return run_moe_core(layer, dispatch_output)
    if not _use_hcu_legacy_channel_fp8_deepep(layer):
        return original(layer, dispatch_output, *args, **kwargs)
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
    if DispatchOutputChecker.format_is_deepep_normal(dispatch_output):
        output = forward_groupgemm_w8a8_fp8_contiguous(layer, dispatch_output)
        wrapper = D.DeepEPNormalCombineInput
    elif DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
        output = forward_groupgemm_w8a8_fp8_masked(layer, dispatch_output)
        wrapper = D.DeepEPLLCombineInput
    else:
        raise ValueError("HCU channel FP8 requires DeepEP normal or low-latency dispatch")
    return wrapper(hidden_states=output, topk_ids=dispatch_output.topk_ids, topk_weights=dispatch_output.topk_weights)


def register_hooks():
    if not _is_hcu:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import CompressedTensorsW8A8Fp8MoE
    CompressedTensorsW8A8Fp8MoE._register_runtime_buffer = _register_runtime_buffer
    HookRegistry.register("sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.__init__", around_ep_init, HookType.AROUND)
    HookRegistry.register("sglang.srt.layers.moe.ep_moe.layer.DeepEPMoE.run_moe_core", around_run_moe_core, HookType.AROUND)
