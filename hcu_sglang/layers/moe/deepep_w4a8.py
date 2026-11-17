# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)


def _hcu_deepep_dispatcher(layer):
    """Return the HCU DeepEPDispatcher that actually backs `layer`.

    FusedMoE gets `layer.dispatcher` from create_moe_dispatcher, which returns
    MaybeTboDeepEPDispatcher: a wrapper holding one DeepEPDispatcher, or two
    when two-batch overlap is on. The plugin REPLACEs the DeepEPDispatcher
    class at registration time, so every inner is an HCU dispatcher. Only the
    inners know the resolved deepep_mode.
    """
    from hcu_sglang.layers.moe import deepep_current

    dispatcher = layer.dispatcher
    if isinstance(dispatcher, deepep_current.DeepEPDispatcher):
        return dispatcher
    inners = getattr(dispatcher, "_inners", None) or []
    if not inners or not all(
        isinstance(inner, deepep_current.DeepEPDispatcher) for inner in inners
    ):
        raise RuntimeError(
            "SlimQuant W4A8 HIPC requires the HCU DeepEP dispatcher, got "
            f"{type(dispatcher).__name__}"
        )
    return inners[0]


def configure_hipc(layer, *, masked, contiguous):
    from hcu_sglang.layers.moe import deepep_current

    dispatcher = _hcu_deepep_dispatcher(layer)
    mode = dispatcher.deepep_mode
    # The caller already packed the weights into the HIPC layout, so silently
    # dropping a requested mode would run the HIPC kernel against Marlin-packed
    # weights. Fail loudly instead.
    if masked and not mode.enable_low_latency():
        raise RuntimeError(
            "SGLANG_USE_W4A8_MASKED_HIPC requires a low-latency DeepEP mode, "
            f"got deepep_mode={mode.value}"
        )
    if contiguous and not mode.enable_normal():
        raise RuntimeError(
            "SGLANG_USE_W4A8_CONTIGUOUS_HIPC requires a normal DeepEP mode, "
            f"got deepep_mode={mode.value}"
        )
    if not (masked or contiguous):
        return
    if not deepep_current.use_groupgemm:
        raise RuntimeError("SlimQuant W4A8 HIPC requires SGLANG_GROUPGEMM")
    if masked and not deepep_current._DEEPEP_LL_DISPATCH_USES_QUANT_TYPE:
        raise RuntimeError(
            "SlimQuant W4A8 masked HIPC requires DeepEP INT8 dispatch, but "
            + (deepep_current.deepep_ll_dispatch_abi_error() or "the quant_type ABI is unavailable")
        )
    if layer.moe_runner_config.activation != "silu":
        raise RuntimeError("SlimQuant W4A8 HIPC currently requires SiLU")

    import deepgemm

    quant_config = dict(dispatcher.quant_config or {})
    for enabled, name in ((masked, "masked"), (contiguous, "contiguous")):
        if enabled:
            op = f"m_grouped_w4a8_gemm_nt_{name}_hipc"
            if not callable(getattr(deepgemm, op, None)):
                raise RuntimeError(f"SlimQuant W4A8 {name} HIPC requires deepgemm.{op}")
            quant_config[f"hcu_slimquant_w4a8_{name}_hipc"] = True
    # Push through layer.dispatcher, not the inner: the TBO wrapper fans the
    # config out to every inner dispatcher (there are two when TBO is on).
    layer.dispatcher.set_quant_config(quant_config)
    layer._hcu_slimquant_w4a8_hipc = masked
    layer._hcu_slimquant_w4a8_contiguous_hipc = contiguous


def _fuse_silu_mul_clamp_quant(gateup, limit, **kwargs):
    try:
        from lightop.activation import fuse_silu_mul_clamp_quant
    except ImportError:
        from lightop import fuse_silu_mul_clamp_quant

    return fuse_silu_mul_clamp_quant(gateup, float(limit), **kwargs)


def run_masked_moe_core(layer, dispatch):
    from deepgemm import m_grouped_w4a8_gemm_nt_masked_hipc
    from lightop.activation import fuse_silu_mul_quant_ep

    x, scales, topk_ids, topk_weights, masked_m, expected_m = dispatch
    if x.dtype != torch.int8 or scales is None or scales.dtype != torch.float32:
        raise RuntimeError("SlimQuant W4A8 masked HIPC requires INT8 activations and FP32 scales")

    num_experts, capacity, _ = x.shape
    expected_m = min(capacity, expected_m)
    gateup = x.new_empty(
        (num_experts, capacity, layer.w13_weight_scale.size(1)),
        dtype=torch.bfloat16,
    )
    m_grouped_w4a8_gemm_nt_masked_hipc(
        (x, scales),
        (layer.w13_weight, layer.w13_weight_scale),
        gateup,
        masked_m,
        expected_m,
    )

    limit = layer.moe_runner_config.swiglu_limit
    if limit is None:
        quantized = fuse_silu_mul_quant_ep(gateup, masked_m)
    else:
        quantized = _fuse_silu_mul_clamp_quant(
            gateup, limit, mask_m=masked_m, expect_m=expected_m
        )
    del gateup

    output = x.new_empty(
        (num_experts, capacity, layer.w2_weight_scale.size(1)),
        dtype=torch.bfloat16,
    )
    m_grouped_w4a8_gemm_nt_masked_hipc(
        quantized,
        (layer.w2_weight, layer.w2_weight_scale),
        output,
        masked_m,
        expected_m,
    )
    return DeepEPLLCombineInput(output, topk_ids, topk_weights)


def run_contiguous_moe_core(layer, dispatch):
    from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc
    from lightop.activation import fuse_silu_mul_quant
    from hcu_sglang.layers.moe.deepep_fp8 import (
        _ep_gather_with_optional_lightop,
        _ep_scatter_with_optional_lightop,
        copy_list_to_gpu_no_ce,
        get_offloader,
    )

    x, scales, topk_ids, topk_weights, counts = dispatch
    if counts and any(count < 0 or count % 256 for count in counts):
        raise RuntimeError(
            f"SlimQuant W4A8 contiguous HIPC requires 256-aligned DeepEP expert counts, got {counts}"
        )
    total_rows = sum(counts) if counts else 0
    if total_rows == 0:
        return DeepEPNormalCombineInput(x.bfloat16(), topk_ids, topk_weights)
    if x.dtype != torch.int8 or scales is None or scales.dtype != torch.float32:
        raise RuntimeError("SlimQuant W4A8 contiguous HIPC requires INT8 activations and FP32 scales")

    if get_offloader().forbid_copy_engine_usage:
        counts_device = copy_list_to_gpu_no_ce(counts)
    else:
        counts_device = torch.tensor(counts, dtype=torch.int32, pin_memory=True).to(
            x.device, non_blocking=True
        )

    grouped_x = x.new_empty((total_rows, x.shape[1]))
    grouped_scales = scales.new_empty((total_rows, 1))
    m_indices, output_index = _ep_scatter_with_optional_lightop(
        x, scales, topk_ids, counts_device, grouped_x, grouped_scales, total_rows,
        counts_are_aligned=True,
    )
    gateup = x.new_empty(
        (total_rows, layer.w13_weight_scale.size(1)), dtype=torch.bfloat16
    )
    m_grouped_w4a8_gemm_nt_contiguous_hipc(
        (grouped_x, grouped_scales),
        (layer.w13_weight, layer.w13_weight_scale),
        gateup,
        m_indices,
    )
    del grouped_x, grouped_scales

    limit = layer.moe_runner_config.swiglu_limit
    quantized = (
        fuse_silu_mul_quant(gateup)
        if limit is None
        else _fuse_silu_mul_clamp_quant(gateup, limit)
    )
    del gateup

    down = x.new_empty((total_rows, x.shape[1]), dtype=torch.bfloat16)
    m_grouped_w4a8_gemm_nt_contiguous_hipc(
        quantized, (layer.w2_weight, layer.w2_weight_scale), down, m_indices
    )
    result = x.new_empty(x.shape, dtype=torch.bfloat16)
    _ep_gather_with_optional_lightop(down, topk_ids, topk_weights, output_index, result)
    return DeepEPNormalCombineInput(result, topk_ids, topk_weights)
