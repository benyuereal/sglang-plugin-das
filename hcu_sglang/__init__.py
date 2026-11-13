"""SGLang HCU general plugin - model and layer adaptations."""

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point for sglang.srt.plugins (called by load_plugins)."""
    from hcu_sglang.compat.aiter import install_aiter_compat

    install_aiter_compat()

    from hcu_sglang.environ import register_envs

    register_envs()

    from hcu_sglang import (
        batch_overlap,
        configs,
        disaggregation,
        distributed,
        kernel_compat,
        multimodal_processor,
        quick_all_reduce,
        server_args,
        server_args_hooks,
        speculative_compat,
    )
    from hcu_sglang import jit_overlay_loader
    from hcu_sglang.layers.attention import (
        base_backend,
        debug_flash_mla_adapter,
        dsa_kernels,
        dsa_topk,
        current_dsa,
        dsv4_compressor,
        flashattention_backend,
        index_topk_share,
        triton_hcu,
    )
    from hcu_sglang.layers.attention import hcu_mla_registry
    from hcu_sglang.layers.attention.fla import l2norm
    from hcu_sglang.layers.attention.linear import gdn_backend
    from hcu_sglang.layers import norm_activation, quantization, linear_fusion, layer_boundary, rotary
    from hcu_sglang.layers.moe import alignment, aiter_w4a8, lightop_runner, mega_moe, topk, standard_dispatcher
    from hcu_sglang.mem_cache import allocation as mem_cache_allocation
    from hcu_sglang.mem_cache import (
        dsa,
        dsv4,
        hcu_host_pool,
        hcu_mha_pool,
        kv_cache_dtype,
        mla,
        swa,
    )
    from hcu_sglang.models import bailing_moe
    from hcu_sglang.models import dspark
    from hcu_sglang.models import hunyuan
    from hcu_sglang.models import hunyuan_v4 as hunyuan_v4_hooks
    from hcu_sglang.models import glm4_moe
    from hcu_sglang.models import minimax_m2
    from hcu_sglang.models import utils as model_utils
    from hcu_sglang.multimodal_gen import attention as multimodal_attention
    from hcu_sglang.multimodal_gen import config_compat as multimodal_config
    from hcu_sglang.multimodal_gen import minimax_h3
    from hcu_sglang.multimodal_gen import platform as multimodal_platform
    from hcu_sglang.multimodal_gen import profiling
    from hcu_sglang.multimodal_gen import qknorm_rope
    from hcu_sglang.multimodal_gen import wan_sla
    from hcu_sglang.models.deepseek import register_hooks as register_deepseek
    from hcu_sglang.models.qwen import qwen2, qwen3, qwen3_moe

    jit_overlay_loader.register_hooks()
    configs.register_hooks()
    from hcu_sglang import configs_mamba
    configs_mamba.register_hooks()
    server_args.register()
    server_args_hooks.register()
    speculative_compat.register_hooks()
    multimodal_processor.register_hooks()
    batch_overlap.register_hooks()
    quick_all_reduce.register_hooks()
    distributed.register_hooks()
    disaggregation.register_hooks()
    kernel_compat.register_hooks()
    hcu_mla_registry.register()
    base_backend.register_hooks()
    flashattention_backend.register()
    debug_flash_mla_adapter.register_hooks()
    dsa_kernels.register_hooks()
    dsa_topk.register_hooks()
    current_dsa.register_hooks()
    dsv4_compressor.register_hooks()
    index_topk_share.register_hooks()
    triton_hcu.register_hooks()
    gdn_backend.register()
    from hcu_sglang.layers.attention.linear import kda_backend
    kda_backend.register_hooks()
    l2norm.register()
    from hcu_sglang.layers.moe import routing
    routing.register_hooks()
    from hcu_sglang.eplb import integration as load_aware_integration
    load_aware_integration.register_hooks()
    from hcu_sglang.eplb import placement
    placement.register_hooks()
    from hcu_sglang.eplb import hook_dispatch
    hook_dispatch.register_hooks()
    from hcu_sglang.eplb import recording
    recording.register_hooks()
    from hcu_sglang import deepep_graph
    deepep_graph.register_hooks()
    from hcu_sglang import graph_batch
    graph_batch.register_hooks()
    from hcu_sglang import graph_setup
    graph_setup.register_hooks()
    from hcu_sglang import cp_runtime, forward_batch
    cp_runtime.register_hooks()
    from hcu_sglang import dsa_utils
    dsa_utils.register_hooks()
    forward_batch.register_hooks()
    from hcu_sglang import prefill_output, prefill_graph_runner
    prefill_output.register_hooks()
    from hcu_sglang import speculative_vp
    speculative_vp.register_hooks()
    prefill_graph_runner.register_hooks()
    from hcu_sglang import prefill_graph_vote
    prefill_graph_vote.register_hooks()
    from hcu_sglang import prefill_graph_sync
    prefill_graph_sync.register_hooks()
    from hcu_sglang import encoder_swa_replay, encoder_swa_worker
    encoder_swa_replay.register_hooks()
    encoder_swa_worker.register_hooks()
    from hcu_sglang import prefill_admission, prefill_memory_budget, prefill_request
    prefill_admission.register_hooks()
    prefill_memory_budget.register_hooks()
    prefill_request.register_hooks()
    from hcu_sglang import replay_validation, model_replay_budget
    replay_validation.register_hooks()
    model_replay_budget.register_hooks()
    from hcu_sglang import v4_defaults
    v4_defaults.register_hooks()
    from hcu_sglang.mem_cache import window_tree, window_rust
    window_tree.register_hooks()
    window_rust.register_hooks()
    from hcu_sglang.mem_cache.rust_tree_core import provider as rust_tree_provider
    rust_tree_provider.register_hooks()
    from hcu_sglang.mem_cache.cpp_utils import native_hash
    native_hash.register_hooks()
    from hcu_sglang.mem_cache import window_pool_name, window_pipeline, window_cache_init
    window_pool_name.register_hooks()
    window_pipeline.register_hooks()
    window_cache_init.register_hooks()
    from hcu_sglang.mem_cache import file_evictor, file_backend
    file_evictor.register_hooks()
    file_backend.register_hooks()
    from hcu_sglang.mem_cache import kv_events, kv_event_admission
    kv_events.register_hooks()
    kv_event_admission.register_hooks()
    standard_dispatcher.register_hooks()
    alignment.register_hooks()
    from hcu_sglang.utils.detection import is_hcu
    if is_hcu():
        from hcu_sglang.layers.moe import deepep_current, deepep_fp8
        deepep_current.register_hooks()
        deepep_fp8.register_hooks()
    aiter_w4a8.register_hooks()
    lightop_runner.register_hooks()
    mega_moe.register_hooks()
    from hcu_sglang.layers.moe import dsv4_ops
    dsv4_ops.register_hooks()
    from hcu_sglang.layers.moe import deepseek_gate
    deepseek_gate.register_hooks()
    quantization.register_hooks()
    norm_activation.register_hooks()
    linear_fusion.register_hooks()
    layer_boundary.register_hooks()
    rotary.register_hooks()
    mem_cache_allocation.register_hooks()
    kv_cache_dtype.register_hooks()
    swa.register_hooks()
    hcu_host_pool.register_hooks()
    hcu_mha_pool.register_hooks()
    mla.register_hooks()
    dsa.register_hooks()
    dsv4.register_hooks()
    bailing_moe.register()
    dspark.register_hooks()
    hunyuan.register_hooks()
    hunyuan_v4_hooks.register_hooks()
    glm4_moe.register_hooks()
    minimax_m2.register_hooks()
    multimodal_platform.register_hooks()
    multimodal_config.register_hooks()
    profiling.register_hooks()
    multimodal_attention.register_hooks()
    minimax_h3.register_hooks()
    qknorm_rope.register_hooks()
    wan_sla.register_hooks()
    register_deepseek()
    model_utils.register_hooks()
    qwen2.register()
    qwen3.register()
    qwen3_moe.register()
    from hcu_sglang import speculative_dspark
    speculative_dspark.register_hooks()
    from hcu_sglang.layers import hbird_hooks
    hbird_hooks.register()
    from hcu_sglang.models import qwen3_5_mtp_fp8
    qwen3_5_mtp_fp8.register()
    logger.info("[hcu_sglang] optimization hooks registered")
