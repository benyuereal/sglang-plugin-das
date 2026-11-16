"""SGLang HCU general plugin - model and layer adaptations."""

import importlib
import logging

logger = logging.getLogger(__name__)

# (module path, entry point name, HCU-only). Rows execute top to bottom:
# later REPLACE hooks override earlier ones for the same target, and some
# registrations patch results of the ones above them, so order is behavioral.
_HOOK_TARGETS = (
    ("hcu_sglang.jit_overlay_loader", "register_hooks", False),
    ("hcu_sglang.configs", "register_hooks", False),
    ("hcu_sglang.configs_mamba", "register_hooks", False),
    ("hcu_sglang.server_args", "register", False),
    ("hcu_sglang.server_args_hooks", "register", False),
    ("hcu_sglang.speculative_compat", "register_hooks", False),
    ("hcu_sglang.multimodal_processor", "register_hooks", False),
    ("hcu_sglang.batch_overlap", "register_hooks", False),
    ("hcu_sglang.quick_all_reduce", "register_hooks", False),
    ("hcu_sglang.distributed", "register_hooks", False),
    ("hcu_sglang.disaggregation", "register_hooks", False),
    ("hcu_sglang.kernel_compat", "register_hooks", False),
    ("hcu_sglang.layers.attention.hcu_mla_registry", "register", False),
    ("hcu_sglang.layers.attention.base_backend", "register_hooks", False),
    ("hcu_sglang.layers.attention.flashattention_backend", "register", False),
    (
        "hcu_sglang.layers.attention.debug_flash_mla_adapter",
        "register_hooks",
        False,
    ),
    ("hcu_sglang.layers.attention.dsa_kernels", "register_hooks", False),
    ("hcu_sglang.layers.attention.dsa_topk", "register_hooks", False),
    ("hcu_sglang.layers.attention.current_dsa", "register_hooks", False),
    ("hcu_sglang.layers.attention.dsv4_compressor", "register_hooks", False),
    ("hcu_sglang.layers.attention.index_topk_share", "register_hooks", False),
    ("hcu_sglang.layers.attention.triton_hcu", "register_hooks", False),
    ("hcu_sglang.layers.attention.linear.gdn_backend", "register", False),
    ("hcu_sglang.layers.attention.linear.kda_backend", "register_hooks", False),
    ("hcu_sglang.layers.attention.fla.l2norm", "register", False),
    ("hcu_sglang.layers.moe.routing", "register_hooks", False),
    ("hcu_sglang.eplb.integration", "register_hooks", False),
    ("hcu_sglang.eplb.placement", "register_hooks", False),
    ("hcu_sglang.eplb.hook_dispatch", "register_hooks", False),
    ("hcu_sglang.eplb.recording", "register_hooks", False),
    ("hcu_sglang.deepep_graph", "register_hooks", False),
    ("hcu_sglang.graph_batch", "register_hooks", False),
    ("hcu_sglang.graph_setup", "register_hooks", False),
    ("hcu_sglang.cp_runtime", "register_hooks", False),
    ("hcu_sglang.dsa_utils", "register_hooks", False),
    ("hcu_sglang.forward_batch", "register_hooks", False),
    ("hcu_sglang.prefill_output", "register_hooks", False),
    ("hcu_sglang.speculative_vp", "register_hooks", False),
    ("hcu_sglang.prefill_graph_runner", "register_hooks", False),
    ("hcu_sglang.prefill_graph_vote", "register_hooks", False),
    ("hcu_sglang.prefill_graph_sync", "register_hooks", False),
    ("hcu_sglang.encoder_swa_replay", "register_hooks", False),
    ("hcu_sglang.encoder_swa_worker", "register_hooks", False),
    ("hcu_sglang.prefill_admission", "register_hooks", False),
    ("hcu_sglang.prefill_memory_budget", "register_hooks", False),
    ("hcu_sglang.prefill_request", "register_hooks", False),
    ("hcu_sglang.replay_validation", "register_hooks", False),
    ("hcu_sglang.model_replay_budget", "register_hooks", False),
    ("hcu_sglang.v4_defaults", "register_hooks", False),
    ("hcu_sglang.mem_cache.window_tree", "register_hooks", False),
    ("hcu_sglang.mem_cache.window_rust", "register_hooks", False),
    ("hcu_sglang.mem_cache.rust_tree_core.provider", "register_hooks", False),
    ("hcu_sglang.mem_cache.cpp_utils.native_hash", "register_hooks", False),
    ("hcu_sglang.mem_cache.window_pool_name", "register_hooks", False),
    ("hcu_sglang.mem_cache.window_pipeline", "register_hooks", False),
    ("hcu_sglang.mem_cache.window_cache_init", "register_hooks", False),
    ("hcu_sglang.mem_cache.file_evictor", "register_hooks", False),
    ("hcu_sglang.mem_cache.file_backend", "register_hooks", False),
    ("hcu_sglang.mem_cache.kv_events", "register_hooks", False),
    ("hcu_sglang.mem_cache.kv_event_admission", "register_hooks", False),
    ("hcu_sglang.layers.moe.standard_dispatcher", "register_hooks", False),
    ("hcu_sglang.layers.moe.alignment", "register_hooks", False),
    ("hcu_sglang.layers.moe.deepep_current", "register_hooks", True),
    ("hcu_sglang.layers.moe.deepep_fp8", "register_hooks", True),
    ("hcu_sglang.layers.moe.aiter_w4a8", "register_hooks", False),
    ("hcu_sglang.layers.moe.lightop_runner", "register_hooks", False),
    ("hcu_sglang.layers.moe.mega_moe", "register_hooks", False),
    ("hcu_sglang.layers.moe.dsv4_ops", "register_hooks", False),
    ("hcu_sglang.layers.moe.deepseek_gate", "register_hooks", False),
    ("hcu_sglang.layers.quantization", "register_hooks", False),
    ("hcu_sglang.layers.norm_activation", "register_hooks", False),
    ("hcu_sglang.layers.linear_fusion", "register_hooks", False),
    ("hcu_sglang.layers.layer_boundary", "register_hooks", False),
    ("hcu_sglang.layers.rotary", "register_hooks", False),
    ("hcu_sglang.mem_cache.allocation", "register_hooks", False),
    ("hcu_sglang.mem_cache.kv_cache_dtype", "register_hooks", False),
    ("hcu_sglang.mem_cache.swa", "register_hooks", False),
    ("hcu_sglang.mem_cache.hcu_host_pool", "register_hooks", False),
    ("hcu_sglang.mem_cache.hcu_mha_pool", "register_hooks", False),
    ("hcu_sglang.mem_cache.mla", "register_hooks", False),
    ("hcu_sglang.mem_cache.dsa", "register_hooks", False),
    ("hcu_sglang.mem_cache.dsv4", "register_hooks", False),
    ("hcu_sglang.models.bailing_moe", "register", False),
    ("hcu_sglang.models.dspark", "register_hooks", False),
    ("hcu_sglang.models.hunyuan", "register_hooks", False),
    ("hcu_sglang.models.hunyuan_v4", "register_hooks", False),
    ("hcu_sglang.models.glm4_moe", "register_hooks", False),
    ("hcu_sglang.models.minimax_m2", "register_hooks", False),
    ("hcu_sglang.multimodal_gen.platform", "register_hooks", False),
    ("hcu_sglang.multimodal_gen.config_compat", "register_hooks", False),
    ("hcu_sglang.multimodal_gen.profiling", "register_hooks", False),
    ("hcu_sglang.multimodal_gen.attention", "register_hooks", False),
    ("hcu_sglang.multimodal_gen.minimax_h3", "register_hooks", False),
    ("hcu_sglang.multimodal_gen.qknorm_rope", "register_hooks", False),
    ("hcu_sglang.multimodal_gen.wan_sla", "register_hooks", False),
    ("hcu_sglang.models.deepseek", "register_hooks", False),
    ("hcu_sglang.models.utils", "register_hooks", False),
    ("hcu_sglang.models.qwen.qwen2", "register", False),
    ("hcu_sglang.models.qwen.qwen3", "register", False),
    ("hcu_sglang.models.qwen.qwen3_moe", "register", False),
    ("hcu_sglang.speculative_dspark", "register_hooks", False),
    ("hcu_sglang.layers.gemm.hooks", "register", False),
    ("hcu_sglang.models.qwen3_5_mtp_lmhead_quant", "register", False),
)


def _hcu_enabled() -> bool:
    # Kept lazy so importing hcu_sglang does not pull in torch/HCU detection.
    from hcu_sglang.utils.detection import is_hcu

    return is_hcu()


def register() -> None:
    """Entry point for sglang.srt.plugins (called by load_plugins)."""
    # Outside the table: the AITER shims must exist before the hook modules
    # are imported, and env fields must be declared before modules read them.
    from hcu_sglang.compat.aiter import install_aiter_compat

    install_aiter_compat()

    from hcu_sglang.environ import register_envs

    register_envs()

    # Import-only: the legacy list imported moe.topk without invoking an
    # entry point. Keep the import; no registration is implied by it.
    importlib.import_module("hcu_sglang.layers.moe.topk")

    for module_path, entry, hcu_only in _HOOK_TARGETS:
        if hcu_only and not _hcu_enabled():
            continue
        module = importlib.import_module(module_path)
        getattr(module, entry)()

    logger.info("[hcu_sglang] optimization hooks registered")
