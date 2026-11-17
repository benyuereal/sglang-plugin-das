"""Method-level HCU adaptations for ``sglang.srt.server_args``."""

from __future__ import annotations

import logging
import os
from sglang.srt.arg_groups.arg_utils import is_record
from copy import deepcopy
from hcu_sglang.config import (configure_minimax_opt, minimax_opt_enabled,
    configure_cp_cache_layer_split, cp_cache_layer_split_enabled,
    configure_custom_all_reduce_backend, CUSTOM_ALL_REDUCE_BACKENDS)
from typing import Any, Callable

from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.config import hy3_sp_enabled, configure_hy3_sp

logger = logging.getLogger(__name__)

_DEEPSEEK_V4_HIP_OVERRIDES = (
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE",
    "SGLANG_OPT_USE_TILELANG_MHC_POST",
)

_EXTENSION_DEFAULTS = {
    "minimax_opt": False,
}


def _model_arch(server_args: Any) -> str | None:
    from sglang.srt.arg_groups.model_override_base import model_config_of
    config = (server_args.get_model_config() if hasattr(server_args, "get_model_config")
              else model_config_of(server_args))
    architectures = getattr(config.hf_config, "architectures", None)
    return architectures[0] if architectures else None


def _snapshot_env_fields(envs: Any, names: tuple[str, ...]) -> dict[str, str | None]:
    return {name: os.environ.get(getattr(envs, name).name) for name in names}


def _restore_env_fields(envs: Any, snapshot: dict[str, str | None]) -> None:
    for name, value in snapshot.items():
        field = getattr(envs, name)
        if value is None:
            field.clear()
        else:
            os.environ[field.name] = value


def handle_model_specific_adjustments(
    original: Callable[..., Any], server_args: Any
) -> Any:
    if not is_hcu() or _model_arch(server_args) != "DeepseekV4ForCausalLM":
        return original(server_args)

    from sglang.srt.environ import envs

    snapshot = _snapshot_env_fields(envs, _DEEPSEEK_V4_HIP_OVERRIDES)
    result = original(server_args)
    _restore_env_fields(envs, snapshot)
    envs.SGLANG_OPT_USE_FUSED_COMPRESS.set(True)
    return result


def after_deepseek_family_overrides(
    result: dict[str, Any], server_args: Any, hf_config: Any
) -> dict[str, Any]:
    if not is_hcu():
        return result
    from sglang.srt.configs.model_config import is_deepseek_dsa

    if is_deepseek_dsa(hf_config):
        result["page_size"] = 64
    return result


def after_dsa_split_backend_resolution(result: dict[str, Any], view: Any):
    if not is_hcu() or view.enable_hisparse:
        return result
    if view.dsa_prefill_backend is not None or view.dsa_decode_backend is not None:
        return result
    if view.kv_cache_dtype == "fp8_e4m3":
        result["dsa_prefill_backend"] = "flashmla_auto"
        result["dsa_decode_backend"] = "flashmla_kv"
    else:
        result["dsa_prefill_backend"] = "flashmla_sparse"
        result["dsa_decode_backend"] = "flashmla_sparse"
    return result


def after_a2a_backend_overrides(result: dict[str, Any], view: Any):
    from sglang.srt.environ import envs

    if envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get():
        return {"moe_a2a_backend": "megamoe"}
    return result


def around_validate_deepseek_v4_mega_moe_token_budget(
    original: Callable[..., Any], server_args: Any
):
    from sglang.srt.environ import envs

    if not envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get():
        return original(server_args)
    previous = server_args.moe_a2a_backend
    server_args.moe_a2a_backend = "megamoe"
    try:
        return original(server_args)
    finally:
        server_args.moe_a2a_backend = previous


def _declare_values(server_args, source, **values):
    if hasattr(server_args, "_declare"):
        server_args._declare(source, **values)
    else:
        from sglang.srt.arg_groups.overrides import declare_resolution
        declare_resolution(server_args, source, **values)


def _resolving_view(server_args):
    if hasattr(server_args, "_resolved"):
        return server_args._resolved()
    from sglang.srt.arg_groups.model_override_base import resolving_view
    return resolving_view(server_args)


def handle_a2a_moe(result: Any, server_args: Any) -> Any:
    if not is_hcu():
        return result

    from sglang.srt.environ import envs
    from sglang.srt.model_executor.cuda_graph_config import Backend

    view = _resolving_view(server_args)
    if (
        view.moe_a2a_backend == "megamoe"
        and envs.SGLANG_HCU_MEGA_MOE_RUNTIME.get().strip().lower() == "megamoe"
        and view.quantization == "w8a8_int8"
    ):
        graph = getattr(view, "cuda_graph_config", None)
        if graph is None:
            graph = server_args.cuda_graph_config
        graph = deepcopy(graph)
        graph.decode.backend = Backend.DISABLED
        graph.prefill.backend = Backend.DISABLED
        _declare_values(server_args, "hcu_sglang:int8_megamoe",
                        disable_shared_experts_fusion=True, cuda_graph_config=graph)
        logger.info(
            "HCU INT8 MegaMoE disables shared-expert fusion and CUDA graph "
            "because the standalone INT8 normal backend is eager-only."
        )

    if (
        view.moe_a2a_backend == "megamoe"
        and not envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.is_set()
    ):
        envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.set(True)

    return result


def enforce_hcu_mla_page_size(result: Any, server_args: Any) -> Any:
    view = _resolving_view(server_args)
    if (
        view.attention_backend == "hcu_mla"
        or view.decode_attention_backend == "hcu_mla"
        or view.prefill_attention_backend == "hcu_mla"
    ) and view.page_size != 64:
        logger.warning(
            "HCU MLA only supports a page_size of 64; changing page_size from %s to 64.",
            view.page_size,
        )
        if hasattr(server_args, "_declare"):
            server_args._declare("hcu_sglang:hcu_mla", page_size=64)
        else:
            from sglang.srt.arg_groups.overrides import declare_resolution
            declare_resolution(server_args, "hcu_sglang:hcu_mla", page_size=64)
    return result


def after_add_cli_args(result: Any, parser: Any) -> Any:
    existing = parser._option_string_actions
    # The user's serving recipe uses the pre-split decode graph flag.
    # An exact alias also prevents argparse's ambiguous-prefix rejection.
    if "--cuda-graph-max-bs" not in existing and "--cuda-graph-max-bs-decode" in existing:
        import argparse
        canonical = existing["--cuda-graph-max-bs-decode"]
        parser.add_argument("--cuda-graph-max-bs", dest=canonical.dest,
                            type=canonical.type, default=argparse.SUPPRESS,
                            help="Compatibility alias for --cuda-graph-max-bs-decode.")
    for alias, canonical in (("--nsa-prefill-backend", "--dsa-prefill-backend"), ("--nsa-decode-backend", "--dsa-decode-backend")):
        if alias not in existing and canonical in existing:
            import argparse
            from sglang.srt.arg_groups.argparse_actions import DeprecatedAliasStoreAction
            parser.add_argument(
                alias, dest=existing[canonical].dest, action=DeprecatedAliasStoreAction,
                new_flag=canonical, default=argparse.SUPPRESS, type=str,
                choices=["flashmla_sparse", "flashmla_sparse_q8", "flashmla_kv", "flashmla_auto", "flashinfer_sparse_mla", "fa3", "tilelang", "aiter", "trtllm"],
                help=f"[Deprecated] Use {canonical} instead.",
            )
    if "--custom-all-reduce-backend" not in existing:
        parser.add_argument(
            "--custom-all-reduce-backend", choices=CUSTOM_ALL_REDUCE_BACKENDS,
            default="auto",
            help="Custom all-reduce backend: auto, native, aiter or off; disable-custom-all-reduce takes precedence.",
        )
    if "--minimax-opt" not in existing:
        parser.add_argument(
            "--minimax-opt",
            action="store_true",
            default=None,
            help="Enable MiniMax M2 sequence-parallel prefill optimization.",
        )
    if "--hy3-sp" not in existing:
        parser.add_argument("--hy3-sp", action="store_true", default=None,
                            help="Enable Hunyuan V3 sequence parallelism over TP ranks.")
    if "--enable-cp-cache-layer-split" not in existing:
        group=parser.add_mutually_exclusive_group()
        group.add_argument("--enable-cp-cache-layer-split", "--hcu-cp-cache-layer-split",
            dest="hcu_plugin_cp_cache_layer_split", action="store_true", default=None,
            help="Enable HCU DeepSeek V4.1 prefill KV layer splitting across CP ranks.")
        group.add_argument("--disable-cp-cache-layer-split",
            dest="hcu_plugin_cp_cache_layer_split", action="store_false", default=None,
            help="Disable HCU CP KV layer splitting for this process and its workers.")
    return result


def before_add_cli_args(*args, **kwargs):
    from sglang.srt import server_args
    from hcu_sglang.server_args import ensure_choices

    ensure_choices(server_args)


def before_from_cli_args(cls, args):
    configure_custom_all_reduce_backend(getattr(args, "custom_all_reduce_backend", None))
    configure_minimax_opt(getattr(args, "minimax_opt", None))
    configure_hy3_sp(getattr(args, "hy3_sp", None))
    configure_cp_cache_layer_split(getattr(args, "hcu_plugin_cp_cache_layer_split", None))


def after_from_cli_args(result: Any, cls: Any, args: Any) -> Any:
    configure_custom_all_reduce_backend(getattr(args, "custom_all_reduce_backend", None))
    configure_minimax_opt(getattr(args, "minimax_opt", None))
    configure_hy3_sp(getattr(args, "hy3_sp", None))
    configure_cp_cache_layer_split(getattr(args, "hcu_plugin_cp_cache_layer_split", None))
    if not is_record(result):
        result.minimax_opt = minimax_opt_enabled()
        result.hy3_sp = hy3_sp_enabled()
    return result


_RESOLUTION_REGISTERED = False


def _hcu_model_defaults(server_args, hf_config):
    return after_deepseek_family_overrides({}, server_args, hf_config)


def _hcu_post_process(view):
    from sglang.srt.arg_groups.model_override_base import model_config_of
    from sglang.srt.configs.model_config import is_deepseek_dsa
    changes = {}
    if not is_hcu():
        if cp_cache_layer_split_enabled():
            raise ValueError("--enable-cp-cache-layer-split requires the HCU plugin platform")
        return changes
    from hcu_sglang.speculative_vp.config import validate as validate_draft_vp
    validate_draft_vp(view)
    if cp_cache_layer_split_enabled():
        from hcu_sglang.mem_cache.cp_cache_layer_split.validation import validate_cp_cache_layer_split
        validate_cp_cache_layer_split(view, model_config_of(view).hf_config)
    if is_deepseek_dsa(model_config_of(view).hf_config):
        changes.update(after_dsa_split_backend_resolution({}, view))
    changes.update(after_a2a_backend_overrides({}, view))
    return changes


def _model_resolution(server_args, previous):
    if is_hcu() and _resolving_view(server_args).enable_dsa_cache_layer_split:
        from hcu_sglang.layer_split_pipeline import adapt_model_adjustments

        previous = adapt_model_adjustments(previous)
    return handle_model_specific_adjustments(previous, server_args)


def _attention_resolution(server_args, previous):
    result = previous(server_args)
    return enforce_hcu_mla_page_size(result, server_args)


def _moe_resolution(server_args, previous):
    result = previous(server_args)
    return handle_a2a_moe(result, server_args)


def register() -> None:
    """Use official named resolution seams instead of removed ServerArgs methods."""
    global _RESOLUTION_REGISTERED
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook
    from sglang.srt.arg_groups.model_override_base import register_model_override_predicate
    from sglang.srt.arg_groups.overrides import register_post_process

    for target, hook, kind in (
        ("add_cli_args", before_add_cli_args, HookType.BEFORE),
        ("add_cli_args", after_add_cli_args, HookType.AFTER),
        ("from_cli_args", before_from_cli_args, HookType.BEFORE),
        ("from_cli_args", after_from_cli_args, HookType.AFTER),
    ):
        HookRegistry.register(f"sglang.srt.server_args.ServerArgs.{target}", hook, kind)
    if not _RESOLUTION_REGISTERED:
        for name, wrapper in (
            ("handle_model_specific_adjustments", _model_resolution),
            ("handle_attention_backend_compatibility", _attention_resolution),
            ("handle_a2a_moe", _moe_resolution),
        ):
            register_resolution_hook(name)(wrapper)
        register_model_override_predicate(lambda _architecture: True)(_hcu_model_defaults)
        register_post_process(_hcu_post_process)
        _RESOLUTION_REGISTERED = True
