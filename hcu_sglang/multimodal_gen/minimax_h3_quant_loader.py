from hcu_sglang.utils.detection import is_hcu


def around_inspect_safetensors(original, safetensors_list):
    if not is_hcu():
        return original(safetensors_list)
    from hcu_sglang.multimodal_gen.minimax_h3_quant import inspect_int8_checkpoint, inspect_fp8_checkpoint
    manifest = inspect_int8_checkpoint(safetensors_list)
    if manifest is None:
        manifest = inspect_fp8_checkpoint(safetensors_list)
    if manifest is None:
        return original(safetensors_list)
    curve = (1025, 8) if manifest.get("profile", "backbone") == "backbone-adaln" else None
    return curve, {}


def around_resolve_quant_load_spec(original, **kwargs):
    if not is_hcu() or kwargs["model_cls"].__name__ != "MiniMaxH3DiTModel":
        return original(**kwargs)
    args = kwargs["server_args"]
    if kwargs.get("gguf_file") is not None or args.transformer_weights_path is None:
        return original(**kwargs)
    from hcu_sglang.multimodal_gen.minimax_h3_quant import inspect_quant_checkpoint
    info = inspect_quant_checkpoint(
        kwargs["safetensors_list"], str(args.model_variant or "fl2va")
    )
    if info is None:
        return original(**kwargs)
    config, diffusers_layout, curve = info
    if kwargs.get("checkpoint_quant_config") is not None:
        raise ValueError("MiniMax-H3 quantization metadata conflicts with another format")
    if args.should_use_fsdp_for_component(kwargs.get("component_name")):
        raise ValueError("MiniMax-H3 INT8/FP8 supports TP/SP, not FSDP inference")
    arch = kwargs["arch_config"]
    if curve is not None and (
        args.minimax_h3_adaln_cache_path is not None or args.minimax_h3_adaln_online
    ):
        raise ValueError("MiniMax-H3 compressed AdaLN cannot use a separate AdaLN cache")
    updated = dict(kwargs, checkpoint_quant_config=config)
    result = original(**updated)
    arch.checkpoint_uses_diffusers_layout = diffusers_layout
    if curve is not None:
        arch.adaln_curve_grid, arch.time_embed_dim = curve
    return result


def after_attention_init(result, attention, *args, **kwargs):
    if is_hcu():
        attention._use_fused_qknorm_rope = False
    return result


def register_hooks(registry=None):
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if registry is not None:
        HookRegistry = registry
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader."
        "inspect_minimax_h3_safetensors",
        around_inspect_safetensors,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader."
        "resolve_transformer_quant_load_spec",
        around_resolve_quant_load_spec,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
        "MiniMaxH3Attention.__init__",
        after_attention_init,
        HookType.AFTER,
    )
