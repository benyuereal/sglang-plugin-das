"""HCU adaptations for SGLang multimodal generation."""


# anchor-baseline(②): main@5792cf6e loads diffusion hooks from a separate process-local registry.
def register() -> None:
    from hcu_sglang.compat.aiter import bootstrap_aiter_compat

    bootstrap_aiter_compat()
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

    from hcu_sglang.multimodal_gen import attention, minimax_h3, qknorm_rope

    rocm_targets = {
        "sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform.get_attn_backend_cls_str",
        "sglang.multimodal_gen.runtime.layers.attention.backends.aiter.AITerImpl.forward_varlen",
    }

    class DiffusionRegistry:
        @staticmethod
        def register(target, hook, hook_type):
            if target in rocm_targets and not current_platform.is_rocm():
                return
            existing = HookRegistry._hooks.get(target, ())
            if any(item[0] == hook_type and item[1] is hook for item in existing):
                return
            HookRegistry.register(target, hook, hook_type)

    attention.register_hooks(registry=DiffusionRegistry)
    minimax_h3.register_hooks(registry=DiffusionRegistry)
    qknorm_rope.register_hooks(registry=DiffusionRegistry)
