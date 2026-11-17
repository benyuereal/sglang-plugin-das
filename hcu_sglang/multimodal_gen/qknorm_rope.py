"""HCU compatibility for the diffusion QK-Norm + RoPE JIT kernel."""

from __future__ import annotations

from typing import Any, Callable

from hcu_sglang.utils.detection import is_hcu


def around_can_use_fused_inplace_qknorm_rope(
    original: Callable[..., bool], *args: Any, **kwargs: Any
) -> bool:
    if is_hcu():
        return False
    return original(*args, **kwargs)


def register_hooks(registry=None) -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    if registry is not None:
        HookRegistry = registry

    HookRegistry.register(
        "sglang.kernels.ops.diffusion.rope.qknorm_rope_jit."
        "can_use_fused_inplace_qknorm_rope",
        around_can_use_fused_inplace_qknorm_rope,
        HookType.AROUND,
    )
