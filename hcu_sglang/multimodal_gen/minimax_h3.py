"""MiniMax-H3 HCU backend selection and QK-Norm compatibility."""

from __future__ import annotations

import logging
from typing import Any, Callable

from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)
_ROTARY_UNSET = object()


def after_adjust_attention_backend(result: Any, server_args: Any) -> Any:
    if not is_hcu() or server_args.attention_backend not in ("fa", "fa2"):
        return result

    from sglang.multimodal_gen.runtime.server_args.server_args import Backend

    if (
        server_args.backend == Backend.DIFFUSERS
        or type(server_args.pipeline_config).__name__ != "MiniMaxH3PipelineConfig"
    ):
        return result

    defaults = {"text_encoder": "torch_sdpa", "transformer": "fa2"}
    for component, backend in defaults.items():
        if component not in server_args.component_attention_backends:
            server_args.component_attention_backends[component] = backend
            logger.info(
                "Automatically set MiniMax-H3 HCU %s attention backend to %s",
                component,
                backend,
            )
    return result


def around_adjust_attention_backend(original, server_args):
    if is_hcu() and server_args.attention_backend in ("fa", "fa2"):
        from sglang.multimodal_gen.runtime.server_args.server_args import Backend
        if server_args.backend != Backend.DIFFUSERS and type(server_args.pipeline_config).__name__ == "MiniMaxH3PipelineConfig":
            components, roles = server_args._split_component_attention_backends(
                server_args.component_attention_backends,
                server_args.component_attention_backend_roles,
            )
            server_args.attention_backend = "fa2"
            if components.get("transformer") == "fa":
                components["transformer"] = "fa2"
            if "transformer" in roles:
                roles["transformer"] = {
                    role: "fa2" if backend == "fa" else backend
                    for role, backend in roles["transformer"].items()
                }
            server_args.component_attention_backends = components
            server_args.component_attention_backend_roles = roles
    result = original(server_args)
    return after_adjust_attention_backend(result, server_args)


def around_apply_qk_norm(
    original: Callable[..., Any], q, k, q_norm, k_norm, head_dim: int
):
    if not is_hcu():
        return original(q, k, q_norm, k_norm, head_dim)
    return q_norm(q), k_norm(k)


def after_post_load_weights(result: Any, model: Any) -> Any:
    if is_hcu():
        model._resolve_attention_backend_once()
    return result


# anchor-baseline(③): _apply_rope_qk — main@5792cf6e requires rotary_emb; the legacy fork uses five positional arguments.
def around_apply_rope_qk(
    original, q, k, cos_sin_cache, positions, *, rotary_emb=_ROTARY_UNSET
):
    if rotary_emb is not _ROTARY_UNSET:
        return original(q, k, cos_sin_cache, positions, rotary_emb=rotary_emb)
    if not is_hcu() or not q.is_cuda:
        return original(q, k, cos_sin_cache, positions)
    import torch
    rotary = getattr(torch.ops.sgl_kernel, "minimax_h3_rotary_embedding", None)
    if rotary is None:
        from sgl_kernel import minimax_h3_rotary_embedding as rotary
    rotary(positions, q.view(q.shape[0], -1), k.view(k.shape[0], -1),
           q.shape[-1], cos_sin_cache, True)
    return q, k


def around_vae_apply_rope_qk(original, query, key, rotary_pos_emb):
    if not is_hcu():
        return original(query, key, rotary_pos_emb)
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import native_rope_cache

    native = native_rope_cache(query, key, rotary_pos_emb)
    if native is None:
        return original(query, key, rotary_pos_emb)
    cache, positions = native
    if (query.shape[-1], cache.shape[1]) not in ((64, 48), (128, 96)):
        return original(query, key, rotary_pos_emb)
    query, key = query.contiguous(), key.contiguous()
    flat_query = query.view(positions.shape[0], query.shape[2], query.shape[3])
    flat_key = key.view(positions.shape[0], key.shape[2], key.shape[3])
    around_apply_rope_qk(None, flat_query, flat_key, cache, positions)
    return query, key


def register_hooks(registry=None) -> None:
    from hcu_sglang.multimodal_gen import minimax_h3_quant_loader, minimax_h3_modulation

    minimax_h3_quant_loader.register_hooks(registry)
    minimax_h3_modulation.register_hooks(registry)

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if registry is not None:
        HookRegistry = registry

    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils.apply_rotary_pos_emb_qk",
        around_vae_apply_rope_qk,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_rope_qk",
        around_apply_rope_qk,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.server_args.server_args."
        "ServerArgs._adjust_attention_backend",
        around_adjust_attention_backend,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_qk_norm",
        around_apply_qk_norm,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
        "MiniMaxH3DiTModel.post_load_weights",
        after_post_load_weights,
        HookType.AFTER,
    )
