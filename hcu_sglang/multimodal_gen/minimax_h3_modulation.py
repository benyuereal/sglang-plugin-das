from contextvars import ContextVar

from hcu_sglang.utils.detection import is_hcu

_ACTIVE = ContextVar("hcu_minimax_h3_fp32_modulation", default=False)
_LEGACY_NORM = ContextVar("hcu_minimax_h3_legacy_quant_norm", default=False)


def after_layer_init(result, layer, arch, quant_config, *args, **kwargs):
    enabled = False
    legacy_norm = False
    if is_hcu() and quant_config is not None:
        from hcu_sglang.multimodal_gen.minimax_h3_quant import MiniMaxH3QuantConfig
        legacy_norm = isinstance(quant_config, MiniMaxH3QuantConfig)
        enabled = (
            legacy_norm
            and arch.adaln_curve_grid is not None
            and quant_config.compressed_adaln
            and quant_config.fused
        )
    layer._hcu_fused_fp32_modulation = enabled
    layer._hcu_legacy_quant_norm = legacy_norm
    return result


def around_layer_forward(original, layer, *args, **kwargs):
    token = _ACTIVE.set(getattr(layer, "_hcu_fused_fp32_modulation", False))
    norm_token = _LEGACY_NORM.set(getattr(layer, "_hcu_legacy_quant_norm", False))
    try:
        return original(layer, *args, **kwargs)
    finally:
        _LEGACY_NORM.reset(norm_token)
        _ACTIVE.reset(token)


def around_scale_shift(original, x, shift, scale, indices, *, dtype):
    if _ACTIVE.get():
        from hcu_sglang.multimodal_gen.minimax_h3_quant import affine, supports_fp32_modulation
        if supports_fp32_modulation(x, (shift, scale), indices, dtype=dtype):
            return affine(x, shift, scale, indices)
    return original(x, shift, scale, indices, dtype=dtype)


def around_gate(original, x, gate, other, indices, *, dtype, allow_inplace=True):
    if _ACTIVE.get():
        from hcu_sglang.multimodal_gen.minimax_h3_quant import gate as fused_gate, supports_fp32_modulation
        if supports_fp32_modulation(x, (gate,), indices, dtype=dtype, other=other):
            return fused_gate(x, gate, other, indices)
    return original(x, gate, other, indices, dtype=dtype, allow_inplace=allow_inplace)


def around_norm_scale_shift(original, x, norm, shift, scale, indices, *, dtype, fused=False):
    if _LEGACY_NORM.get():
        from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import _modulate_scale_shift
        return _modulate_scale_shift(norm(x), shift, scale, indices, dtype=dtype)
    return original(x, norm, shift, scale, indices, dtype=dtype, fused=fused)


def register_hooks(registry=None):
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if registry is not None:
        HookRegistry = registry
    prefix = "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
    for name in ("MiniMaxH3DiTBlock", "MiniMaxH3FinalLayer"):
        HookRegistry.register(prefix + name + ".__init__", after_layer_init, HookType.AFTER)
        HookRegistry.register(prefix + name + ".forward", around_layer_forward, HookType.AROUND)
    for name, hook in (
        ("_modulate_scale_shift", around_scale_shift),
        ("_modulate_gate", around_gate),
        ("_modulate_rmsnorm_scale_shift", around_norm_scale_shift),
    ):
        HookRegistry.register(prefix + name, hook, HookType.AROUND)
