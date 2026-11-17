"""Lazy registration for plugin-owned quantization configurations."""

from __future__ import annotations

import sys
from functools import cache
from typing import Any, Callable


QUANTIZATION_NAME = "w4a8_int8"
SLIMQUANT_NAME = "slimquant_w4a8"
SLIMQUANT_MARLIN_NAME = "slimquant_w4a8_marlin"
_HCU_QUANTIZATION_NAMES = {
    QUANTIZATION_NAME,
    SLIMQUANT_NAME,
    SLIMQUANT_MARLIN_NAME,
    "slimquant_marlin",
}


@cache
def _load_config_class():
    from hcu_sglang.layers.quantization.slimquant_w4a8_marlin import (
        SlimQuantW4A8Int8MarlinConfig,
    )

    return SlimQuantW4A8Int8MarlinConfig


@cache
def _load_base_config_class():
    from hcu_sglang.layers.quantization.slimquant_w4a8 import (
        SlimQuantW4A8Int8Config,
    )

    return SlimQuantW4A8Int8Config


class _LazyConfigMeta(type):
    def __call__(cls, *args, **kwargs):
        return _load_config_class()(*args, **kwargs)

    def __getattr__(cls, name):
        return getattr(_load_config_class(), name)


class LazySlimQuantW4A8Int8MarlinConfig(metaclass=_LazyConfigMeta):
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant):
        if hf_quant_cfg.get("quant_method") == "slimquant_w4a8" and user_quant in (
            QUANTIZATION_NAME,
            "slimquant_w4a8_marlin",
            "slimquant_marlin",
        ):
            return SLIMQUANT_MARLIN_NAME
        if (
            hf_quant_cfg.get("quant_method") == "compressed-tensors"
            and user_quant == SLIMQUANT_MARLIN_NAME
        ):
            return _load_config_class().override_quantization_method(
                hf_quant_cfg, user_quant
            )
        return None


class _LazyBaseConfigMeta(type):
    def __call__(cls, *args, **kwargs):
        return _load_base_config_class()(*args, **kwargs)

    def __getattr__(cls, name):
        return getattr(_load_base_config_class(), name)


class LazySlimQuantW4A8Int8Config(metaclass=_LazyBaseConfigMeta):
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant):
        del hf_quant_cfg, user_quant
        return None



@cache
def _load_compressed_config_class():
    from hcu_sglang.layers.quantization.compressed_marlin import SlimQuantCompressedTensorsMarlinConfig
    return SlimQuantCompressedTensorsMarlinConfig


class _LazyCompressedConfigMeta(type):
    def __call__(cls, *args, **kwargs):
        return _load_compressed_config_class()(*args, **kwargs)

    def __getattr__(cls, name):
        return getattr(_load_compressed_config_class(), name)


class LazySlimQuantCompressedTensorsMarlinConfig(metaclass=_LazyCompressedConfigMeta):
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant):
        if hf_quant_cfg.get("quant_method") == "compressed-tensors" and user_quant == "slimquant_marlin":
            return "slimquant_marlin"
        return None


def install_quantization_method() -> bool:
    quantization = sys.modules.get("sglang.srt.layers.quantization")
    if quantization is None:
        return False

    methods = {
        QUANTIZATION_NAME: LazySlimQuantW4A8Int8MarlinConfig,
        SLIMQUANT_NAME: LazySlimQuantW4A8Int8Config,
        SLIMQUANT_MARLIN_NAME: LazySlimQuantW4A8Int8MarlinConfig,
        "slimquant_marlin": LazySlimQuantCompressedTensorsMarlinConfig,
    }
    for name, config in methods.items():
        quantization.BASE_QUANTIZATION_METHODS.setdefault(name, config)
        quantization.QUANTIZATION_METHODS.setdefault(name, config)

    quark_config = getattr(quantization, "QuarkConfig", None)
    if quark_config is not None:
        quantization.BASE_QUANTIZATION_METHODS.setdefault("quark", quark_config)
        quantization.BASE_QUANTIZATION_METHODS.setdefault("quark_mxfp4", quark_config)
        quantization.QUANTIZATION_METHODS.setdefault("quark", quark_config)
        quantization.QUANTIZATION_METHODS.setdefault("quark_mxfp4", quark_config)

    model_config = sys.modules.get("sglang.srt.configs.model_config")
    legacy_methods = getattr(model_config, "QUANTIZATION_METHODS", None)
    if legacy_methods is not None:
        for name, config in methods.items():
            legacy_methods.setdefault(name, config)
    return True


def around_get_quantization_config(
    original: Callable[..., Any], quantization: str
):
    install_quantization_method()
    return original(quantization)


def around_verify_quantization(original: Callable[..., Any], model_config):
    install_quantization_method()
    if getattr(model_config, "quantization", None) not in _HCU_QUANTIZATION_NAMES:
        return original(model_config)

    module = sys.modules.get("sglang.srt.configs.model_config")
    if module is None or not hasattr(module, "is_hip"):
        return original(model_config)

    original_is_hip = module.is_hip
    module.is_hip = lambda: False
    try:
        return original(model_config)
    finally:
        module.is_hip = original_is_hip


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.get_quantization_config",
        around_get_quantization_config,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.configs.model_config.ModelConfig._verify_quantization",
        around_verify_quantization,
        HookType.AROUND,
    )
