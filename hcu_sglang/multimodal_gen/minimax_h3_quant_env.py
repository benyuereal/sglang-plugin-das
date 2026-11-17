_NAMES = frozenset((
    "SGLANG_MINIMAX_H3_QUANT_OPTIMIZE",
    "SGLANG_MINIMAX_H3_INT8_ENABLE",
    "SGLANG_MINIMAX_H3_INT8_FUSION",
    "SGLANG_MINIMAX_H3_FP8_OPTIMIZE",
))


def __getattr__(name):
    if name not in _NAMES:
        raise AttributeError(name)
    from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
    return get_bool_env_var(name, "true")
