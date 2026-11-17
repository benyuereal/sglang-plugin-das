"""Pluggable bf16 GEMM backends for the decode path.

A provider owns the packed-B copy of a bf16 ``LinearBase`` and answers the
GEMM. ``convert_model`` walks the model once after load and hands every
eligible module to the selected provider; ``lm_head_linear`` is the
unquantized lm_head fast path, which SGLang routes around ``quant_method``.
"""

from .backend import convert_model, enabled, lm_head_linear, selected

__all__ = ["convert_model", "enabled", "lm_head_linear", "selected"]
