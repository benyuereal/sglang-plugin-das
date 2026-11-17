"""Packed GEMM for a quantized LM head.

Which backend serves this GEMM is decided by the GEMM backend seam, not here:
the caller hands over a quantized weight and gets back a callable plus the
largest row count it was prepared for, or ``(None, 0)`` when no backend can
take it.
"""

from hcu_sglang.layers.gemm import backend


def prepare(weight_q):
    """The backend's packed GEMM and its largest M, or (None, 0)."""
    return backend.prepare_quant_head(weight_q)
