"""Packed GEMM for a quantized LM head.

Which backend serves this GEMM is decided here, not in the quantization code:
the caller hands over a quantized weight and gets back a callable plus the largest
row count it was prepared for, or ``(None, 0)`` when no backend can take it.
"""

from __future__ import annotations

import logging
import os
import time

import torch

from hcu_sglang.constants import BYTES_PER_GIB

logger = logging.getLogger(__name__)

_MAX_M_DEFAULT = 192


def _ops():
    try:
        import hbird_ops

        return hbird_ops
    except ImportError:
        import sys

        root = os.environ.get("HBIRD_GEMM_HOME") or ""
        path = os.path.join(root, "plugins")
        if not root or not os.path.isdir(path):
            raise
        if path not in sys.path:
            sys.path.insert(0, path)
        import hbird_ops

        return hbird_ops


def _supported_ms(ops, *, vocab: int, hidden: int):
    ms = []
    for m in range(1, _MAX_M_DEFAULT + 1):
        if ops.guard(m, vocab, hidden) != 0:
            break
        ms.append(m)
    if not ms:
        raise RuntimeError("guard refuses every M")
    return ms


def prepare(weight_q: torch.Tensor):
    """The packed GEMM and its largest M, or (None, 0) when no backend takes it."""
    try:
        ops = _ops()
        vocab, hidden = weight_q.shape
        device = weight_q.device
        ms = _supported_ms(ops, vocab=vocab, hidden=hidden)
        started = time.perf_counter()
        ops.prepare_weight(weight_q)
        for m in ms:
            ops.prepare(m, vocab, hidden, device)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - optional path
        logger.info(
            "[quant_lm_head] backend unavailable (%s); head stays bf16", exc
        )
        return None, 0
    logger.info(
        "[quant_lm_head] head on packed tiles: %.2f GiB B, "
        "%d M workspaces (%d..%d), %.2fs",
        weight_q.numel() / BYTES_PER_GIB,
        len(ms),
        ms[0],
        ms[-1],
        time.perf_counter() - started,
    )
    return ops.hbird_fp8_gemm, ms[-1]
