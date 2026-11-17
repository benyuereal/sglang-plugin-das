"""Packed GEMM for a quantized LM head.

The head can be served by three fp8 W8A8 GEMMs and never depends on one of
them: hbird's packed tiles, lightop's hipBLASLt channelwise kernel, or
``torch._scaled_mm``. ``SGLANG_QUANT_HEAD_BACKEND`` picks one by name;
``auto`` tries hbird, then lightop, then torch. Every adapter implements the
same ``gemm(xq, wq, xs, ws, out=..., prepare_shape=...)`` contract.
"""

import logging

import torch

from . import backend

logger = logging.getLogger(__name__)

# Same row budget the hbird path prepares workspaces for; the caller's output
# buffer is sized by it and wider rows fall back to the plain matmul.
_MAX_M = 192


def _with_out(result: torch.Tensor, out) -> torch.Tensor:
    """lightop/torch allocate their result; the caller's buffer is graph state."""
    if out is None:
        return result
    out.copy_(result)
    return out


def _prepare_hbird(weight_q):
    return backend.prepare_quant_head(weight_q)


def _prepare_lightop(weight_q):
    from lightop.gemm_ops import native

    def gemm(xq, wq, xs, ws, *, out=None, prepare_shape=True):
        status, y = native.hipblaslt_w8a8_channelwise_gemm(
            xq, wq, xs, ws,
            int(xq.shape[0]), int(wq.shape[0]), int(xq.shape[1]),
            "NT", torch.bfloat16,
        )
        if not status:
            raise RuntimeError("lightop channelwise fp8 gemm declined")
        if y.dim() == 3 and y.shape[0] == 1:
            y = y[0]
        return _with_out(y, out)

    return gemm, _MAX_M


def _prepare_torch(weight_q):
    # _scaled_mm wants [K, N] column-major; the transposed view is exactly that.
    transposed = weight_q.t()

    def gemm(xq, wq, xs, ws, *, out=None, prepare_shape=True):
        y = torch._scaled_mm(
            xq, transposed, xs, ws.reshape(1, -1), out_dtype=torch.bfloat16
        )
        return _with_out(y, out)

    return gemm, _MAX_M


_BACKENDS = {
    "hbird": _prepare_hbird,
    "lightop": _prepare_lightop,
    "torch": _prepare_torch,
}
_AUTO_ORDER = ("hbird", "lightop", "torch")


def prepare(weight_q):
    """The packed GEMM and its largest M, or (None, 0) when none takes it."""
    name = backend.quant_head_name()
    order = _AUTO_ORDER if name == "auto" else (name,)
    for candidate in order:
        make = _BACKENDS.get(candidate)
        if make is None:
            continue
        try:
            gemm, max_m = make(weight_q)
        except Exception as exc:  # noqa: BLE001 - optional backend
            logger.info(
                "[gemm] quant head backend %s unavailable (%s)", candidate, exc
            )
            continue
        if gemm is not None:
            logger.info("[gemm] quant head on %s (max_m=%d)", candidate, max_m)
            return gemm, max_m
    return None, 0
