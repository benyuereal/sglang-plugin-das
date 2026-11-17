"""hbird-gemm as a bf16 GEMM backend."""

import logging
import os
import sys

import torch

from hcu_sglang.constants import BYTES_PER_GIB

from ..backend import GemmProvider

_LOG = logging.getLogger(__name__)

_MAX_M_DEFAULT = 192


def _ops():
    try:
        import hbird_ops

        return hbird_ops
    except ImportError:
        root = os.environ.get("HBIRD_GEMM_HOME") or ""
        path = os.path.join(root, "plugins")
        if not root or not os.path.isdir(path):
            raise
        if path not in sys.path:
            sys.path.insert(0, path)
        import hbird_ops

        return hbird_ops


class HbirdProvider(GemmProvider):
    name = "hbird"

    def __init__(self):
        self._ops = None
        self._max_m = 0
        try:
            ops = _ops()
            self._max_m = int(ops.bf16_max_m())
            self._ops = ops
        except Exception:  # noqa: BLE001 - no usable .so, stay off
            self._ops = None

    def available(self) -> bool:
        return self._ops is not None and self._max_m > 0

    def max_m(self) -> int:
        return self._max_m

    def owns(self, module) -> bool:
        apply = getattr(getattr(module, "quant_method", None), "apply", None)
        # The shim in hbird-gemm sets this on the apply it wrapped.
        return bool(getattr(apply, "_hbird_wrapped", False))

    def can_pack(self, weight):
        return self._ops.bf16_can_pack(
            weight.detach(), nt=False, max_m=self._max_m
        )

    def pack(self, weight):
        return self._ops.bf16_pack_weight(
            weight.detach(), nt=False, max_m=self._max_m
        )

    def release(self, module, handle) -> None:
        self._ops.bf16_release_weight(module, handle, empty_cache=False)

    def prepare(self, m: int, n: int, k: int, device) -> None:
        self._ops.prepare_bf16(m, n, k, device)

    def prepare_prefill(self, device) -> None:
        self._ops.prepare_bf16_prefill(device)

    def prepare_lm_head(self, weight) -> None:
        self._ops.prepare_bf16_weight(weight.detach(), nt=False)

    def linear(self, x, weight, handle=None):
        if handle is not None:
            return self._ops.bf16_linear_packed(x, handle)
        return self._ops.bf16_linear(x, weight)

    def packed_handle(self, module):
        return self._ops.bf16_packed_handle(module)

    def exclude(self, modules) -> None:
        """Mark modules this worker keeps on the upstream GEMM."""
        for _, module in modules:
            try:
                module._hbird_off = True
            except AttributeError:
                pass

    def prepare_quant_head(self, weight_q):
        ops = self._ops
        vocab, hidden = weight_q.shape
        ms = [
            m
            for m in range(1, _MAX_M_DEFAULT + 1)
            if ops.guard(m, vocab, hidden) == 0
        ]
        if not ms:
            raise RuntimeError("guard refuses every M")
        ops.prepare_weight(weight_q)
        for m in ms:
            ops.prepare(m, vocab, hidden, weight_q.device)
        torch.cuda.synchronize()
        _LOG.info(
            "[gemm] quant head on packed tiles: %.2f GiB B, "
            "%d M workspaces (%d..%d)",
            weight_q.numel() / BYTES_PER_GIB,
            len(ms),
            ms[0],
            ms[-1],
        )
        return ops.hbird_fp8_gemm, ms[-1]

    def install_shim(self) -> bool:
        import hbird_sglang

        return bool(hbird_sglang.install())


def load() -> HbirdProvider | None:
    provider = HbirdProvider()
    return provider if provider.available() else None
