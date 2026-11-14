"""hbird-gemm as a bf16 GEMM backend."""

from ..provider import Bf16GemmProvider


def _ops():
    import hbird_ops

    return hbird_ops


class HbirdProvider(Bf16GemmProvider):
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


def load() -> HbirdProvider | None:
    provider = HbirdProvider()
    return provider if provider.available() else None
