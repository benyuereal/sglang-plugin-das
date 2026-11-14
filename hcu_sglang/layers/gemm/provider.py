"""Contract a bf16 GEMM backend implements."""

from typing import Protocol


class Bf16GemmProvider(Protocol):
    name: str

    def available(self) -> bool:
        """Whether the backend's library loaded and reports a usable shape."""

    def max_m(self) -> int:
        """Largest M with a dedicated plan; past it the shared buffer serves."""

    def owns(self, module) -> bool:
        """Whether this module's bf16 linear belongs to the backend."""

    def can_pack(self, weight) -> tuple[bool, str]:
        """(True, "") when the weight packs, else (False, reason)."""

    def pack(self, weight):
        """Build the packed-B copy and return its handle."""

    def release(self, module, handle) -> None:
        """Drop the source weight; the packed copy owns the memory now."""

    def prepare(self, m: int, n: int, k: int, device) -> None:
        """Allocate the workspace for one (M, N, K) a captured graph can issue."""

    def prepare_prefill(self, device) -> None:
        """Allocate the shared prefill-sized workspace."""

    def prepare_lm_head(self, weight) -> None:
        """Warm the packed copy without releasing the source weight."""

    def linear(self, x, weight, handle=None):
        """Return the GEMM result, or None to fall back to the caller's path."""
