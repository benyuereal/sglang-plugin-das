"""The GEMM backend seam: select one implementation and call it through here.

A backend owns a packed-weight GEMM family. The choice is per capability:
``SGLANG_GEMM_BACKEND`` picks the bf16 linears and the unquantized lm_head
(legacy: ``HBIRD_GEMM_SGLANG=1`` means ``hbird``); ``SGLANG_QUANT_HEAD_BACKEND``
picks the quantized draft head and defaults to the same name. When unset or
unavailable the caller keeps the upstream GEMM. Concrete implementations live
in :mod:`hcu_sglang.layers.gemm.providers`.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Protocol

import torch

from hcu_sglang.constants import BYTES_PER_GIB, DECODE_MAX_M, SKIPPED_LOG_LIMIT

_LOG = logging.getLogger(__name__)

#: Forward reads ``.weight`` directly, so an embedding cannot be packed away.
_EMBEDDINGS = {"VocabParallelEmbedding"}
#: The head keeps its source weight; ``_compute_lm_head`` reads it for dispatch.
_LM_HEADS = {"ParallelLMHead"}


class GemmProvider(Protocol):
    """What a concrete backend implements; see ``providers/hbird.py``."""

    name: str

    def available(self) -> bool:
        """Whether the backend loaded and reports a usable shape."""

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

    def packed_handle(self, module):
        """The handle ``pack`` installed on this module, for the lm_head call."""

    def prepare_quant_head(self, weight_q):
        """Pack a quantized lm_head; return ``(gemm, max_m)`` or ``(None, 0)``."""

    def exclude(self, modules) -> None:
        """Mark modules this worker keeps on the upstream GEMM."""

    def install_shim(self) -> bool:
        """Install the backend's own linear-layer seam, if it has one."""

# --- selection --------------------------------------------------------------

def _load_hbird():
    from .providers import hbird

    return hbird.load()


_LOADERS = {"hbird": _load_hbird}
_DISABLED = ("", "none", "off", "0", "false", "no")

#: Loaded backends, keyed by name: the probe behind a load is memoised, so its
#: answer cannot change within the process, and two capabilities that pick the
#: same name share one instance.
_providers = {}


def selected_name() -> str:
    """The backend name from ``SGLANG_GEMM_BACKEND``, or the legacy switch."""
    value = os.environ.get("SGLANG_GEMM_BACKEND", "").strip().lower()
    if value:
        return value
    legacy = os.environ.get("HBIRD_GEMM_SGLANG", "0")
    return "hbird" if legacy.strip().lower() not in _DISABLED else "none"


def quant_head_name() -> str:
    """The quantized-head backend; defaults to the GEMM backend."""
    value = os.environ.get("SGLANG_QUANT_HEAD_BACKEND", "").strip().lower()
    return value or selected_name()

def enabled() -> bool:
    return selected_name() not in _DISABLED

def role() -> str:
    """Which worker this process converts: ``all``, ``target`` or ``draft``."""
    value = os.environ.get("SGLANG_GEMM_ROLE")
    if value is None:
        value = os.environ.get("HBIRD_GEMM_ROLE", "all")
    value = value.strip().lower()
    if value in ("", "all", "both", "1", "true", "yes", "on"):
        return "all"
    if value in ("none", "off", "0", "false", "no"):
        return "none"
    return value

def selected(runner) -> bool:
    """Whether the role covers this worker; target and draft share one process."""
    want = role()
    if want == "all":
        return True
    if want == "none" or runner is None:
        return False
    return (want == "draft") == bool(runner.is_draft_worker)

def _provider_for(name):
    if name in _providers:
        return _providers[name]
    loader = _LOADERS.get(name)
    if loader is None:
        if name not in _DISABLED:
            _LOG.warning(
                "[gemm] backend %r is not available; upstream GEMM stays", name
            )
        loaded = None
    else:
        loaded = loader()
    _providers[name] = loaded
    return loaded

def provider():
    """The bf16 GEMM backend, or None."""
    return _provider_for(selected_name())

def quant_head_provider():
    """The quantized-head backend, or None; chosen independently."""
    return _provider_for(quant_head_name())

def install_shim() -> bool:
    """Install the backend's linear-layer seam; False when none is selected."""
    backend = provider()
    return False if backend is None else backend.install_shim()

def prepare_quant_head(weight_q):
    """Ask the quantized-head backend for its GEMM, or (None, 0) when it declines."""
    backend = quant_head_provider()
    if backend is None:
        return None, 0
    try:
        return backend.prepare_quant_head(weight_q)
    except Exception as exc:  # noqa: BLE001 - optional path
        _LOG.info("[gemm] quant head unavailable (%s); head stays bf16", exc)
        return None, 0

# --- conversion -------------------------------------------------------------

@dataclass
class _Stats:
    """Sinks for one conversion pass; ``as_result`` is the caller-facing view."""

    limit: int
    converted: int = 0
    kept: int = 0
    freed: int = 0
    skipped: list = field(default_factory=list)
    shapes: dict = field(default_factory=dict)
    prepared: int = 0
    refused: int = 0

    def as_result(self) -> dict:
        return {
            "converted": self.converted,
            "freed": self.freed,
            "kept": self.kept,
            "skipped": self.skipped,
            "shapes": sorted(self.shapes),
            "prepared": self.prepared,
        }

def convert_model(model, runner=None):
    """Pack every weight the provider owns and warm the decode workspaces."""
    backend = provider()
    if backend is None:
        return None
    modules = _walk(model)
    if modules is None:
        return None
    if not selected(runner):
        _exclude(modules, runner)
        return None

    stats = _Stats(limit=backend.max_m())
    for name, module in modules:
        _convert_one(backend, model, name, module, stats)
    _warm_workspaces(backend, stats.shapes, stats)
    _log_summary(stats)
    return stats.as_result()

def _walk(model):
    try:
        return list(model.named_modules())
    except Exception as exc:  # noqa: BLE001 - optional path
        _LOG.warning("[gemm] cannot walk model: %s", exc)
        return None

def _exclude(modules, runner):
    """Mark every module off; the backend's dispatch reads this flag."""
    backend = provider()
    if backend is not None:
        backend.exclude(modules)
    _LOG.info(
        "[gemm] %s excluded by role=%s; kept on upstream",
        "draft worker" if runner is not None and runner.is_draft_worker
        else "target",
        role(),
    )

def _convert_one(backend, model, name, module, stats):
    weight = _eligible_weight(module)
    if weight is None:
        return
    if type(module).__name__ in _LM_HEADS or module is getattr(model, "lm_head", None):
        _attempt(stats, name, _keep_lm_head, backend, weight)
    elif backend.owns(module):
        _attempt(stats, name, _pack_linear, backend, name, module, weight)

def _attempt(stats, name, fn, *args):
    """One layer's failure becomes a skip line, never a failed model load."""
    try:
        fn(*args, stats)
    except Exception as exc:  # noqa: BLE001 - optional path
        stats.skipped.append((name, f"{type(exc).__name__}: {exc}"))

def _eligible_weight(module):
    """The weight to pack, or None when this module keeps the upstream GEMM."""
    # Not every module in the walk has a weight (activations, norms, pools).
    weight = getattr(module, "weight", None)
    if not _is_bf16_matrix(weight) or not weight.is_cuda:
        return None
    if module._parameters.get("bias") is not None:
        return None
    if type(module).__name__ in _EMBEDDINGS:
        return None
    return weight

def _keep_lm_head(backend, weight, stats):
    backend.prepare_lm_head(weight)
    # Key by the weight's own shape so a padded pad_n cannot leak in.
    stats.shapes[(int(weight.shape[0]), int(weight.shape[1]))] = weight.device
    stats.kept += 1

def _pack_linear(backend, name, module, weight, stats):
    ok, why = backend.can_pack(weight)
    if not ok:
        stats.skipped.append((name, why))
        return
    stats.freed += int(weight.numel()) * int(weight.element_size())
    handle = backend.pack(weight)
    backend.release(module, handle)
    stats.shapes[(int(handle.N), int(handle.K))] = weight.device
    stats.converted += 1

def _warm_workspaces(backend, shapes, stats):
    """Allocate every (M, N, K) a captured graph may issue, plus the prefill one."""
    # A captured graph may not allocate, so every reachable (M, N, K) must exist
    # now; M past DECODE_MAX_M binds to the shared prefill buffer.
    try:
        for (n, k), device in shapes.items():
            for m in range(1, DECODE_MAX_M + 1):
                try:
                    backend.prepare(m, n, k, device)
                    stats.prepared += 1
                except Exception:  # noqa: BLE001 - shape refused at this M
                    stats.refused += 1
        backend.prepare_prefill(next(iter(shapes.values())) if shapes else None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("[gemm] workspace warm-up failed: %s", exc)

def _log_summary(stats):
    _LOG.info(
        "[gemm] packed %d linear(s), freed %.2f GiB, lm_head kept %d, "
        "skipped %d, %d (M,N,K) ws prepared (%d refused), cap M <= %d",
        stats.converted, stats.freed / BYTES_PER_GIB, stats.kept,
        len(stats.skipped), stats.prepared, stats.refused, stats.limit,
    )
    if stats.skipped:
        _LOG.info("[gemm] skipped: %s", stats.skipped[:SKIPPED_LOG_LIMIT])

# --- lm_head fast path ------------------------------------------------------

def lm_head_linear(hidden_states, lm_head, embedding_bias, use_fp32_lm_head):
    """Return the backend's lm_head GEMM, or None to keep the original path."""
    backend = provider()
    if backend is None:
        return None
    weight = _lm_head_weight(hidden_states, lm_head, embedding_bias, use_fp32_lm_head)
    if weight is None:
        return None
    try:
        return backend.linear(hidden_states, weight, backend.packed_handle(lm_head))
    except Exception:  # noqa: BLE001 - the fallback is always correct
        return None

def _lm_head_weight(hidden_states, lm_head, embedding_bias, use_fp32_lm_head):
    """The bf16 weight to hand the backend, or None to keep the caller's GEMM."""
    if use_fp32_lm_head or embedding_bias is not None:
        return None
    if _head_owned_elsewhere(lm_head):
        return None
    weight = lm_head.weight
    if not _is_bf16_matrix(weight):
        return None
    if not _rows_fit(hidden_states, weight):
        return None
    return weight

def _head_owned_elsewhere(lm_head) -> bool:
    """Our quantized draft head replaced the method; its GEMM is not ours."""
    # Custom heads do not all subclass LinearBase, so the field may be absent.
    method = getattr(lm_head, "quant_method", None)
    return method is not None and type(method).__module__.startswith("hcu_sglang.")

def _is_bf16_matrix(weight) -> bool:
    return (
        isinstance(weight, torch.Tensor)
        and weight.dim() == 2
        and weight.dtype == torch.bfloat16
    )

def _rows_fit(hidden_states, weight) -> bool:
    """The input the provider can consume as-is; a copy of it is the caller's."""
    # No M cap: verify runs bs * draft rows and extend prunes to one token per
    # sequence, so M is bounded by MAX_BS. Capping at DECODE_MAX_M sent
    # M = 144/160 to hipBLASLt, 1.85x slower (measured, CUDA graph).
    return (
        hidden_states.dtype == torch.bfloat16
        and hidden_states.is_cuda
        and hidden_states.is_contiguous()
        and hidden_states.numel() > 0
        and hidden_states.shape[-1] == int(weight.shape[1])
    )
