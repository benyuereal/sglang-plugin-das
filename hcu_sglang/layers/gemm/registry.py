"""Backend selection and the one-shot model conversion."""

import logging
import os
from dataclasses import dataclass, field

import torch

from hcu_sglang.constants import BYTES_PER_GIB, DECODE_MAX_M, SKIPPED_LOG_LIMIT

from .providers import hbird

_LOG = logging.getLogger(__name__)

#: Forward reads ``.weight`` directly, so an embedding cannot be packed away.
_EMBEDDINGS = {"VocabParallelEmbedding"}
#: The head keeps its source weight; ``_compute_lm_head`` reads it for dispatch.
_LM_HEADS = {"ParallelLMHead"}

#: A successful load is cached: the .so probe behind it is memoised, so its
#: answer cannot change within the process.
_backend = None

# --- enablement -------------------------------------------------------------

def enabled() -> bool:
    value = os.environ.get("HBIRD_GEMM_SGLANG", "0")
    return value.strip().lower() not in ("", "0", "false", "off", "no")

def role() -> str:
    """Which worker this process converts: ``all``, ``target`` or ``draft``."""
    value = os.environ.get("HBIRD_GEMM_ROLE", "all").strip().lower()
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

def provider():
    """The loaded backend, or None; a failed load is retried on the next call."""
    global _backend
    if not enabled():
        return None
    if _backend is None:
        _backend = hbird.load()
    return _backend

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
        _LOG.warning("[hbird_bf16] cannot walk model: %s", exc)
        return None

def _exclude(modules, runner):
    """Mark every module off; hbird-gemm's shim reads this flag on dispatch."""
    for _, module in modules:
        try:
            module._hbird_off = True
        except Exception:  # noqa: BLE001 - a slotted module: nothing to mark
            pass
    _LOG.info(
        "[hbird_bf16] %s excluded by HBIRD_GEMM_ROLE=%s; kept on SGLang",
        "draft worker" if runner is not None and runner.is_draft_worker
        else "target",
        role(),
    )

def _convert_one(backend, model, name, module, stats):
    weight = _eligible_weight(module)
    if weight is None:
        return
    if type(module).__name__ in _LM_HEADS or module is getattr(model, "lm_head", None):
        _keep_lm_head(backend, name, weight, stats)
    elif backend.owns(module):
        _pack_linear(backend, name, module, weight, stats)

def _eligible_weight(module):
    """The weight to pack, or None when this module keeps the upstream GEMM."""
    weight = getattr(module, "weight", None)
    packable = (
        isinstance(weight, torch.Tensor)
        and weight.dim() == 2
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and module._parameters.get("bias") is None
        and type(module).__name__ not in _EMBEDDINGS
    )
    return weight if packable else None

def _keep_lm_head(backend, name, weight, stats):
    try:
        backend.prepare_lm_head(weight)
        # Key by the weight's own shape so a padded pad_n cannot leak in.
        stats.shapes[(int(weight.shape[0]), int(weight.shape[1]))] = weight.device
        stats.kept += 1
    except Exception as exc:  # noqa: BLE001
        stats.skipped.append((name, f"{type(exc).__name__}: {exc}"))

def _pack_linear(backend, name, module, weight, stats):
    try:
        ok, why = backend.can_pack(weight)
        if not ok:
            stats.skipped.append((name, why))
            return
        stats.freed += int(weight.numel()) * int(weight.element_size())
        handle = backend.pack(weight)
        backend.release(module, handle)
        stats.shapes[(int(handle.N), int(handle.K))] = weight.device
        stats.converted += 1
    except Exception as exc:  # noqa: BLE001
        stats.skipped.append((name, f"{type(exc).__name__}: {exc}"))

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
        _LOG.warning("[hbird_bf16] workspace warm-up failed: %s", exc)

def _log_summary(stats):
    _LOG.info(
        "[hbird_bf16] packed %d linear(s), freed %.2f GiB, lm_head kept %d, "
        "skipped %d, %d (M,N,K) ws prepared (%d refused), cap M <= %d",
        stats.converted, stats.freed / BYTES_PER_GIB, stats.kept,
        len(stats.skipped), stats.prepared, stats.refused, stats.limit,
    )
    if stats.skipped:
        _LOG.info("[hbird_bf16] skipped: %s", stats.skipped[:SKIPPED_LOG_LIMIT])

# --- lm_head fast path ------------------------------------------------------

def lm_head_linear(hidden_states, lm_head, embedding_bias, use_fp32_lm_head):
    """Return the provider's lm_head GEMM, or None to keep the original path."""
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
    weight = getattr(lm_head, "weight", None)
    method = getattr(lm_head, "quant_method", None)
    # A method we installed (the fp8 draft head) owns this GEMM.
    ours = method is not None and type(method).__module__.startswith("hcu_sglang.")
    # A non-contiguous hidden state is the caller's to copy; never add one here.
    # No M cap: verify runs bs * draft rows and extend prunes to one token per
    # sequence, so M is bounded by MAX_BS. Capping at DECODE_MAX_M sent
    # M = 144/160 to hipBLASLt, 1.85x slower (measured, CUDA graph).
    usable = (
        not use_fp32_lm_head
        and embedding_bias is None
        and not ours
        and isinstance(weight, torch.Tensor)
        and weight.dtype == torch.bfloat16
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_cuda
        and hidden_states.is_contiguous()
        and hidden_states.numel() > 0
        and weight.dim() == 2
        and hidden_states.shape[-1] == int(weight.shape[1])
    )
    return weight if usable else None
