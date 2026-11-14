"""Backend selection and the one-shot model conversion."""

import logging
import os

import torch

from hcu_sglang.constants import DECODE_MAX_M, SKIPPED_LOG_LIMIT

from .providers import hbird

_LOG = logging.getLogger(__name__)

#: Forward reads ``.weight`` directly, so an embedding cannot be packed away.
_EMBEDDINGS = {"VocabParallelEmbedding"}
#: The head keeps its source weight; ``_compute_lm_head`` reads it for dispatch.
_LM_HEADS = {"ParallelLMHead"}


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
    return hbird.load() if enabled() else None


def convert_model(model, runner=None):
    """Pack every weight the provider owns and warm the decode workspaces."""
    backend = provider()
    if backend is None:
        return None
    try:
        modules = list(model.named_modules())
    except Exception as exc:  # noqa: BLE001 - optional path
        _LOG.warning("[hbird_bf16] cannot walk model: %s", exc)
        return None

    if not selected(runner):
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
        return None

    converted, kept, skipped, freed = 0, 0, [], 0
    shapes = {}
    limit = backend.max_m()
    for name, module in modules:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
            continue
        if not weight.is_cuda or weight.dtype != torch.bfloat16:
            continue
        if getattr(module, "bias", None) is not None:
            continue
        kind = type(module).__name__
        if kind in _EMBEDDINGS:
            continue
        if kind in _LM_HEADS or module is getattr(model, "lm_head", None):
            try:
                backend.prepare_lm_head(weight)
                # Key by the weight's own shape so a padded pad_n cannot leak in.
                shapes[(int(weight.shape[0]), int(weight.shape[1]))] = weight.device
                kept += 1
            except Exception as exc:  # noqa: BLE001
                skipped.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        if not backend.owns(module):
            continue
        try:
            ok, why = backend.can_pack(weight)
            if not ok:
                skipped.append((name, why))
                continue
            freed += int(weight.numel()) * int(weight.element_size())
            handle = backend.pack(weight)
            backend.release(module, handle)
            module._hbird_dbg_name = name
            shapes[(int(handle.N), int(handle.K))] = weight.device
            converted += 1
        except Exception as exc:  # noqa: BLE001
            skipped.append((name, f"{type(exc).__name__}: {exc}"))

    # A captured graph may not allocate, so every reachable (M, N, K) must exist
    # now; M past DECODE_MAX_M binds to the shared prefill buffer.
    prepared, failed = 0, 0
    try:
        for (n, k), device in shapes.items():
            for m in range(1, DECODE_MAX_M + 1):
                try:
                    backend.prepare(m, n, k, device)
                    prepared += 1
                except Exception:  # noqa: BLE001 - shape refused at this M
                    failed += 1
        backend.prepare_prefill(next(iter(shapes.values())) if shapes else None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("[hbird_bf16] workspace warm-up failed: %s", exc)

    _LOG.info(
        "[hbird_bf16] packed %d linear(s), freed %.2f GiB, lm_head kept %d, "
        "skipped %d, %d (M,N,K) ws prepared (%d refused), cap M <= %d",
        converted, freed / 2**30, kept, len(skipped), prepared, failed, limit,
    )
    if skipped:
        _LOG.info("[hbird_bf16] skipped: %s", skipped[:SKIPPED_LOG_LIMIT])
    return {
        "converted": converted,
        "freed": freed,
        "kept": kept,
        "skipped": skipped,
        "shapes": sorted(shapes),
        "prepared": prepared,
    }


def lm_head_linear(hidden_states, lm_head, embedding_bias, use_fp32_lm_head):
    """Return the provider's lm_head GEMM, or None to keep the original path."""
    backend = provider()
    if backend is None or use_fp32_lm_head or embedding_bias is not None:
        return None
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16:
        return None
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_cuda:
        return None
    if not hidden_states.is_contiguous():
        return None  # the caller's path owns the copy; never add one here
    k = int(weight.shape[1])
    # No M cap: verify runs bs * draft rows and extend prunes to one token per
    # sequence, so M is bounded by MAX_BS. Capping at DECODE_MAX_M sent
    # M = 144/160 to hipBLASLt, 1.85x slower (measured, CUDA graph).
    if hidden_states.shape[-1] != k or hidden_states.numel() == 0:
        return None
    try:
        return backend.linear(hidden_states, weight, backend.packed_handle(lm_head))
    except Exception:  # noqa: BLE001 - the fallback is always correct
        return None
