"""[hbird_bf16_gemm] SGLang glue for the hbird-gemm bf16 decode GEMM.

``prepare_model`` builds the packed-B copy for every bf16 LinearBase the hbird
hook wrapped and sizes the decode workspaces before CUDA-graph capture.
``lm_head_matmul`` is the unquantized lm_head fast path, which SGLang routes
around quant_method.apply and the hook never sees. Both are gated by
``HBIRD_GEMM_SGLANG`` and return None on anything they do not own.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

#: M values a decode CUDA graph can issue.  Spec decode runs the target at
#: bs * num_draft_tokens and the draft module at bs, so sweeping 1..128 covers
#: every captured tier without having to introspect the graph config.
_DECODE_MAX_M = 128

_EMBEDDING_TYPES = {"VocabParallelEmbedding"}
_LM_HEAD_TYPES = {"ParallelLMHead"}


def enabled() -> bool:
    value = os.environ.get("HBIRD_GEMM_SGLANG", "0")
    return value.strip().lower() not in ("", "0", "false", "off", "no")


def _ops():
    import hbird_ops

    return hbird_ops


def role() -> str:
    """Which worker this process converts: ``all``, ``target`` or ``draft``."""
    value = os.environ.get("HBIRD_GEMM_ROLE", "all").strip().lower()
    if value in ("", "all", "both", "1", "true", "yes", "on"):
        return "all"
    if value in ("none", "off", "0", "false", "no"):
        return "none"
    return value


def _selected(runner) -> bool:
    """Whether ``HBIRD_GEMM_ROLE`` covers this worker.

    An excluded worker keeps every bf16 weight and every ``UnquantizedLinear``
    on SGLang's own GEMM: the hook is installed process-wide, so exclusion is
    recorded per module as ``_hbird_off`` and the hook honours it.  Target and
    draft live in one process at TP=1, so the flag has to be per module.
    """
    want = role()
    if want == "all":
        return True
    if want == "none":
        return False
    is_draft = bool(getattr(runner, "is_draft_worker", False))
    return (want == "draft") == is_draft


def _packed_shape(handle):
    return int(handle.N), int(handle.K)


def prepare_model(model, runner=None):
    """One-copy conversion + workspace warm-up; never raises."""
    if not enabled():
        return None
    try:
        ops = _ops()
        limit = int(ops.bf16_max_m())
    except Exception as exc:  # noqa: BLE001 - optional path
        logger.warning("[hbird_bf16_gemm] disabled, no usable .so: %s", exc)
        return None
    if limit <= 0:
        logger.warning("[hbird_bf16_gemm] disabled, capability probe says 0")
        return None

    converted, kept, skipped = 0, 0, []
    shapes = {}
    freed = 0
    try:
        modules = list(model.named_modules())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[hbird_bf16_gemm] cannot walk model: %s", exc)
        return None

    if not _selected(runner):
        for _, module in modules:
            try:
                module._hbird_off = True
            except Exception:  # noqa: BLE001 - a slotted module: nothing to mark
                pass
        logger.info(
            "[hbird_bf16_gemm] %s excluded by HBIRD_GEMM_ROLE=%s; kept on SGLang",
            "draft worker" if getattr(runner, "is_draft_worker", False) else "target",
            role(),
        )
        return None

    for name, module in modules:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
            continue
        if not weight.is_cuda or weight.dtype != torch.bfloat16:
            continue
        if getattr(module, "bias", None) is not None:
            continue
        kind = type(module).__name__
        if kind in _EMBEDDING_TYPES:
            continue
        if kind in _LM_HEAD_TYPES or module is getattr(model, "lm_head", None):
            try:
                ops.prepare_bf16_weight(weight.detach(), nt=False)
                # ``prepare_bf16_weight`` returns the warm packed buffer (a
                # flat tensor), not a Bf16PackedWeight -- it has no .N/.K.  Take
                # the key from the weight so a padded pad_n cannot leak in.
                shapes[(int(weight.shape[0]), int(weight.shape[1]))] = weight.device
                kept += 1
            except Exception as exc:  # noqa: BLE001
                skipped.append((name, f"{type(exc).__name__}: {exc}"))
            continue
        apply = getattr(getattr(module, "quant_method", None), "apply", None)
        if not getattr(apply, "_hbird_wrapped", False):
            continue
        try:
            ok, why = ops.bf16_can_pack(weight.detach(), nt=False, max_m=limit)
            if not ok:
                skipped.append((name, why))
                continue
            nbytes = int(weight.numel()) * int(weight.element_size())
            handle = ops.bf16_pack_weight(weight.detach(), nt=False, max_m=limit)
            ops.bf16_release_weight(module, handle, empty_cache=False)
            module._hbird_dbg_name = name
            shapes[_packed_shape(handle)] = weight.device
            converted += 1
            freed += nbytes
        except Exception as exc:  # noqa: BLE001
            skipped.append((name, f"{type(exc).__name__}: {exc}"))

    # Workspaces: a captured graph may not allocate, so every (M, N, K) it can
    # reach must exist now.  M > 128 binds to the one shared prefill buffer.
    prepared, failed = 0, 0
    try:
        for (n, k), device in shapes.items():
            for m in range(1, _DECODE_MAX_M + 1):
                try:
                    ops.prepare_bf16(m, n, k, device)
                    prepared += 1
                except Exception:  # noqa: BLE001 - shape refused at this M
                    failed += 1
        ops.prepare_bf16_prefill(next(iter(shapes.values())) if shapes else None)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[hbird_bf16_gemm] workspace warm-up failed: %s", exc)

    logger.info(
        "[hbird_bf16_gemm] %s: packed %d linear(s), freed %.2f GiB, lm_head kept "
        "(packed copy only) %d, skipped %d, %d (M,N,K) ws prepared (%d refused), "
        ".so cap M <= %d",
        "draft worker" if getattr(runner, "is_draft_worker", False) else "target",
        converted,
        freed / 2**30,
        kept,
        len(skipped),
        prepared,
        failed,
        limit,
    )
    if skipped:
        logger.info("[hbird_bf16_gemm] skipped: %s", skipped[:8])
    return {
        "converted": converted,
        "freed": freed,
        "kept": kept,
        "skipped": skipped,
        "shapes": sorted(shapes),
        "prepared": prepared,
    }


def lm_head_matmul(hidden_states, lm_head, embedding_bias, use_fp32_lm_head):
    """hbird Gemm for an unquantized bf16 lm_head, or None to keep the original."""
    if not enabled() or use_fp32_lm_head or embedding_bias is not None:
        return None
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16:
        return None
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_cuda:
        return None
    if not hidden_states.is_contiguous():
        return None  # let the caller's path own the copy; never add one here
    k = int(weight.shape[1])
    if hidden_states.shape[-1] != k:
        return None
    m = hidden_states.numel() // k
    # No upper bound on M: verify is bs * draft_tokens, and _get_pruned_states
    # already hands us at most bs*draft rows here (extend is pruned to the last
    # token of each sequence), so the reachable M is bounded by MAX_BS, not by
    # a decode tier.  Capping it at 128 silently sent M = 144/160 -- i.e. every
    # c36 and c40 step -- to hipBLASLt, which is 1.85x slower on this head
    # (3024 us vs 1639 us at M = 144, CUDA graph, measured).  Workspaces past
    # the sweep band are left to the eager warm-up, exactly as the layer GEMMs
    # have always done.
    if m <= 0:
        return None
    try:
        ops = _ops()
        handle = ops.bf16_packed_handle(lm_head)
        if handle is not None:
            return ops.bf16_linear_packed(hidden_states, handle)
        return ops.bf16_linear(hidden_states, weight)
    except Exception:  # noqa: BLE001 - the fallback is always correct
        return None
