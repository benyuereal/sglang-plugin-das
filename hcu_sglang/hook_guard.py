# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Keep HCU hook registration from flooding startup logs with tracebacks.

SGLang resolves every registered target lazily inside
``HookRegistry._apply_target()`` and ``apply_hooks()`` logs a full traceback per
unresolvable target. The HCU plugin registers hooks for optional surfaces --
multimodal / video-generation pipelines, CuTeDSL-only kernels and upstream
private helpers that get renamed between revisions -- so one startup can emit
hundreds of tracebacks whose only meaning is "this optional target is not
present in the current build".

Measured on the 2026-11-09 Qwen3.8-Flash-Next PD run against official v0.5.22:
226 tracebacks, 210 of them ``Failed to apply hooks to <target>`` over 15
targets x 7 processes. 14 of those 15 targets are multimodal/video-generation
surfaces that a text model never loads.

This module wraps ``HookRegistry._apply_target`` once per process so that an
unresolvable target is skipped and reported once in a summary line at INFO,
instead of one traceback per target per process.

``HCU_HOOK_GUARD=0`` restores the unguarded behaviour for debugging.

Why wrap ``_apply_target`` instead of ``register``: resolution is attempted
first and only a *failed* attempt is diverted. Every target that does resolve
takes the original code path unchanged, so the guard cannot skip a hook that
would otherwise have been applied -- which a pre-resolution filter could, since
some target modules only become importable after every plugin has registered.
Registration itself is untouched, so ``HookRegistry._hooks`` still lists exactly
what the plugin declared and the target-contract tooling keeps working.

Scope note: this guard deliberately does not merge or drop the pairs of AROUND
hooks SGLang flags with its "Multiple AROUND hooks" notice. Those pairs wrap
mutually exclusive or complementary runner-backend paths (``w8a8_lightop`` vs
``w8a8_aiter``, marlin vs aiter for unquantized MoE), and every one of them
falls back to ``original(...)`` when its gate misses, so chain order cannot drop
a call. What it does do is *silence the notice for the pairs audited to be
safe*, so a genuinely risky future duplicate stands out. See ``SAFE_AROUND_PAIRS``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Targets where this plugin intentionally registers two AROUND wrappers, both of
# which call ``original(...)`` when their own gate misses, so the inner wrapper
# is never bypassed. Audited 2026-11-10; each entry names the pair and why it is
# safe. SGLang's "Multiple AROUND hooks" warning is suppressed for exactly these
# targets and nothing else -- a new duplicate elsewhere still warns.
SAFE_AROUND_PAIRS: dict[str, str] = {
    "sglang.srt.layers.quantization.w8a8_int8.W8A8Int8MoEMethod.apply":
        "lightop gate (_requested_lightop) and aiter gate (_use_aiter_runner) are "
        "mutually exclusive; both miss-cases return original(...)",
    "sglang.srt.layers.quantization.w8a8_int8.W8A8Int8MoEMethod.create_moe_runner":
        "same lightop/aiter pair; lightop and aiter each delegate when their "
        "backend gate misses",
    "sglang.srt.layers.quantization.w8a8_fp8.W8A8FP8MoEMethod.apply":
        "same lightop/aiter pair; both miss-cases return original(...)",
    "sglang.srt.layers.quantization.w8a8_fp8.W8A8FP8MoEMethod.create_moe_runner":
        "same lightop/aiter pair; both miss-cases return original(...)",
    "sglang.srt.layers.quantization.unquant.UnquantizedFusedMoEMethod.forward_cuda":
        "marlin wrapper additionally requires layer._marlin_w16a16_moe_packed and "
        "the aiter wrapper requires its own gate; both miss-cases call original",
    "sglang.srt.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w8a8_fp8_moe.CompressedTensorsW8A8Fp8MoE.apply_weights":
        "lightop channelwise gate vs aiter gate, both fall through to original",
    "sglang.srt.layers.quantization.compressed_tensors.schemes."
    "compressed_tensors_w8a8_fp8_moe.CompressedTensorsW8A8Fp8MoE."
    "process_weights_after_loading":
        "lightop channelwise gate vs aiter gate, both fall through to original",
    "sglang.srt.arg_groups.parallel_hook.handle_eplb_and_dispatch":
        "placement and integration each call original(server_args) first, then "
        "apply their own validation; neither can skip the other",
}

# Target -> exception type name that made it unresolvable, in encounter order.
_skipped: dict[str, str] = {}
_installed = False


def _report() -> None:
    """Emit one summary line per failure kind for targets this build lacks.

    Logged at WARNING, not INFO: ``register()`` runs before SGLang configures
    logging, so an INFO record emitted from this phase is dropped by the root
    logger's last-resort handler and the operator sees nothing at all. WARNING
    survives that window and still reads as a notice rather than an error.
    """
    if not _skipped:
        return
    grouped: dict[str, list[str]] = {}
    for target, reason in _skipped.items():
        grouped.setdefault(reason, []).append(target)
    for reason, targets in grouped.items():
        logger.warning(
            "hcu hook guard: skipped %d hook target(s) absent in this SGLang "
            "build (%s): %s",
            len(targets),
            reason,
            ", ".join(targets),
        )
    _skipped.clear()


class _KnownPairFilter(logging.Filter):
    """Drop the "Multiple AROUND hooks" notice for audited-safe targets only.

    Filters on the exact message prefix plus the target name, so an unrelated
    warning from the same logger still gets through. Enabled by default because
    the pairs it hides are audited; ``HCU_HOOK_KNOWN_PAIRS=0`` restores them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if not message.startswith("Multiple AROUND hooks on "):
            return True
        if any(f"on '{target}'" in message for target in SAFE_AROUND_PAIRS):
            return False
        return True


def _install_known_pair_filter(registry_module) -> None:
    """Attach the filter to the SGLang hook_registry logger once."""
    if os.environ.get("HCU_HOOK_KNOWN_PAIRS", "1") == "0":
        logger.debug("hcu hook guard: known-pair silencing disabled")
        return
    hook_logger = logging.getLogger(registry_module.__name__)
    if any(isinstance(f, _KnownPairFilter) for f in hook_logger.filters):
        return
    hook_logger.addFilter(_KnownPairFilter())


def install() -> None:
    """Install the guard once per process; no-op when disabled."""
    global _installed
    if _installed:
        return
    if os.environ.get("HCU_HOOK_GUARD", "1") == "0":
        logger.warning("hcu hook guard disabled by HCU_HOOK_GUARD=0")
        return

    from sglang.srt.plugins import hook_registry

    _install_known_pair_filter(hook_registry)

    registry = hook_registry.HookRegistry
    # Both members are classmethods in SGLang, so attribute access yields a
    # *bound* method. Reach through __func__ for the raw callable, otherwise
    # re-invoking with cls would bind it twice.
    raw_apply_target = registry.__dict__["_apply_target"].__func__
    raw_apply_hooks = registry.__dict__["apply_hooks"].__func__
    if getattr(raw_apply_target, "_hcu_guarded", False):
        return

    def guarded_apply_target(cls, target, hooks):
        try:
            return raw_apply_target(cls, target, hooks)
        except Exception as exc:
            _skipped[target] = type(exc).__name__
            logger.debug("hcu hook guard: skip %s (%s)", target, exc)
            return None

    def guarded_apply_hooks(cls):
        try:
            return raw_apply_hooks(cls)
        finally:
            _report()

    guarded_apply_target._hcu_guarded = True
    registry._apply_target = classmethod(guarded_apply_target)
    registry.apply_hooks = classmethod(guarded_apply_hooks)
    _installed = True
    logger.debug("hcu hook guard installed")
