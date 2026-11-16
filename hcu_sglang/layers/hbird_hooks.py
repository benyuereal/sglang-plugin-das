"""Install the hbird GEMM hooks: model conversion and the lm_head fast path.

The linear-layer seam itself comes from hbird-gemm's sitecustomize
(``HBIRD_GEMM_SGLANG_SHIM=1`` + ``HBIRD_GEMM_HOME/plugins`` on PYTHONPATH).
These two hooks are the pieces 0.5.18 patched into the source tree.

Importing this module never requires hbird-gemm; when the backend is unset or
absent, ``register`` leaves the model on the upstream GEMM.
"""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from hcu_sglang.constants import BYTES_PER_GIB

from . import gemm

_LOG = logging.getLogger(__name__)
_installed = False


def _around_load_model(original, self):
    out = original(self)
    _convert_model(self.model, self)
    return out


def _convert_model(model, runner):
    """Pack the loaded model; a failure here must never take serving down."""
    try:
        result = gemm.convert_model(model, runner)
        if result:
            _LOG.info(
                "[hbird_bf16] packed=%d freed=%.2fGiB kept=%d skipped=%d ws=%d",
                result["converted"],
                result["freed"] / BYTES_PER_GIB,
                result["kept"],
                len(result["skipped"]),
                result["prepared"],
            )
    except Exception:  # noqa: BLE001 - optional path, never take serving down
        _LOG.warning("[hbird_bf16] convert_model failed", exc_info=True)


def _around_compute_lm_head(
    original, self, hidden_states, lm_head, embedding_bias=None
):
    out = _lm_head(
        hidden_states, lm_head, embedding_bias, bool(self.use_fp32_lm_head)
    )
    if out is not None:
        return out
    return original(self, hidden_states, lm_head, embedding_bias)


def _lm_head(hidden_states, lm_head, embedding_bias, use_fp32_lm_head):
    """The provider's logits, or None when the original path should run."""
    try:
        return gemm.lm_head_linear(
            hidden_states, lm_head, embedding_bias, use_fp32_lm_head
        )
    except Exception:  # noqa: BLE001
        return None


def _install_shim():
    """Install hbird-gemm's own seam; the import stays here so it is optional."""
    try:
        import hbird_sglang

        ok = hbird_sglang.install()
        _LOG.info("[hbird_bf16] hbird_sglang.install -> %s", ok)
    except Exception:  # noqa: BLE001 - optional path
        _LOG.warning("[hbird_bf16] hbird_sglang.install failed", exc_info=True)


def _install_hooks():
    for target, hook in (
        (
            "sglang.srt.model_executor.model_runner.ModelRunner.load_model",
            _around_load_model,
        ),
        (
            "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head",
            _around_compute_lm_head,
        ),
    ):
        HookRegistry.register(target, hook, HookType.AROUND)


def register() -> bool:
    global _installed
    if _installed:
        return True
    if not gemm.enabled():
        return False
    _install_shim()
    _install_hooks()
    _installed = True
    _LOG.info("[hbird_bf16] prepare_model and lm_head hooks registered")
    return True
