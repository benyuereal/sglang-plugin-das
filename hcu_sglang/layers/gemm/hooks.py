"""Register the alternative GEMM: weight conversion and the lm_head fast path.

The linear-layer seam itself comes from the selected backend, which may ship
its own sitecustomize (hbird-gemm uses ``HBIRD_GEMM_SGLANG_SHIM=1`` and
``HBIRD_GEMM_HOME/plugins`` on PYTHONPATH). These two hooks are the pieces
0.5.18 patched into the source tree.

Importing this module never requires the private backend; when it is unset
or absent, ``register`` leaves the model on the upstream GEMM.
"""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from hcu_sglang.constants import BYTES_PER_GIB

from hcu_sglang.layers.gemm import backend as gemm

_LOG = logging.getLogger(__name__)
_installed = False


def _load_model(wrapped, self):
    out = wrapped(self)
    _convert_model(self.model, self)
    return out


def _convert_model(model, runner):
    """Pack the loaded model; a failure here must never take serving down."""
    try:
        result = gemm.convert_model(model, runner)
        if result:
            _LOG.info(
                "[gemm] packed=%d freed=%.2fGiB kept=%d skipped=%d ws=%d",
                result["converted"],
                result["freed"] / BYTES_PER_GIB,
                result["kept"],
                len(result["skipped"]),
                result["prepared"],
            )
    except Exception:  # noqa: BLE001 - optional path, never take serving down
        _LOG.warning("[gemm] convert_model failed", exc_info=True)


def _compute_lm_head(
    wrapped, self, hidden_states, lm_head, embedding_bias=None
):
    out = _provider_logits(
        hidden_states, lm_head, embedding_bias, bool(self.use_fp32_lm_head)
    )
    if out is not None:
        return out
    return wrapped(self, hidden_states, lm_head, embedding_bias)


def _provider_logits(hidden_states, lm_head, embedding_bias, use_fp32_lm_head):
    """The provider's logits, or None when the original path should run."""
    try:
        return gemm.lm_head_linear(
            hidden_states, lm_head, embedding_bias, use_fp32_lm_head
        )
    except Exception:  # noqa: BLE001
        return None


def _install_shim():
    """Install the selected backend's own seam, if it has one."""
    try:
        ok = gemm.install_shim()
        _LOG.info("[gemm] install_shim -> %s", ok)
    except Exception:  # noqa: BLE001 - optional path
        _LOG.warning("[gemm] install_shim failed", exc_info=True)


def _install_hooks():
    for target, hook in (
        (
            "sglang.srt.model_executor.model_runner.ModelRunner.load_model",
            _load_model,
        ),
        (
            "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head",
            _compute_lm_head,
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
    _LOG.info("[gemm] prepare_model and lm_head hooks registered")
    return True
