"""Install the hbird bf16 GEMM glue (prepare_model / lm_head fast path).

The linear-layer seam itself is installed by hbird-gemm's sitecustomize
(``HBIRD_GEMM_SGLANG_SHIM=1`` + ``HBIRD_GEMM_HOME/plugins`` on PYTHONPATH).
This module adds the two pieces that upstream 0.5.18 patched into the source
tree: the one-copy weight conversion after the model is loaded, and the
unquantized lm_head fast path.
"""

import logging
import os

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from . import hbird_bf16

_LOG = logging.getLogger(__name__)
_installed = False


def _enabled() -> bool:
    value = os.environ.get("HBIRD_GEMM_SGLANG", "0").strip().lower()
    return value not in ("", "0", "false", "off", "no")


def _around_load_model(original, self, *args, **kwargs):
    out = original(self, *args, **kwargs)
    try:
        result = hbird_bf16.prepare_model(getattr(self, "model", None), self)
        if result:
            print(
                "[hbird_bf16] packed=%d freed=%.2fGiB kept=%d skipped=%d ws=%d"
                % (
                    result["converted"],
                    result["freed"] / 2**30,
                    result["kept"],
                    len(result["skipped"]),
                    result["prepared"],
                ),
                flush=True,
            )
    except Exception:  # noqa: BLE001 - optional path, never take serving down
        _LOG.warning("hbird_bf16: prepare_model failed", exc_info=True)
    return out


def _around_compute_lm_head(
    original, self, hidden_states, lm_head, embedding_bias=None
):
    try:
        out = hbird_bf16.lm_head_matmul(
            hidden_states,
            lm_head,
            embedding_bias,
            bool(getattr(self, "use_fp32_lm_head", False)),
        )
    except Exception:  # noqa: BLE001
        out = None
    if out is not None:
        return out
    return original(self, hidden_states, lm_head, embedding_bias)


def register() -> bool:
    global _installed
    if _installed:
        return True
    if not _enabled():
        return False
    try:
        import hbird_sglang

        ok = hbird_sglang.install()
        print(f"[hbird_bf16] hbird_sglang.install -> {ok}", flush=True)
    except Exception:  # noqa: BLE001 - optional path
        _LOG.warning("hbird_bf16: hbird_sglang.install failed", exc_info=True)
    HookRegistry.register(
        "sglang.srt.model_executor.model_runner.ModelRunner.load_model",
        _around_load_model,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head",
        _around_compute_lm_head,
        HookType.AROUND,
    )
    _installed = True
    _LOG.info("hbird_bf16: prepare_model and lm_head hooks registered")
    return True
