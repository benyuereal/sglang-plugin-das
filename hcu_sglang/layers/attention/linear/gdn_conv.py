# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""causal-conv1d update dispatch for GDN decode.

The vendored forward calls ``update``. It tries lightop's HIP twin when
``SGLANG_GDN_HIP`` includes ``conv``; a decline or a failure falls back to the
update the upstream tree would have picked (the HCU twin, or Triton).

This seam imports neither sglang nor lightop eagerly, so the fallback contract
is testable without a device.
"""

from __future__ import annotations

import logging

from hcu_sglang.layers.attention.linear import gdn_recurrent

logger = logging.getLogger(__name__)

_state = {"is_hcu": False, "hcu": None, "triton": None, "log": logger.info}
_logged = {"done": False}


def configure(*, is_hcu, triton, hcu=None, log=None):
    """Bind the upstream updates; called once when the overlay module loads."""
    _state["is_hcu"] = is_hcu
    _state["triton"] = triton
    _state["hcu"] = hcu
    if log is not None:
        _state["log"] = log


def update(x, conv_state, weight, bias=None, activation=None, **kwargs):
    """The HIP conv update, or the upstream one when it declines."""
    if _state["is_hcu"] and gdn_recurrent.conv_enabled():
        try:
            out = gdn_recurrent.conv_update(
                x, conv_state, weight, bias, activation, **kwargs
            )
            if out is not None:
                return out
            if not _logged["done"]:
                _logged["done"] = True
                _state["log"](
                    "SGLANG_GDN_HIP includes conv but this call is outside "
                    f"the HIP kernel ({gdn_recurrent.conv_last_reason()}); "
                    "using the upstream update."
                )
        except Exception as exc:  # noqa: BLE001 - never fail a forward here
            if not _logged["done"]:
                _logged["done"] = True
                _state["log"](f"SGLANG_GDN_HIP conv path disabled: {exc}")
    updater = _state["hcu"] or _state["triton"]
    if updater is None:
        raise RuntimeError("gdn_conv.update() used before configure()")
    return updater(x, conv_state, weight, bias, activation, **kwargs)
