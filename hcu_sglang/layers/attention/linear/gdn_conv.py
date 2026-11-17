# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""causal-conv1d update dispatch for GDN decode and verify.

``update`` is the decode path: try lightop's HIP conv, fall back to the
upstream update on a decline or failure. ``update_required`` is the verify
path: verify passes intermediate-window arguments the upstream update does not
accept, so a decline is an error instead of a fallback. This seam imports
neither sglang nor lightop eagerly, so the contract is testable without a
device.
"""

from __future__ import annotations

import logging

from hcu_sglang.layers.attention.linear import gdn_recurrent

logger = logging.getLogger(__name__)

_state = {"triton": None, "hcu": None, "log": logger.info,
          "logged": False, "last": ""}


def configure(*, triton, hcu=None, log=None):
    """Bind the upstream updates; called once when the overlay module loads."""
    _state["triton"] = triton
    _state["hcu"] = hcu
    if log is not None:
        _state["log"] = log


def _attempt(x, conv_state, weight, bias, activation, kwargs):
    """The lightop result, or None with the reason recorded."""
    if not gdn_recurrent.overlay_enabled():
        _state["last"] = "lightop is off"
        return None
    try:
        out = gdn_recurrent.conv_update(
            x, conv_state, weight, bias, activation, **kwargs
        )
    except Exception as exc:  # noqa: BLE001 - the fallback is always correct
        _state["last"] = str(exc)
        return None
    if out is None:
        _state["last"] = gdn_recurrent.conv_last_reason() or "declined"
    return out


def update(x, conv_state, weight, bias=None, activation=None, **kwargs):
    """The HIP conv update, or the upstream one when it declines."""
    out = _attempt(x, conv_state, weight, bias, activation, kwargs)
    if out is not None:
        return out
    if not _state["logged"]:
        _state["logged"] = True
        _state["log"](
            "SGLANG_GDN_HIP conv fell back to the upstream update "
            f"({_state['last'] or 'declined'})"
        )
    updater = _state["hcu"] or _state["triton"]
    if updater is None:
        raise RuntimeError("gdn_conv used before configure()")
    return updater(x, conv_state, weight, bias, activation, **kwargs)


def update_required(x, conv_state, weight, bias=None, activation=None,
                    **kwargs):
    """The HIP conv update; verify cannot use the upstream signature."""
    out = _attempt(x, conv_state, weight, bias, activation, kwargs)
    if out is not None:
        return out
    raise RuntimeError(
        f"GDN verify needs the HIP conv update ({_state['last'] or 'declined'})"
    )
