# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The GDN HIP switch surface: one env var, capabilities as data.

``SGLANG_GDN_HIP`` holds a comma-separated capability list:

- ``all`` (or ``1``): the delivered stack -- conv, prenorm, kernel, bplus,
  handoff. ``bplus`` implies ``handoff``.
- ``0`` / unset / ``off``: none of it; upstream Triton stays.
- a subset, for an isolation arm: ``bplus``, ``conv,prenorm``, ``kernel``.
- ``all,replay-all``: the debug arm that replays fully accepted rows too.

Every capability that needs the operator library still goes through
``gdn_recurrent.enabled()``; this module only parses the operator's choice.
"""

from __future__ import annotations

import logging
import os

ENV = "SGLANG_GDN_HIP"

CONV = "conv"
PRENORM = "prenorm"
KERNEL = "kernel"
BPLUS = "bplus"
HANDOFF = "handoff"
REPLAY_ALL = "replay-all"

_DELIVERED = frozenset({CONV, PRENORM, KERNEL, BPLUS, HANDOFF})
_KNOWN = _DELIVERED | {REPLAY_ALL}
_ON = {"1", "all", "on", "true", "yes"}
_OFF = {"0", "off", "none", "false", "no"}
_IMPLIED = {HANDOFF: BPLUS}
_LEGACY_PREFIX = "SGLANG_GDN_HIP_"

_LOG = logging.getLogger(__name__)
_cache: dict[str, frozenset[str]] = {}
_warned = False


def _parse(value: str) -> frozenset[str]:
    tokens = {
        token.strip().lower()
        for token in value.replace("+", ",").replace(" ", ",").split(",")
        if token.strip()
    }
    if not tokens or tokens & _OFF:
        return frozenset()
    if tokens & _ON:
        tokens |= _DELIVERED
    unknown = tokens - _KNOWN
    if unknown:
        _LOG.warning("[gdn] ignoring unknown %s token(s): %s", ENV, sorted(unknown))
        tokens -= unknown
    for capability, required in _IMPLIED.items():
        if capability in tokens:
            tokens.add(required)
    return frozenset(tokens)


def _warn_legacy() -> None:
    global _warned
    if _warned:
        return
    _warned = True
    legacy = sorted(
        name for name in os.environ
        if name.startswith(_LEGACY_PREFIX) and name != ENV
    )
    if legacy:
        _LOG.warning(
            "[gdn] %s are ignored; set %s=all (or a capability subset) instead",
            ", ".join(legacy), ENV,
        )


def active() -> frozenset[str]:
    _warn_legacy()
    value = os.environ.get(ENV, "")
    parsed = _cache.get(value)
    if parsed is None:
        parsed = _cache[value] = _parse(value)
    return parsed


def has(capability: str) -> bool:
    return capability in active()
