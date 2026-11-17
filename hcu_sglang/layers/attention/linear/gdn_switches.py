# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The GDN HIP switch: one env var, resolved once per process.

``SGLANG_GDN_HIP=1`` turns on the delivered stack: the B+ handoff verify,
the HIP conv update and the prenorm kernel. ``0`` / unset keeps the upstream
Triton dispatcher. There are no per-capability switches; a capability that
cannot run declines at runtime and the caller falls back.

The parsed mode is cached by raw value, so the steady-state hot path reads a
bool instead of re-parsing the environment.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

ENV = "SGLANG_GDN_HIP"

_ON = frozenset({"1", "all", "on", "true", "yes"})
_OFF = frozenset({"", "0", "off", "none", "false", "no"})
_LEGACY_PREFIX = "SGLANG_GDN_HIP_"

_LOG = logging.getLogger(__name__)
_cache: dict[str, "GdnMode"] = {}
_warned_legacy = False


class GdnMode(NamedTuple):
    """The immutable result of the one switch."""

    enabled: bool


def _parse(value: str) -> GdnMode:
    value = value.strip().lower()
    if value in _OFF:
        return GdnMode(False)
    if value not in _ON:
        _LOG.warning("[gdn] unknown %s=%r; treating it as on", ENV, value)
    return GdnMode(True)


def _warn_legacy() -> None:
    global _warned_legacy
    if _warned_legacy:
        return
    _warned_legacy = True
    legacy = sorted(
        name for name in os.environ
        if name.startswith(_LEGACY_PREFIX) and name != ENV
    )
    if legacy:
        _LOG.warning(
            "[gdn] %s are ignored; set %s=1 instead", ", ".join(legacy), ENV
        )


def mode() -> GdnMode:
    _warn_legacy()
    value = os.environ.get(ENV, "")
    parsed = _cache.get(value)
    if parsed is None:
        parsed = _cache[value] = _parse(value)
    return parsed


def enabled() -> bool:
    return mode().enabled
