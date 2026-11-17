# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pure PLE ngram quantization probe, extracted from ple.py so it stays
testable offline (no sglang/triton/host stack needed).

The probe naming is load-bearing (PR #29 follow-up): quant ignore/target
regexes in -ngram checkpoints are written against ``<ple>.ngram_embedding``
(and its ``.shard_N`` children), so the name must never double-append
``.ngram_embedding`` when the constructor prefix already ends with it.

The sglang matchers (``should_ignore_layer`` / ``check_equal_or_regex_match``)
are injected as callables so this module carries no sglang import; the
production call site in ple.py passes the real ones.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple


def ngram_probe_name(prefix: str) -> str:
    """Map the NGramMethods constructor prefix to the ngram table name that
    quant ignore/target regexes are written against."""
    if prefix.endswith("ngram_embedding"):
        return prefix
    return f"{prefix}.ngram_embedding" if prefix else "ngram_embedding"


def probe_ngram_quant_flags(
    prefix: str,
    quant_config,
    config,
    should_ignore_layer: Callable[..., bool],
    check_equal_or_regex_match: Callable[..., bool],
) -> Tuple[bool, bool, bool]:
    """Return (ple_not_ignored, ple_int8, ple_fp8_channelwise).

    Verbatim semantics of the probe block in NGramMethods.__init__ (ple.py);
    keep the two in sync when touching either.
    """
    ple_int8 = False
    ple_fp8_channelwise = False
    ple_not_ignored = False
    # Constructor prefix is the PLE module ("...ple.ple_embedding").
    # Quant ignore/target regexes are written against the ngram table.
    ngram_name = ngram_probe_name(prefix)
    if quant_config is not None and quant_config.get_name() == "compressed_tensors":
        ignore = getattr(quant_config, "ignore", ())
        # Match the shard naming used in -ngram checkpoints. Using
        # ".weight" is wrong for FP8-ngram ignore regexes that only carve
        # out "...ngram_embedding.shard_N" without a trailing ".weight".
        ple_not_ignored = not should_ignore_layer(
            f"{ngram_name}.shard_0", ignore
        )
        if ple_not_ignored:
            targets = getattr(quant_config, "target_scheme_map", {})
            for target, scheme in targets.items():
                if not check_equal_or_regex_match(ngram_name, (target,)):
                    continue
                weight_args = scheme.get("weights") if scheme else None
                weight_type = getattr(weight_args, "type", None)
                weight_type = getattr(weight_type, "value", weight_type)
                ple_int8 = (
                    getattr(weight_args, "num_bits", None) == 8
                    and weight_type == "int"
                )
                break
            # FP8-ngram checkpoints leave ngram_embedding shards un-ignored
            # but only list Linear in config_groups.targets.
            if not ple_int8:
                ple_fp8_channelwise = True
    if (
        not ple_int8
        and quant_config is not None
        and quant_config.get_name()
        in ("slimquant_w4a8_marlin", "slimquant_w4a8")
    ):
        comp = getattr(config, "compression_config", None) or {}
        mixed = comp.get("mixed_precision") if isinstance(comp, dict) else {}
        ngram = (mixed or {}).get("ngram_embedding") or {}
        weight_bits = ngram.get("weight_bits")
        if weight_bits == 8 or weight_bits == "8":
            ignore = getattr(quant_config, "ignore", ()) or ()
            ple_not_ignored = not should_ignore_layer(
                f"{ngram_name}.shard_0", ignore
            )
            if ple_not_ignored:
                ple_int8 = True
    return ple_not_ignored, ple_int8, ple_fp8_channelwise
