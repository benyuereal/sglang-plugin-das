# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Offline red-green tests for the PLE ngram quantization probe (PR #29
follow-up). CPU-only, no sglang/triton/GPU: the probe lives in
hcu_sglang.models.qwen.ple_probe with the sglang matchers injected, so these
tests exercise the naming rules and branch semantics directly.

One pit per test: the naming pit is that quant ignore/target regexes are
written against "<ple>.ngram_embedding(.shard_N)", so a prefix already ending
in "ngram_embedding" must NOT get a second ".ngram_embedding" appended (the
doubled probe name stops matching checkpoint regexes and an ignored table is
silently treated as quantized).
"""

import re
from types import SimpleNamespace

from hcu_sglang.models.qwen.ple_probe import (
    ngram_probe_name,
    probe_ngram_quant_flags,
)

REGULAR_PREFIX = "model.layers.0.ple.ple_embedding"
SUFFIXED_PREFIX = "model.ple.ngram_embedding"
EMPTY_PREFIX = ""


def _match(name, patterns):
    """Stand-in for sglang's check_equal_or_regex_match (subset used by the
    probe): exact equality or "re:"-prefixed fullmatch. Regex semantics beyond
    this stay sglang's contract; these tests are about naming/branching."""
    if isinstance(patterns, str):
        patterns = (patterns,)
    for key in patterns:
        if key.startswith("re:"):
            if re.fullmatch(key[len("re:"):], name):
                return True
        elif name == key:
            return True
    return False


def _should_ignore(name, ignore):
    if isinstance(ignore, str):
        ignore = (ignore,)
    return _match(name, tuple(ignore))


def _probe(prefix, quant, config=SimpleNamespace()):
    return probe_ngram_quant_flags(
        prefix,
        quant,
        config,
        should_ignore_layer=_should_ignore,
        check_equal_or_regex_match=_match,
    )


def _ct_quant(ignore=(), targets=None):
    return SimpleNamespace(
        get_name=lambda: "compressed_tensors",
        ignore=tuple(ignore),
        target_scheme_map=targets or {},
    )


def _slim_quant(ignore=(), name="slimquant_w4a8_marlin"):
    return SimpleNamespace(get_name=lambda: name, ignore=tuple(ignore))


def _slim_config(weight_bits=None):
    compression_config = {}
    if weight_bits is not None:
        compression_config = {
            "mixed_precision": {"ngram_embedding": {"weight_bits": weight_bits}}
        }
    return SimpleNamespace(compression_config=compression_config)


INT8_SCHEME = {"weights": SimpleNamespace(num_bits=8, type="int")}


def test_ngram_probe_name_covers_three_prefix_forms():
    assert ngram_probe_name(REGULAR_PREFIX) == (
        "model.layers.0.ple.ple_embedding.ngram_embedding"
    )
    assert ngram_probe_name(SUFFIXED_PREFIX) == "model.ple.ngram_embedding"
    assert ngram_probe_name(EMPTY_PREFIX) == "ngram_embedding"


def test_compressed_tensors_regular_prefix_int8_target():
    quant = _ct_quant(
        ignore=("re:.*lm_head.*",),
        targets={r"re:.*ple_embedding\.ngram_embedding": INT8_SCHEME},
    )
    assert _probe(REGULAR_PREFIX, quant) == (True, True, False)


def test_compressed_tensors_regular_prefix_fp8_fallback_and_ignored():
    # FP8-ngram: shard not ignored, but targets only list Linear layers.
    fp8 = _ct_quant(targets={r"re:.*\.linear_proj": INT8_SCHEME})
    assert _probe(REGULAR_PREFIX, fp8) == (True, False, True)
    # FP8-ngram with the table carved out via shard regex: nothing quantized.
    ignored = _ct_quant(ignore=(r"re:.*ngram_embedding\.shard_\d+",))
    assert _probe(REGULAR_PREFIX, ignored) == (False, False, False)


def test_compressed_tensors_suffixed_prefix_is_probed_as_is():
    # Pit: prefix already ends with "ngram_embedding" -- appending again
    # breaks both the ignore and the target regexes below.
    int8 = _ct_quant(targets={r"re:model\.ple\.ngram_embedding": INT8_SCHEME})
    assert _probe(SUFFIXED_PREFIX, int8) == (True, True, False)
    ignored = _ct_quant(
        ignore=(r"re:model\.ple\.ngram_embedding\.shard_\d+",),
        targets={r"re:model\.ple\.ngram_embedding": INT8_SCHEME},
    )
    assert _probe(SUFFIXED_PREFIX, ignored) == (False, False, False)


def test_compressed_tensors_empty_prefix():
    int8 = _ct_quant(targets={r"re:ngram_embedding": INT8_SCHEME})
    assert _probe(EMPTY_PREFIX, int8) == (True, True, False)
    ignored = _ct_quant(ignore=("ngram_embedding.shard_0",))
    assert _probe(EMPTY_PREFIX, ignored) == (False, False, False)


def test_compressed_tensors_unwraps_enum_weight_type():
    enum_int8 = {
        "weights": SimpleNamespace(num_bits=8, type=SimpleNamespace(value="int"))
    }
    quant = _ct_quant(targets={r"re:.*ngram_embedding": enum_int8})
    assert _probe(REGULAR_PREFIX, quant) == (True, True, False)
    enum_fp8 = {
        "weights": SimpleNamespace(num_bits=8, type=SimpleNamespace(value="float"))
    }
    quant = _ct_quant(targets={r"re:.*ngram_embedding": enum_fp8})
    assert _probe(REGULAR_PREFIX, quant) == (True, False, True)


def test_slimquant_int8_all_three_prefix_forms():
    config = _slim_config(weight_bits=8)
    assert _probe(REGULAR_PREFIX, _slim_quant(), config) == (True, True, False)
    assert _probe(SUFFIXED_PREFIX, _slim_quant(), config) == (True, True, False)
    assert _probe(EMPTY_PREFIX, _slim_quant(), config) == (True, True, False)
    # String form and the bare slimquant_w4a8 name accept int8 too.
    assert _probe(
        REGULAR_PREFIX, _slim_quant(), _slim_config(weight_bits="8")
    ) == (True, True, False)
    assert _probe(
        REGULAR_PREFIX, _slim_quant(name="slimquant_w4a8"), config
    ) == (True, True, False)


def test_slimquant_suffixed_prefix_ignore_and_non_int8_bits():
    config8 = _slim_config(weight_bits=8)
    # Pit mirror: shard-regex ignore must match the as-is probe name.
    ignored = _slim_quant(ignore=(r"re:model\.ple\.ngram_embedding\.shard_\d+",))
    assert _probe(SUFFIXED_PREFIX, ignored, config8) == (False, False, False)
    # 16-bit ngram table or no compression_config: probe stays fully off.
    assert _probe(
        REGULAR_PREFIX, _slim_quant(), _slim_config(weight_bits=16)
    ) == (False, False, False)
    assert _probe(REGULAR_PREFIX, _slim_quant(), _slim_config()) == (False, False, False)


def test_no_quant_config_keeps_all_flags_off():
    assert _probe(REGULAR_PREFIX, None) == (False, False, False)
