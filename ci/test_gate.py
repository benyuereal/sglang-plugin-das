#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run the unit suite and fail only on failures outside the recorded baseline.

``main`` carries failures that no plugin change can fix (stale provenance
pins and a source checkout the workspace does not have). They are recorded in
``tests/contracts/known_test_failures.json`` with their root cause, so this
gate answers the question a reviewer actually asks: *did this branch add a
failure?* It does not hide the raw suite; ``python3 -m unittest discover -s
tests`` still prints every failure.

Exit status: 0 when every failure is recorded, 1 when one is not.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/contracts/known_test_failures.json"


def _test_id(test) -> str:
    # A subTest id carries its parameters: drop them, the parent is the contract.
    return test.id().split(" (", 1)[0]


def main() -> int:
    baseline = json.loads(BASELINE.read_text())
    known = {row["id"]: row["reason"] for row in baseline["failures"]}
    sys.path.insert(0, str(ROOT))
    suite = unittest.TestLoader().discover(str(ROOT / "tests"))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)

    failures = {}
    for test, traceback in result.failures + result.errors:
        failures.setdefault(_test_id(test), traceback)
    new = sorted(set(failures) - set(known))
    fixed = sorted(set(known) - set(failures))

    print(
        f"{result.testsRun} tests: {len(failures)} failure(s), "
        f"{len(new)} new, {len(result.skipped)} skipped "
        f"(baseline {len(known)} on {baseline['recorded_on']})"
    )
    for test_id in new:
        print(f"\nNEW FAILURE: {test_id}\n{failures[test_id]}")
    for test_id in fixed:
        print(f"now passing, prune the baseline: {test_id}")
    if not new:
        print("no failure outside the recorded baseline")
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
