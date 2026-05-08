#!/usr/bin/env python3
"""Check VC4_TEST_RESULT output against a hardware-run expected.json file."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

PREFIX = "VC4_TEST_RESULT "


def fail(message: str) -> None:
    print(f"[vc4-result] ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_result_line(line: str) -> Dict[str, str]:
    if not line.startswith(PREFIX):
        fail("internal error: result line missing prefix")
    fields: Dict[str, str] = {}
    for token in line[len(PREFIX) :].strip().split():
        if "=" not in token:
            fail(f"malformed VC4_TEST_RESULT token: {token!r}")
        key, value = token.split("=", 1)
        if not key or key in fields:
            fail(f"malformed or duplicate VC4_TEST_RESULT key: {key!r}")
        fields[key] = value
    return fields


def load_last_result(log_path: Path) -> Dict[str, str]:
    last = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(PREFIX):
            last = line
    if last is None:
        fail(f"no VC4_TEST_RESULT line found in {log_path}")
    return parse_result_line(last)


def coerce_exact(actual: str, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual.lower() == ("true" if expected else "false")
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual, 0) == expected
        except ValueError:
            return False
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=0.0)
        except ValueError:
            return False
    return actual == str(expected)


def check_expected(expected_path: Path, log_path: Path) -> None:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if not isinstance(expected, dict):
        fail("expected.json must contain an object")

    result = load_last_result(log_path)

    for key in ("name", "status"):
        if key not in expected:
            fail(f"expected.json missing required top-level key {key!r}")
        if key not in result:
            fail(f"VC4_TEST_RESULT missing required key {key!r}")
        if result[key] != str(expected[key]):
            fail(f"{key} mismatch: expected {expected[key]!r}, got {result[key]!r}")

    required = expected.get("required", {})
    if required is None:
        required = {}
    if not isinstance(required, dict):
        fail("expected.json 'required' must be an object when present")
    for key, value in required.items():
        if key not in result:
            fail(f"VC4_TEST_RESULT missing required field {key!r}")
        if not coerce_exact(result[key], value):
            fail(f"field {key!r} mismatch: expected {value!r}, got {result[key]!r}")

    float_max = expected.get("float_max", {})
    if float_max is None:
        float_max = {}
    if not isinstance(float_max, dict):
        fail("expected.json 'float_max' must be an object when present")
    for key, limit in float_max.items():
        if key not in result:
            fail(f"VC4_TEST_RESULT missing float field {key!r}")
        try:
            actual = float(result[key])
            max_value = float(limit)
        except (TypeError, ValueError):
            fail(f"float_max field {key!r} is not numeric")
        if abs(actual) > max_value:
            fail(f"float field {key!r}={actual!r} exceeds limit {max_value!r}")

    print(f"[vc4-result] PASS {expected_path} {log_path}")


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        expected = tmp / "expected.json"
        log = tmp / "run.log"
        expected.write_text(
            json.dumps(
                {
                    "name": "runner_self_test",
                    "status": "PASS",
                    "required": {"mismatches": 0, "count": 1},
                    "float_max": {"max_abs_diff": 0.001},
                }
            ),
            encoding="utf-8",
        )
        log.write_text(
            "noise\n"
            "VC4_TEST_RESULT name=runner_self_test status=PASS mismatches=0 count=1 max_abs_diff=0.0005\n",
            encoding="utf-8",
        )
        check_expected(expected, log)
    print("check_vc4_test_result.py self-test PASS")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if len(argv) != 3:
        print("usage: check_vc4_test_result.py EXPECTED_JSON RUN_LOG", file=sys.stderr)
        print("       check_vc4_test_result.py --self-test", file=sys.stderr)
        return 2
    check_expected(Path(argv[1]), Path(argv[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
