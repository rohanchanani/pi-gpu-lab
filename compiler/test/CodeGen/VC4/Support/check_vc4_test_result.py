#!/usr/bin/env python3
"""Check a VC4 hardware-test run log against a small JSON oracle.

The checker looks for the last line containing tokens of the form:

    VC4_TEST_RESULT key=value key=value ...

It compares that parsed result against an expected JSON document.  The intended
use is hardware golden tests where the log is noisy but the final result line is
stable and machine-readable.

Expected JSON example:

{
  "name": "saxpy_reference",
  "status": "PASS",
  "required": {"mismatches": 0},
  "float_max": {"max_abs_diff": 0.0001}
}

Top-level `name` and `status` are treated as exact required fields.  Values in
`required` are compared after coercing the actual string to the type of the
expected value.  Values in `float_max` require abs(actual) <= the expected max.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any, Dict, Iterable, Tuple

MARKER = "VC4_TEST_RESULT"


class CheckError(Exception):
    pass


def _parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if text.startswith(("0x", "0X")):
            return int(text, 16)
        return int(text, 10)
    except ValueError:
        pass
    try:
        value = float(text)
        if math.isfinite(value):
            return value
        return text
    except ValueError:
        return text


def _coerce_to_expected(actual_text: str, expected: Any) -> Any:
    if isinstance(expected, bool):
        lowered = actual_text.lower()
        if lowered not in {"true", "false", "0", "1"}:
            raise CheckError(f"cannot coerce {actual_text!r} to bool")
        return lowered in {"true", "1"}
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual_text, 0)
        except ValueError as exc:
            raise CheckError(f"cannot coerce {actual_text!r} to int") from exc
    if isinstance(expected, float):
        try:
            return float(actual_text)
        except ValueError as exc:
            raise CheckError(f"cannot coerce {actual_text!r} to float") from exc
    return actual_text


def parse_result_line(line: str) -> Dict[str, str]:
    """Parse one VC4_TEST_RESULT line into a string dictionary."""
    if MARKER not in line:
        raise CheckError(f"line does not contain {MARKER}")
    suffix = line.split(MARKER, 1)[1].strip()
    if not suffix:
        raise CheckError("result line has no key=value fields")

    fields: Dict[str, str] = {}
    for token in shlex.split(suffix):
        if "=" not in token:
            raise CheckError(f"malformed token {token!r}; expected key=value")
        key, value = token.split("=", 1)
        if not key:
            raise CheckError(f"empty key in token {token!r}")
        fields[key] = value
    return fields


def find_last_result(log_text: str) -> Tuple[str, Dict[str, str]]:
    last_line = None
    for line in log_text.splitlines():
        if MARKER in line:
            last_line = line.strip()
    if last_line is None:
        raise CheckError(f"no {MARKER} line found in log")
    return last_line, parse_result_line(last_line)


def _iter_required(expected: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    if "name" in expected:
        yield "name", expected["name"]
    if "status" in expected:
        yield "status", expected["status"]
    required = expected.get("required", {})
    if not isinstance(required, dict):
        raise CheckError("expected['required'] must be a dictionary")
    for key, value in required.items():
        yield key, value


def check_result(expected: Dict[str, Any], actual: Dict[str, str]) -> None:
    errors = []

    for key, expected_value in _iter_required(expected):
        if key not in actual:
            errors.append(f"missing required field {key!r}")
            continue
        try:
            actual_value = _coerce_to_expected(actual[key], expected_value)
        except CheckError as exc:
            errors.append(str(exc))
            continue
        if actual_value != expected_value:
            errors.append(
                f"field {key!r}: expected {expected_value!r}, got {actual_value!r}"
            )

    float_max = expected.get("float_max", {})
    if not isinstance(float_max, dict):
        errors.append("expected['float_max'] must be a dictionary")
    else:
        for key, max_value in float_max.items():
            if key not in actual:
                errors.append(f"missing float_max field {key!r}")
                continue
            try:
                actual_float = float(actual[key])
                limit = float(max_value)
            except ValueError:
                errors.append(f"field {key!r}: expected numeric value, got {actual[key]!r}")
                continue
            if not math.isfinite(actual_float):
                errors.append(f"field {key!r}: value is not finite: {actual_float!r}")
                continue
            if abs(actual_float) > limit:
                errors.append(
                    f"field {key!r}: abs({actual_float!r}) exceeds max {limit!r}"
                )

    if errors:
        raise CheckError("; ".join(errors))


def load_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CheckError(f"failed to parse JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckError(f"expected JSON object in {path}")
    return data


def run_check(expected_path: Path, log_path: Path) -> int:
    expected = load_json(expected_path)
    log_text = log_path.read_text(errors="replace")
    line, actual = find_last_result(log_text)
    check_result(expected, actual)
    print(f"ok: matched {MARKER}: {line}")
    return 0


def self_test() -> int:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        expected = base / "expected.json"
        log = base / "run.log"
        expected.write_text(
            json.dumps(
                {
                    "name": "self_test",
                    "status": "PASS",
                    "required": {"mismatches": 0, "count": 12},
                    "float_max": {"max_abs_diff": 0.001},
                }
            )
        )
        log.write_text(
            "noise before\n"
            "VC4_TEST_RESULT name=old status=FAIL mismatches=99 max_abs_diff=1.0\n"
            "more noise\n"
            "VC4_TEST_RESULT name=self_test status=PASS mismatches=0 count=12 max_abs_diff=0.0005\n"
        )
        run_check(expected, log)

        failing_expected = base / "failing_expected.json"
        failing_expected.write_text(
            json.dumps(
                {
                    "name": "self_test",
                    "status": "PASS",
                    "required": {"mismatches": 1},
                }
            )
        )
        try:
            run_check(failing_expected, log)
        except CheckError:
            print("ok: negative self-test failed as expected")
        else:
            raise CheckError("negative self-test unexpectedly passed")

    print("self-test PASS")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", nargs="?", type=Path, help="expected JSON file")
    parser.add_argument("log", nargs="?", type=Path, help="run log file")
    parser.add_argument("--self-test", action="store_true", help="run built-in tests")
    args = parser.parse_args(argv)

    try:
        if args.self_test:
            return self_test()
        if args.expected is None or args.log is None:
            parser.error("expected and log are required unless --self-test is used")
        return run_check(args.expected, args.log)
    except CheckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
