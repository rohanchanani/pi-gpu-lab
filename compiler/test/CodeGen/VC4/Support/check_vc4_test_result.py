#!/usr/bin/env python3
"""Check VC4_TEST_RESULT output against a hardware-run expected.json file.

The checker is intentionally small and deterministic.  It accepts the expected
schema used by VC4 hardware ground-truth tests:

  {
    "name": "test_name",
    "status": "PASS",
    "required": {"field": exact_value, ...},
    "float_max": {"field": max_allowed_value, ...},
    "float_min": {"field": min_allowed_value, ...},
    "float_abs": {"field": max_abs_allowed_value, ...}
  }

Only `name` and `status` are intrinsic.  `required` fields are exact checks.
Tolerance maps are optional and checked only when present.  The result line may
be either JSON after VC4_TEST_RESULT or whitespace/comma-separated key=value
fields.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


class CheckError(RuntimeError):
    pass


def coerce_scalar(text: str) -> Any:
    value = text.strip().strip('"')
    if re.fullmatch(r"[-+]?\d+", value):
        try:
            return int(value, 10)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def parse_key_value_tail(tail: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for part in re.split(r"[\s,]+", tail.strip()):
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if not key:
            continue
        fields[key] = coerce_scalar(value)
    return fields


def parse_result_line(line: str) -> dict[str, Any] | None:
    if "VC4_TEST_RESULT" not in line:
        return None
    tail = line.split("VC4_TEST_RESULT", 1)[1].strip(" :\t")
    if not tail:
        return None
    if tail.startswith("{"):
        try:
            parsed = json.loads(tail)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    fields = parse_key_value_tail(tail)
    return fields or None


def parse_last_result(log_text: str) -> tuple[dict[str, Any], str]:
    parsed: dict[str, Any] | None = None
    raw_line = ""
    for line in log_text.splitlines():
        maybe = parse_result_line(line)
        if maybe is not None:
            parsed = maybe
            raw_line = line.strip()
    if parsed is None:
        raise CheckError("no VC4_TEST_RESULT line found")
    return parsed, raw_line


def values_equal_exact(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        if isinstance(actual, str):
            return actual.lower() in {"1", "true", "yes"} if expected else actual.lower() in {"0", "false", "no"}
        return bool(actual) is expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            return int(actual) == expected and float(actual) == float(expected)
        if isinstance(actual, str) and re.fullmatch(r"[-+]?\d+", actual):
            return int(actual) == expected
        return False
    if isinstance(expected, float):
        try:
            return float(actual) == expected
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def as_float_field(parsed: dict[str, Any], key: str) -> float:
    if key not in parsed:
        raise CheckError(f"missing numeric field {key!r}")
    try:
        value = float(parsed[key])
    except (TypeError, ValueError) as exc:
        raise CheckError(f"field {key!r} is not numeric: {parsed[key]!r}") from exc
    if not math.isfinite(value):
        raise CheckError(f"field {key!r} is not finite: {parsed[key]!r}")
    return value


def check_expected(expected: dict[str, Any], parsed: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    expected_name = expected.get("name")
    if expected_name is not None:
        actual_name = parsed.get("name")
        if actual_name != expected_name:
            errors.append(f"name mismatch: expected {expected_name!r}, got {actual_name!r}")

    expected_status = expected.get("status", "PASS")
    actual_status = parsed.get("status")
    if actual_status != expected_status:
        errors.append(f"status mismatch: expected {expected_status!r}, got {actual_status!r}")

    required = expected.get("required", {})
    if required is None:
        required = {}
    if not isinstance(required, dict):
        errors.append("expected.json field 'required' must be an object when present")
    else:
        for key, expected_value in required.items():
            if key not in parsed:
                errors.append(f"missing required field {key!r}")
                continue
            actual = parsed[key]
            if not values_equal_exact(actual, expected_value):
                errors.append(f"field {key!r} mismatch: expected {expected_value!r}, got {actual!r}")

    float_max = expected.get("float_max", {}) or {}
    if not isinstance(float_max, dict):
        errors.append("expected.json field 'float_max' must be an object when present")
    else:
        for key, max_value in float_max.items():
            try:
                actual = as_float_field(parsed, key)
                limit = float(max_value)
            except CheckError as exc:
                errors.append(str(exc))
                continue
            except (TypeError, ValueError):
                errors.append(f"float_max limit for {key!r} is not numeric: {max_value!r}")
                continue
            if actual > limit:
                errors.append(f"field {key!r} exceeds float_max: actual {actual} > limit {limit}")

    float_min = expected.get("float_min", {}) or {}
    if not isinstance(float_min, dict):
        errors.append("expected.json field 'float_min' must be an object when present")
    else:
        for key, min_value in float_min.items():
            try:
                actual = as_float_field(parsed, key)
                limit = float(min_value)
            except CheckError as exc:
                errors.append(str(exc))
                continue
            except (TypeError, ValueError):
                errors.append(f"float_min limit for {key!r} is not numeric: {min_value!r}")
                continue
            if actual < limit:
                errors.append(f"field {key!r} is below float_min: actual {actual} < limit {limit}")

    float_abs = expected.get("float_abs", {}) or {}
    if not isinstance(float_abs, dict):
        errors.append("expected.json field 'float_abs' must be an object when present")
    else:
        for key, max_abs in float_abs.items():
            try:
                actual = as_float_field(parsed, key)
                limit = float(max_abs)
            except CheckError as exc:
                errors.append(str(exc))
                continue
            except (TypeError, ValueError):
                errors.append(f"float_abs limit for {key!r} is not numeric: {max_abs!r}")
                continue
            if abs(actual) > limit:
                errors.append(f"field {key!r} exceeds float_abs: |{actual}| > limit {limit}")

    return errors


def read_expected(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckError(f"could not read expected.json {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CheckError(f"expected.json must contain an object: {path}")
    return data


def read_logs(paths: Iterable[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        if not path.exists():
            raise CheckError(f"log path does not exist: {path}")
        parts.append(f"\n# BEGIN LOG {path}\n")
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
        parts.append(f"\n# END LOG {path}\n")
    return "".join(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_json", type=Path)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--dump-parsed", action="store_true", help="print parsed result JSON even on success")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        expected = read_expected(args.expected_json)
        parsed, raw_line = parse_last_result(read_logs(args.logs))
        errors = check_expected(expected, parsed)
    except CheckError as exc:
        print(f"[vc4-result] FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"[vc4-result] result line: {raw_line}")
    if args.dump_parsed or errors:
        print("[vc4-result] parsed:")
        print(json.dumps(parsed, indent=2, sort_keys=True))
        print("[vc4-result] expected:")
        print(json.dumps(expected, indent=2, sort_keys=True))

    if errors:
        print("[vc4-result] FAIL:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("[vc4-result] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
