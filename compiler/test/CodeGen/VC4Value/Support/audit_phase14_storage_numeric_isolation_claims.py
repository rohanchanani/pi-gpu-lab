#!/usr/bin/env python3
"""Audit Phase 14 VC4Value storage/numeric hardware isolation claims."""

import argparse
import json
import sys
from pathlib import Path

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_FIXTURES = {
    "value_f16_load_f32_compute_store_f32_vc4value",
    "value_f32_compute_store_f16_vc4value",
    "value_f16_row_dot_f32_accum_vc4value",
    "value_f16_empty_repeat_vc4value",
    "phase14_static_staged_guards",
}
HARDWARE_FIXTURES = REQUIRED_FIXTURES - {"phase14_static_staged_guards"}
REQUIRED_CLAIMS = {
    "saw_value_f16_storage_load",
    "saw_value_f32_compute_after_f16_load",
    "saw_value_f16_storage_store",
    "saw_f16_storage_finite_policy",
    "saw_value_f16_gemv_input_storage",
    "saw_value_gemv_f32_accum",
    "row_padding_sentinels",
    "saw_repeat_invocation",
    "empty_launch_sentinel_preserve",
    "value_i32_to_f32_staged_by_lower_half_gap",
    "native_f16_static_reject",
    "bf16_fp8_static_reject",
    "int8_quant_static_reject",
    "fp_to_int_static_reject",
    "no_after_boot_timeout",
}


def fail(message: str) -> None:
    print(f"FAIL Phase 14 storage/numeric isolation claim audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def require_marker(text: str, marker: str, context: str) -> None:
    if marker not in text:
        fail(f"{context} missing marker {marker!r}")


def validate_static_reject(repo_root: Path, static_test: str) -> None:
    expected_diagnostics = {
        "invalid-i32-to-f32-cast-staged.mlir": "i32 to f32 numeric cast staged by lower-half gap",
        "invalid-native-f16-arithmetic.mlir": "native f16 arithmetic is staged",
        "invalid-bf16-fp8-storage.mlir": "bf16/fp8 storage is staged",
        "invalid-int8-quant-storage.mlir": "int8/int16 quantized storage is staged",
        "invalid-f32-to-i32-cast.mlir": "fp-to-int numeric cast is staged",
    }
    path = repo_root / "compiler/test/Conversion/VC4ValueToVC4Kernel" / static_test
    if not path.exists():
        fail(f"missing static reject test {static_test}")
    text = path.read_text()
    require_marker(text, "RUN: not", static_test)
    require_marker(text, expected_diagnostics.get(static_test, "expected-error"), static_test)


def validate_claim(repo_root: Path, name: str, claim: dict, required: dict, input_text: str, harness_text: str) -> str:
    claim_name = claim.get("claim")
    kind = claim.get("kind")
    if kind not in CLAIM_KINDS:
        fail(f"{name}:{claim_name} invalid kind {kind!r}")
    if "expected_field" in claim and required.get(claim["expected_field"]) != 1:
        fail(f"{name}:{claim_name} expected field {claim['expected_field']} must be 1")
    for field in claim.get("expected_zero_fields", []):
        if required.get(field) != 0:
            fail(f"{name}:{claim_name} expected zero field {field} must be 0")
    if "input_marker" in claim:
        require_marker(input_text, claim["input_marker"], f"{name}:{claim_name} input")
    if "oracle_marker" in claim:
        require_marker(harness_text, claim["oracle_marker"], f"{name}:{claim_name} harness")
    for forbidden in claim.get("forbidden_input_markers", []):
        if forbidden in input_text:
            fail(f"{name}:{claim_name} forbidden input marker {forbidden!r}")
    for static_test in claim.get("static_reject_tests", []):
        validate_static_reject(repo_root, static_test)
    return claim_name


def validate_hardware_fixture(repo_root: Path, fixture: dict) -> set[str]:
    name = fixture.get("name")
    if name not in HARDWARE_FIXTURES:
        fail(f"unexpected hardware fixture {name!r}")
    fixture_dir = repo_root / fixture["path"]
    input_path = fixture_dir / "input.mlir"
    expected_path = fixture_dir / "expected.json"
    harness_path = fixture_dir / "candidate" / f"{name}_candidate_harness.c"
    for path in (fixture_dir, input_path, expected_path, harness_path):
        if not path.exists():
            fail(f"{name} missing required path {path}")

    input_text = input_path.read_text()
    harness_text = harness_path.read_text()
    expected = load_json(expected_path)
    required = expected.get("required", {})
    if expected.get("status") != "PASS":
        fail(f"{name} expected.json must require PASS")
    for field in ("total_mismatches", "sentinel_mismatches", "launch_failures"):
        if required.get(field) != 0:
            fail(f"{name} expected.json must require {field}=0")
    if required.get("active_qpus") != 12:
        fail(f"{name} expected.json must require active_qpus=12")
    if required.get("lanes") != 16:
        fail(f"{name} expected.json must require lanes=16")
    if required.get("output_hash_nonzero") != 1:
        fail(f"{name} expected.json must require output_hash_nonzero=1")
    if required.get("runtime_launches", 0) < 8:
        fail(f"{name} expected.json must require at least eight runtime launches")
    require_marker(harness_text, "#define ACTIVE_QPUS 12u", f"{name} harness")
    require_marker(harness_text, "VC4_TEST_RESULT", f"{name} harness")

    for forbidden in ("tt.", "ttg.", "vc4kernel.", "ssavc4.", "vc4.qpu.", "vector.contract", "arith.fptosi", "arith.fptoui", "arith.sitofp", "math."):
        if forbidden in input_text:
            fail(f"{name} input contains forbidden marker {forbidden}")
    for line in input_text.splitlines():
        if "arith.addf" in line and "vector<16xf16>" in line:
            fail(f"{name} input contains native f16 arithmetic")
        if "arith.mulf" in line and "vector<16xf16>" in line:
            fail(f"{name} input contains native f16 arithmetic")
    if "vector<16x" not in input_text:
        fail(f"{name} input must use vector<16xT> fragments")

    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = validate_claim(repo_root, name, claim, required, input_text, harness_text)
        if claim_name in seen:
            fail(f"{name} duplicate claim {claim_name}")
        seen.add(claim_name)
    return seen


def validate_static_fixture(repo_root: Path, fixture: dict) -> set[str]:
    name = fixture.get("name")
    if name != "phase14_static_staged_guards":
        fail(f"unexpected static fixture {name!r}")
    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = validate_claim(repo_root, name, claim, {}, "", "")
        if claim_name in seen:
            fail(f"{name} duplicate claim {claim_name}")
        seen.add(claim_name)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--claims", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    claims_path = Path(args.claims)
    if not claims_path.is_absolute():
        claims_path = repo_root / claims_path
    data = load_json(claims_path)
    if data.get("schema_version") != 1:
        fail("schema_version must be 1")

    fixtures = data.get("fixtures", [])
    fixture_names = {fixture.get("name") for fixture in fixtures}
    if fixture_names != REQUIRED_FIXTURES:
        fail(f"fixture set mismatch got={sorted(fixture_names)}")

    claims = set()
    for fixture in fixtures:
        if fixture.get("name") == "phase14_static_staged_guards":
            claims.update(validate_static_fixture(repo_root, fixture))
        else:
            claims.update(validate_hardware_fixture(repo_root, fixture))
    missing = REQUIRED_CLAIMS - claims
    if missing:
        fail(f"missing required claims {sorted(missing)}")

    print(f"PASS Phase 14 storage/numeric isolation claim audit: fixtures={len(fixtures)} claims={len(claims)}")
    print("VALUE_STORAGE_NUMERIC_CLAIM_AUDIT=PASS")


if __name__ == "__main__":
    main()
