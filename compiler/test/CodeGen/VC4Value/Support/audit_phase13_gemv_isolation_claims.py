#!/usr/bin/env python3
"""Audit Phase 13 VC4Value GEMV hardware isolation claims."""

import argparse
import json
import sys
from pathlib import Path

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_FIXTURES = {
    "value_gemv_row_dot_f32_tail_vc4value",
    "value_gemv_partial_kblock_f32_vc4value",
    "value_gemv_empty_repeat_vc4value",
}
REQUIRED_CLAIMS = {
    "f32_row_wise_dot",
    "f32_finite_dot_policy_tolerance",
    "scalar_dot_store",
    "tail_inactive_zero_dot_inputs",
    "row_strided_A",
    "x_vector_contiguous",
    "partial_kblock_dot",
    "no_multiblock_accumulation",
    "i32_dot_staged_by_policy",
    "empty_repeat_launch",
    "empty_k_sentinel_preserve",
    "tt_dot_vector_contract_multiblock_static_rejects",
    "no_after_boot_timeout",
}


def fail(message: str) -> None:
    print(f"FAIL Phase 13 GEMV isolation claim audit: {message}", file=sys.stderr)
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
    path = repo_root / "compiler/test/Conversion/VC4ValueToVC4Kernel" / static_test
    if not path.exists():
        fail(f"missing static reject test {static_test}")
    text = path.read_text()
    if static_test == "invalid-gemv-vector-contract-staged.mlir":
        require_marker(text, "vector.contract is staged for Phase 15/contract", static_test)
    elif static_test == "invalid-gemv-tt-dot-staged.mlir":
        require_marker(text, "tt.dot is staged", static_test)
    elif static_test == "invalid-gemv-multiblock-k-accumulation-staged.mlir":
        require_marker(text, "multi-block K accumulation is staged", static_test)
    elif static_test == "invalid-gemv-exact-f32-policy.mlir":
        require_marker(text, "exact f32 dot requires unsupported exact reduction policy", static_test)
    elif static_test == "invalid-gemv-i32-dot-staged-by-policy.mlir":
        require_marker(text, "i32 dot is staged by current i32 multiply policy", static_test)
    else:
        require_marker(text, "RUN: not", static_test)


def validate_fixture(repo_root: Path, fixture: dict) -> set[str]:
    name = fixture.get("name")
    if name not in REQUIRED_FIXTURES:
        fail(f"unexpected fixture {name!r}")
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
    if required.get("runtime_launches", 0) < 6:
        fail(f"{name} expected.json must require at least six runtime launches")
    require_marker(harness_text, "#define ACTIVE_QPUS 12u", f"{name} harness")
    require_marker(harness_text, "VC4_TEST_RESULT", f"{name} harness")

    for forbidden in (
        "tt.",
        "ttg.",
        "vc4kernel.",
        "ssavc4.",
        "vc4.qpu.",
        "vector.contract",
        "vector.multi_reduction",
        "arith.sitofp",
        "arith.fptosi",
        "math.",
    ):
        if forbidden in input_text:
            fail(f"{name} input contains forbidden marker {forbidden}")
    if "vector.reduction <add>" not in input_text:
        fail(f"{name} input must contain vector.reduction <add>")
    if "vector<16x" not in input_text:
        fail(f"{name} input must use vector<16xT> fragments")

    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = claim.get("claim")
        kind = claim.get("kind")
        if claim_name in seen:
            fail(f"{name} duplicate claim {claim_name}")
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
        claims.update(validate_fixture(repo_root, fixture))
    missing = REQUIRED_CLAIMS - claims
    if missing:
        fail(f"missing required claims {sorted(missing)}")

    print(f"PASS Phase 13 GEMV isolation claim audit: fixtures={len(fixtures)} claims={len(claims)}")
    print("VALUE_GEMV_CLAIM_AUDIT=PASS")


if __name__ == "__main__":
    main()
