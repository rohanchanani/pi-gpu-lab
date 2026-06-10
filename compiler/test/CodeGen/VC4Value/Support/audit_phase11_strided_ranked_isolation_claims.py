#!/usr/bin/env python3
"""Audit Phase 11 VC4Value strided/ranked hardware isolation claims."""

import argparse
import json
import sys
from pathlib import Path

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_FIXTURES = {
    "value_strided_rank1_row_copy_vc4value",
    "value_rank2_row_slice_copy_vc4value",
    "value_rank2_row_slice_strided_vc4value",
    "value_memref_dim_row_bounds_vc4value",
    "value_ranked_memory_repeat_empty_vc4value",
}
REQUIRED_CLAIMS = {
    "saw_value_rank1_flattened_stride",
    "saw_value_row_stride",
    "saw_value_rank2_row_slice_identity",
    "saw_value_rank2_row_slice_strided",
    "saw_value_stride_args",
    "saw_value_shape_args",
    "saw_value_memref_dim_metadata",
    "saw_row_padding_sentinels",
    "saw_phase10_tail_masks",
    "saw_no_hidden_descriptor",
    "saw_repeat_invocation",
    "saw_empty_rows_cols",
    "gather_lane_stride_static_reject",
    "no_after_boot_timeout",
}


def fail(message: str) -> None:
    print(f"FAIL Phase 11 strided/ranked isolation claim audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def require_marker(text: str, marker: str, context: str) -> None:
    if marker not in text:
        fail(f"{context} missing marker {marker!r}")


def validate_static_reject(repo_root: Path, fixture_name: str, claim_name: str, static_test: str) -> None:
    path = repo_root / "compiler/test/Conversion/VC4ValueToVC4Kernel" / static_test
    if not path.exists():
        fail(f"{fixture_name}:{claim_name} missing static reject test {static_test}")
    text = path.read_text()
    if static_test == "invalid-rank2-nonunit-inner-stride.mlir":
        require_marker(text, "static inner stride 1", f"{static_test} static reject")
    elif static_test in {
        "invalid-rank2-column-slice-transfer-map.mlir",
        "invalid-rank2-transposed-transfer-map.mlir",
    }:
        require_marker(text, "gather-like maps are staged", f"{static_test} static reject")
    elif static_test == "invalid-hidden-memref-descriptor-dim.mlir":
        require_marker(text, "hidden descriptor ABI is rejected", f"{static_test} static reject")
    else:
        require_marker(text, "RUN: not", f"{static_test} static reject")


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

    for forbidden in (
        "tt.",
        "ttg.",
        "vc4kernel.",
        "ssavc4.",
        "vc4.qpu.",
        "vector.gather",
        "vector.scatter",
        "vector.reduction",
        "vector.contract",
        "arith.sitofp",
        "arith.fptosi",
    ):
        if forbidden in input_text:
            fail(f"{name} input contains forbidden marker {forbidden}")
    if "vector<16x" not in input_text:
        fail(f"{name} input must use vector<16xT> row-slice fragments")

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
            validate_static_reject(repo_root, name, claim_name, static_test)
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

    print(f"PASS Phase 11 strided/ranked isolation claim audit: fixtures={len(fixtures)} claims={len(claims)}")


if __name__ == "__main__":
    main()
