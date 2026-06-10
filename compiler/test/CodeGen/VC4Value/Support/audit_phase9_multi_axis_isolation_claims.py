#!/usr/bin/env python3
"""Audit Phase 9 VC4Value multi-axis hardware isolation claims."""

import argparse
import json
import sys
from pathlib import Path

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_FIXTURES = {
    "value_multi_axis_pid2d_i32_vc4value",
    "value_multi_axis_pid3d_i32_vc4value",
    "value_multi_axis_num_programs_axes_vc4value",
    "value_multi_axis_tail_cf_vc4value",
}
REQUIRED_CLAIMS = {
    "program_id_axis0",
    "program_id_axis1",
    "program_id_axis2",
    "num_programs_axis0",
    "num_programs_axis1",
    "num_programs_axis2",
    "grid_rank_2",
    "grid_rank_3",
    "flattened_2d_index",
    "flattened_3d_index",
    "tail_interaction",
    "control_flow_interaction",
    "no_mask_rank2_scope_creep",
}


def fail(message: str) -> None:
    print(f"FAIL Phase 9 multi-axis isolation claim audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def require_marker(text: str, marker: str, context: str) -> None:
    if marker not in text:
        fail(f"{context} missing marker {marker!r}")


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
        fail(f"{name} expected.json must require status PASS")
    for field in ("total_mismatches", "sentinel_mismatches", "launch_failures"):
        if required.get(field) != 0:
            fail(f"{name} expected.json must require {field}=0")
    if required.get("active_qpus") != 12:
        fail(f"{name} expected.json must require active_qpus=12")
    if required.get("lanes") != 16:
        fail(f"{name} expected.json must require lanes=16")
    if not isinstance(required.get("output_hash"), int) or required["output_hash"] == 0:
        fail(f"{name} expected.json must require nonzero output_hash")
    if required.get("saw_value_multi_axis_launch") != 1:
        fail(f"{name} expected.json must require saw_value_multi_axis_launch=1")

    for forbidden in ("tt.", "ttg.", "vc4kernel.", "ssavc4.", "vc4.qpu.", "arith.sitofp", "arith.fptosi"):
        if forbidden in input_text:
            fail(f"{name} input contains forbidden marker {forbidden}")
    if "memref<?x?x" in input_text or "memref<4x16x" in input_text:
        fail(f"{name} input appears to use rank-2 memory")

    claims = set()
    for claim in fixture.get("claims", []):
        claim_name = claim.get("claim")
        kind = claim.get("kind")
        if kind not in CLAIM_KINDS:
            fail(f"{name}:{claim_name} has invalid kind {kind!r}")
        if kind == "CHECKED_OUTPUT" and "expected_field" in claim:
            if required.get(claim["expected_field"]) != 1:
                fail(f"{name}:{claim_name} expected field {claim['expected_field']} must be 1")
        if "input_marker" in claim:
            require_marker(input_text, claim["input_marker"], f"{name}:{claim_name} input")
        if "oracle_marker" in claim:
            require_marker(harness_text, claim["oracle_marker"], f"{name}:{claim_name} harness")
        for forbidden in claim.get("forbidden_input_markers", []):
            if forbidden in input_text:
                fail(f"{name}:{claim_name} forbidden input marker {forbidden}")
        claims.add(claim_name)
    return claims


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

    seen_claims = set()
    for fixture in fixtures:
        seen_claims.update(validate_fixture(repo_root, fixture))

    missing = REQUIRED_CLAIMS - seen_claims
    if missing:
        fail(f"missing required claims {sorted(missing)}")

    print(
        "PASS Phase 9 multi-axis isolation claim audit: "
        f"fixtures={len(fixtures)} claims={len(seen_claims)}"
    )


if __name__ == "__main__":
    main()
