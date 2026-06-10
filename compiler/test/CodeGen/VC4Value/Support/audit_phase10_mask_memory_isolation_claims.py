#!/usr/bin/env python3
"""Audit Phase 10 VC4Value mask/memory hardware isolation claims."""

import argparse
import json
import sys
from pathlib import Path

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_FIXTURES = {
    "value_mask_tail_full_empty_vc4value",
    "value_mask_compute_select_tail_vc4value",
    "value_mask_memory_policy_other_zero_vc4value",
    "value_mask_empty_launch_repeat_vc4value",
}
REQUIRED_CLAIMS = {
    "saw_value_mask_empty",
    "saw_value_mask_full",
    "saw_value_mask_tail",
    "saw_value_create_mask_clamp",
    "saw_value_compute_mask_select",
    "saw_value_memory_tail_mask",
    "saw_no_sparse_memory_mask",
    "saw_value_load_inactive_zero",
    "saw_value_store_inactive_preserve",
    "saw_value_tmu_safe_offset",
    "saw_repeat_invocation",
    "no_after_boot_timeout",
    "sparse_transfer_masks_static_reject",
}


def fail(message: str) -> None:
    print(f"FAIL Phase 10 mask/memory isolation claim audit: {message}", file=sys.stderr)
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
    if required.get("runtime_launches", 0) < 3:
        fail(f"{name} expected.json must require at least three runtime launches")

    for forbidden in ("tt.", "ttg.", "vc4kernel.", "ssavc4.", "vc4.qpu.", "arith.sitofp", "arith.fptosi"):
        if forbidden in input_text:
            fail(f"{name} input contains forbidden marker {forbidden}")
    for forbidden in ("memref<?x?x", "memref<4x16x", "vector.contract", "vector.reduction"):
        if forbidden in input_text:
            fail(f"{name} input contains staged memory/compute marker {forbidden}")

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
        if "lowered_marker" in claim:
            # The static lowering test/audit is structural: the transfer_read
            # source plus Phase 10.4 docs require inactive-zero lowering.
            require_marker(input_text, "vector.transfer_read", f"{name}:{claim_name} input")
            require_marker(harness_text, "saw_value_tmu_safe_offset=1", f"{name}:{claim_name} harness")
        for forbidden in claim.get("forbidden_input_markers", []):
            if forbidden in input_text:
                fail(f"{name}:{claim_name} forbidden input marker {forbidden!r}")
        for static_test in claim.get("static_reject_tests", []):
            path = repo_root / "compiler/test/Conversion/VC4ValueToVC4Kernel" / static_test
            if not path.exists():
                fail(f"{name}:{claim_name} missing static reject test {static_test}")
            text = path.read_text()
            require_marker(text, "sparse or unknown transfer", f"{static_test} static reject")
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

    print(f"PASS Phase 10 mask/memory isolation claim audit: fixtures={len(fixtures)} claims={len(claims)}")


if __name__ == "__main__":
    main()
