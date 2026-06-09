#!/usr/bin/env python3
"""Audit VC4Value mixed fixture result-field feature claims."""

import argparse
import json
import re
import sys
from pathlib import Path


CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
CLAIM_PREFIXES = ("saw_", "no_")


def fail(message):
    print(f"FAIL VC4Value mixed fixture claim audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def require_string(value, context):
    if not isinstance(value, str) or not value.strip():
        fail(f"{context} must be a non-empty string")


def claim_fields_from_expected(repo_root, fixture):
    expected = load_json(repo_root / fixture["path"] / "expected.json")
    fields = {
        field
        for field in fixture["required_result_fields"]
        if field.startswith(CLAIM_PREFIXES)
    }
    fields.update(
        field
        for field in expected.get("required", {})
        if field.startswith(CLAIM_PREFIXES)
    )
    return fields


def require_marker(text, marker, context):
    if marker not in text:
        fail(f"{context} missing marker {marker}")


def audit_known_patterns(repo_root, fixture_name, fixture_claims):
    fixture_dir = repo_root / fixture_claims["path"]
    input_path = fixture_dir / "input.mlir"
    harness_path = fixture_dir / "candidate" / f"{fixture_name}_candidate_harness.c"
    mlir = input_path.read_text()
    harness = harness_path.read_text()
    claim_names = {claim["claim"] for claim in fixture_claims["claims"]}

    if "saw_value_mixed_elementwise" in claim_names:
        for marker in ("arith.mulf", "arith.addf", "arith.cmpf", "arith.select", "vector.transfer_write"):
            require_marker(mlir, marker, "saw_value_mixed_elementwise input")
        for marker in ("tmp > THRESHOLD", "fallback_values[index]", "total_mismatches"):
            require_marker(harness, marker, "saw_value_mixed_elementwise harness")

    if "saw_value_tail_mask" in claim_names:
        require_marker(mlir, "vector.create_mask", "saw_value_tail_mask input")
        for marker in ("verify_sentinels", "sentinel_mismatches"):
            require_marker(harness, marker, "saw_value_tail_mask harness")

    if "saw_value_tmu_load" in claim_names:
        if mlir.count("vector.transfer_read") < 2:
            fail("saw_value_tmu_load requires multiple checked transfer_read operations")
        require_marker(harness, "vc4_m2_copy_htod", "saw_value_tmu_load harness")
        require_marker(harness, "total_mismatches", "saw_value_tmu_load harness")

    if "saw_value_vdw_preserve" in claim_names or "saw_value_store_preserve" in claim_names:
        require_marker(mlir, "vector.transfer_write", "store preserve input")
        for marker in ("fill_buffers", "verify_sentinels", "sentinel_mismatches"):
            require_marker(harness, marker, "store preserve harness")

    if "saw_value_f32_cmp_select" in claim_names:
        for marker in ('vc4value.fp_domain = "finite"', "arith.cmpf", "arith.select"):
            require_marker(mlir, marker, "saw_value_f32_cmp_select input")
        if "tmp > THRESHOLD" not in harness and "looped > THRESHOLD" not in harness:
            fail("saw_value_f32_cmp_select harness missing finite threshold oracle")

    if "saw_value_mixed_i32_f32" in claim_names:
        for marker in ("memref<?xi32", "memref<?xf32", "arith.cmpi", "arith.mulf", "arith.addf", "arith.select"):
            require_marker(mlir, marker, "saw_value_mixed_i32_f32 input")
        for marker in ("xi_values[index] > THRESHOLD_I", "candidate", "yf_values[index]"):
            require_marker(harness, marker, "saw_value_mixed_i32_f32 harness")

    if "saw_value_i32_cmp" in claim_names:
        require_marker(mlir, "arith.cmpi sgt", "saw_value_i32_cmp input")
        require_marker(harness, "xi_values[index] > THRESHOLD_I", "saw_value_i32_cmp harness")

    if "saw_value_f32_alu" in claim_names:
        for marker in ("arith.mulf", "arith.addf"):
            require_marker(mlir, marker, "saw_value_f32_alu input")
        if (
            "a * xf_values[index] + yf_values[index]" not in harness
            and "factor * xf_values[index] + yf_values[index]" not in harness
            and "factor * yf_values[index] + xf_values[index]" not in harness
        ):
            fail("saw_value_f32_alu harness missing checked f32 arithmetic oracle")

    if "saw_value_cf_loop" in claim_names:
        for marker in ("cf.br ^loop", "cf.cond_br", "arith.cmpi ult"):
            require_marker(mlir, marker, "saw_value_cf_loop input")
        if "trip_cases" not in harness or "trip" not in harness:
            fail("saw_value_cf_loop harness missing varied trip-count oracle")

    if "saw_value_cf_cond_br" in claim_names:
        require_marker(mlir, "cf.cond_br", "saw_value_cf_cond_br input")
        if "flag_cases" not in harness and "use_select" not in harness and "use_i32_path" not in harness:
            fail("saw_value_cf_cond_br harness missing runtime branch flag")

    if "saw_value_vector_block_arg" in claim_names:
        require_marker(mlir, "vector<16xf32>", "saw_value_vector_block_arg input")
        if mlir.count("vector<16xf32>") < 4:
            fail("saw_value_vector_block_arg requires multiple vector block argument sites")
        if "merged vector" in harness:
            fail("saw_value_vector_block_arg cannot be claimed by comments only")

    for claim in fixture_claims["claims"]:
        evidence_text = json.dumps(claim.get("evidence", {}), sort_keys=True).lower()
        if "fixture name" in evidence_text or "status string" in evidence_text or "generated path" in evidence_text:
            fail(f"{fixture_name}:{claim['claim']} uses decorative evidence")


def validate_claim_manifest(repo_root, mixed_manifest, claim_manifest):
    if claim_manifest.get("schema_version") != 1:
        fail("claim manifest schema_version must be 1")
    require_string(claim_manifest.get("suite_name"), "suite_name")

    mixed_fixtures = {fixture["name"]: fixture for fixture in mixed_manifest["fixtures"]}
    claim_fixtures = {fixture.get("name"): fixture for fixture in claim_manifest.get("fixtures", [])}
    if set(claim_fixtures) != set(mixed_fixtures):
        fail(
            "claim fixture set mismatch "
            f"missing={sorted(set(mixed_fixtures) - set(claim_fixtures))} "
            f"extra={sorted(set(claim_fixtures) - set(mixed_fixtures))}"
        )

    total_claims = 0
    phase_guard_claims = 0
    for fixture_name, mixed_fixture in sorted(mixed_fixtures.items()):
        fixture_claims = claim_fixtures[fixture_name]
        if fixture_claims.get("path") != mixed_fixture.get("path"):
            fail(f"{fixture_name} claim path does not match manifest path")
        source_paths = fixture_claims.get("source_paths")
        if not isinstance(source_paths, list) or len(source_paths) < 2:
            fail(f"{fixture_name} must list input and harness source_paths")
        for rel_path in source_paths:
            path = repo_root / rel_path
            if not path.exists():
                fail(f"{fixture_name} source path does not exist: {rel_path}")

        required_claims = claim_fields_from_expected(repo_root, mixed_fixture)
        claims = fixture_claims.get("claims")
        if not isinstance(claims, list) or not claims:
            fail(f"{fixture_name} must have non-empty claims")
        by_claim = {}
        for claim in claims:
            claim_name = claim.get("claim")
            require_string(claim_name, f"{fixture_name} claim")
            if not claim_name.startswith(CLAIM_PREFIXES):
                fail(f"{fixture_name}:{claim_name} is not a saw_/no_ result claim")
            if claim_name in by_claim:
                fail(f"{fixture_name} duplicate claim {claim_name}")
            kind = claim.get("claim_kind")
            if kind not in CLAIM_KINDS:
                fail(f"{fixture_name}:{claim_name} invalid claim_kind {kind!r}")
            evidence = claim.get("evidence")
            if not isinstance(evidence, dict) or not evidence:
                fail(f"{fixture_name}:{claim_name} lacks evidence")
            if kind == "CHECKED_OUTPUT" and "checked_output" not in evidence:
                fail(f"{fixture_name}:{claim_name} lacks checked_output evidence")
            if kind == "CHECKED_AUDIT" and "checked_audit" not in evidence:
                fail(f"{fixture_name}:{claim_name} lacks checked_audit evidence")
            if kind == "PHASE_GUARD":
                phase_guard_claims += 1
                guards = evidence.get("guard_fixtures")
                if not isinstance(guards, list) or not guards:
                    fail(f"{fixture_name}:{claim_name} PHASE_GUARD lacks guard_fixtures")
            by_claim[claim_name] = claim
        if set(by_claim) != required_claims:
            fail(
                f"{fixture_name} claim set mismatch "
                f"missing={sorted(required_claims - set(by_claim))} "
                f"extra={sorted(set(by_claim) - required_claims)}"
            )
        audit_known_patterns(repo_root, fixture_name, fixture_claims)
        total_claims += len(claims)

    print(
        "PASS VC4Value mixed fixture claim audit: "
        f"fixtures={len(mixed_fixtures)} claims={total_claims} "
        f"phase_guards={phase_guard_claims}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    validate_claim_manifest(
        args.repo_root.resolve(),
        load_json(args.manifest),
        load_json(args.claims),
    )


if __name__ == "__main__":
    main()
