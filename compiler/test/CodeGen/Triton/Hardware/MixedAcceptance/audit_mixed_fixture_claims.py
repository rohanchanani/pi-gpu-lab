#!/usr/bin/env python3
"""Audit Phase 7 TTIR mixed fixture result-field feature claims."""

import argparse
import json
import sys
from pathlib import Path


CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
CLAIM_PREFIXES = ("saw_", "no_")


def fail(message):
    print(f"FAIL TTIR mixed fixture claim audit: {message}", file=sys.stderr)
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


def fixture_ttir_text(fixture_dir):
    single = fixture_dir / "input.ttir.mlir"
    if single.exists():
        return single.read_text(encoding="utf-8", errors="replace")
    inputs = sorted((fixture_dir / "inputs").glob("*.ttir.mlir"))
    return "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in inputs)


def audit_known_patterns(repo_root, fixture_name, fixture_claims):
    fixture_dir = repo_root / fixture_claims["path"]
    harness_path = fixture_dir / "candidate" / f"{fixture_name}_candidate_harness.c"
    ttir = fixture_ttir_text(fixture_dir)
    harness = harness_path.read_text()
    claim_names = {claim["claim"] for claim in fixture_claims["claims"]}

    if "saw_real_ttir_input" in claim_names:
        require_marker(ttir, "tt.func", "saw_real_ttir_input TTIR")
        require_marker(ttir, "tt.return", "saw_real_ttir_input TTIR")

    if "saw_ttir_importer_lower_elementwise_v1" in claim_names:
        require_marker(harness, "VC4_CASE_SAW_TTIR_IMPORT", "saw_ttir_importer_lower_elementwise_v1 harness")

    if "saw_value_surface_verification" in claim_names or "saw_value_to_vc4kernel" in claim_names:
        if "runner_audit" not in json.dumps(fixture_claims["claims"]).lower():
            fail(f"{fixture_name} value/vc4kernel claims must cite runner_audit evidence")

    if "saw_program_id_axis0" in claim_names:
        require_marker(ttir, "tt.get_program_id", "saw_program_id_axis0 TTIR")

    if "saw_arange_make_range_16" in claim_names:
        require_marker(ttir, "tt.make_range", "saw_arange_make_range_16 TTIR")
        require_marker(ttir, "end = 16", "saw_arange_make_range_16 TTIR")

    if "saw_masked_load_other_zero" in claim_names:
        require_marker(ttir, "tt.load", "saw_masked_load_other_zero TTIR")
        require_marker(ttir, "arith.constant dense<0", "saw_masked_load_other_zero TTIR")
        require_marker(harness, "total_mismatches", "saw_masked_load_other_zero harness")

    if "saw_masked_store_tail" in claim_names:
        require_marker(ttir, "tt.store", "saw_masked_store_tail TTIR")
        require_marker(harness, "verify_sentinels", "saw_masked_store_tail harness")

    if "saw_tmu_load" in claim_names:
        if ttir.count("tt.load") < 2:
            fail(f"{fixture_name} saw_tmu_load requires at least two TTIR loads")
        require_marker(harness, "vc4_m2_copy_htod", "saw_tmu_load harness")

    if "saw_vdw_preserve_store" in claim_names or "saw_sentinel_preserve" in claim_names:
        require_marker(harness, "sentinel_mismatches", "preserve harness")
        require_marker(harness, "verify_sentinels", "preserve harness")

    if "saw_tail_mask_clamp_overlaunch" in claim_names:
        require_marker(harness, "rounded_waves", "saw_tail_mask_clamp_overlaunch harness")
        require_marker(harness, "1000u", "saw_tail_mask_clamp_overlaunch harness")

    if "saw_f32_alu" in claim_names:
        if "arith.addf" not in ttir and "arith.mulf" not in ttir:
            fail(f"{fixture_name} saw_f32_alu requires f32 arithmetic in TTIR")
        require_marker(harness, "float", "saw_f32_alu harness")

    if "saw_f32_cmp_select" in claim_names:
        require_marker(ttir, "arith.cmpf", "saw_f32_cmp_select TTIR")
        require_marker(ttir, "arith.select", "saw_f32_cmp_select TTIR")
        require_marker(harness, "threshold", "saw_f32_cmp_select harness")

    if "saw_i32_alu" in claim_names:
        if "arith.addi" not in ttir and "arith.subi" not in ttir:
            fail(f"{fixture_name} saw_i32_alu requires i32 arithmetic in TTIR")
        require_marker(harness, "int32_t", "saw_i32_alu harness")

    if "saw_i32_cmp_select" in claim_names:
        require_marker(ttir, "arith.cmpi", "saw_i32_cmp_select TTIR")
        require_marker(ttir, "arith.select", "saw_i32_cmp_select TTIR")
        require_marker(harness, "tmp > threshold", "saw_i32_cmp_select harness")

    if "saw_nonzero_output_hash" in claim_names:
        require_marker(harness, "output_hash != 0u", "saw_nonzero_output_hash harness")

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
            fail(f"{fixture_name} must list TTIR input and harness source_paths")
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
        "PASS TTIR mixed fixture claim audit: "
        f"fixtures={len(mixed_fixtures)} claims={total_claims} phase_guards={phase_guard_claims}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    validate_claim_manifest(args.repo_root.resolve(), load_json(args.manifest), load_json(args.claims))


if __name__ == "__main__":
    main()
