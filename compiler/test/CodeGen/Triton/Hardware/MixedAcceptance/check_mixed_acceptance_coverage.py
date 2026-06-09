#!/usr/bin/env python3
"""Validate the Phase 7 TTIR mixed acceptance manifest."""

import argparse
import json
import re
import sys
from pathlib import Path


TOP_LEVEL_FIELDS = {
    "schema_version",
    "suite_name",
    "fixtures",
    "required_features",
    "isolation_backstops",
    "final_acceptance_policy",
}

FIXTURE_FIELDS = {
    "name",
    "dialect",
    "path",
    "runner_kind",
    "status",
    "real_world_role",
    "feature_tags",
    "required_result_fields",
    "expected_cases_summary",
    "triage_feature_bands",
}

FEATURE_FIELDS = {"id", "verification_kind", "covered_by", "required"}

REQUIRED_FIXTURES = {
    "mixed_ttir_saxpy_cmp_select_tail_vc4triton",
    "mixed_ttir_i32_f32_dual_kernel_tail_vc4triton",
}

REQUIRED_FEATURES = {
    "real_ttir_input",
    "ttir_importer_lower_elementwise_v1",
    "value_surface_verification",
    "value_to_vc4kernel",
    "program_id_axis0",
    "arange_make_range_16",
    "masked_load_other_zero",
    "masked_store_tail",
    "tmu_load",
    "vdw_preserve_store",
    "tail_mask_clamp_overlaunch",
    "f32_alu",
    "f32_cmp_select",
    "i32_alu",
    "i32_cmp_select",
    "active_qpus_12",
    "sentinel_preserve",
    "nonzero_output_hash",
}

FORBIDDEN_TTIR_INPUT_MARKERS = (
    "vc4value.",
    "vc4kernel.",
    "ssavc4.",
    "vc4.module",
    "vc4.qpu.",
    "ttg.",
    "gpu.",
    "nvgpu.",
    "nvvm.",
    "rocdl.",
    "llvm.",
)


def fail(message):
    print(f"FAIL TTIR mixed acceptance coverage: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def require_string(value, context):
    if not isinstance(value, str) or not value.strip():
        fail(f"{context} must be a non-empty string")


def require_string_list(value, context):
    if not isinstance(value, list) or not value:
        fail(f"{context} must be a non-empty list")
    for item in value:
        require_string(item, context)


def ttir_inputs(fixture_dir):
    single = fixture_dir / "input.ttir.mlir"
    if single.exists():
        return [single]
    inputs_dir = fixture_dir / "inputs"
    if inputs_dir.exists():
        return sorted(inputs_dir.glob("*.ttir.mlir"))
    return []


def require_expected_field(expected, field, fixture_name, lock_mode):
    if field == "status":
        if expected.get("status") != "PASS":
            fail(f"{fixture_name} expected.json must require top-level status=PASS")
        return
    required = expected.get("required")
    if not isinstance(required, dict):
        fail(f"{fixture_name} expected.json must contain required object")
    if field == "output_hash" and not lock_mode:
        return
    if field not in required:
        fail(f"{fixture_name} expected.json missing required result field {field}")
    if field == "output_hash" and lock_mode and int(required[field]) == 0:
        fail(f"{fixture_name} expected.json must require nonzero output_hash in lock mode")


def validate_ttir_inputs(name, paths):
    if not paths:
        fail(f"{name} has no real TTIR input snapshots")
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r'(?<![A-Za-z0-9_])"?tt\.', text):
            fail(f"{name} input {path} lacks tt dialect operations")
        for marker in FORBIDDEN_TTIR_INPUT_MARKERS:
            if marker in text:
                fail(f"{name} input {path} contains forbidden marker {marker}")
    combined = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths)
    for marker in ("tt.get_program_id", "tt.make_range", "tt.load", "tt.store"):
        if marker not in combined:
            fail(f"{name} TTIR inputs missing {marker}")
    if "tt.dot" in combined or "tt.reduce" in combined:
        fail(f"{name} TTIR inputs contain unsupported staged ops")


def validate_fixture(repo_root, fixture, lock_mode):
    extra = set(fixture) - FIXTURE_FIELDS
    missing = FIXTURE_FIELDS - set(fixture)
    if extra:
        fail(f"{fixture.get('name', '<unnamed>')} has unexpected fields {sorted(extra)}")
    if missing:
        fail(f"{fixture.get('name', '<unnamed>')} missing fields {sorted(missing)}")

    name = fixture["name"]
    if fixture["dialect"] != "ttir":
        fail(f"{name} dialect must be ttir")
    if fixture["runner_kind"] != "ttir_candidate":
        fail(f"{name} runner_kind must be ttir_candidate")
    if fixture["status"] != "implemented":
        fail(f"{name} status must be implemented")
    require_string(fixture["real_world_role"], f"{name}.real_world_role")
    require_string_list(fixture["feature_tags"], f"{name}.feature_tags")
    require_string_list(fixture["required_result_fields"], f"{name}.required_result_fields")
    require_string_list(fixture["triage_feature_bands"], f"{name}.triage_feature_bands")

    fixture_dir = repo_root / fixture["path"]
    expected_path = fixture_dir / "expected.json"
    harness_path = fixture_dir / "candidate" / f"{name}_candidate_harness.c"
    for path in (fixture_dir, expected_path, harness_path):
        if not path.exists():
            fail(f"{name} missing required path {path}")

    inputs = ttir_inputs(fixture_dir)
    validate_ttir_inputs(name, inputs)
    if name == "mixed_ttir_i32_f32_dual_kernel_tail_vc4triton" and len(inputs) < 2:
        fail(f"{name} must contain at least two real TTIR inputs")

    expected = load_json(expected_path)
    for field in fixture["required_result_fields"]:
        require_expected_field(expected, field, name, lock_mode)
    if expected.get("required", {}).get("active_qpus") != 12:
        fail(f"{name} expected.json must require active_qpus=12")
    if expected.get("required", {}).get("lanes") != 16:
        fail(f"{name} expected.json must require lanes=16")

    tags = set(fixture["feature_tags"])
    ttir_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in inputs)
    for required in ("real_ttir_input", "ttir_importer_lower_elementwise_v1", "active_qpus_12", "sentinel_preserve", "nonzero_output_hash"):
        if required not in tags:
            fail(f"{name} missing required feature tag {required}")
    if "f32_cmp_select" in tags and ("arith.cmpf" not in ttir_text or "arith.select" not in ttir_text):
        fail(f"{name} claims f32_cmp_select but TTIR lacks cmpf/select")
    if "i32_cmp_select" in tags and ("arith.cmpi" not in ttir_text or "arith.select" not in ttir_text):
        fail(f"{name} claims i32_cmp_select but TTIR lacks cmpi/select")


def validate_manifest(repo_root, manifest, lock_mode):
    if manifest.get("schema_version") != 1:
        fail("schema_version must be 1")
    if set(manifest) != TOP_LEVEL_FIELDS:
        fail(
            "top-level fields mismatch "
            f"missing={sorted(TOP_LEVEL_FIELDS - set(manifest))} "
            f"extra={sorted(set(manifest) - TOP_LEVEL_FIELDS)}"
        )
    if manifest.get("suite_name") != "vc4_triton_phase7_mixed_acceptance":
        fail("unexpected suite_name")

    fixtures = manifest["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        fail("fixtures must be a non-empty list")
    fixture_names = [fixture.get("name") for fixture in fixtures]
    if set(fixture_names) != REQUIRED_FIXTURES:
        fail(
            "mixed fixture set mismatch "
            f"missing={sorted(REQUIRED_FIXTURES - set(fixture_names))} "
            f"extra={sorted(set(fixture_names) - REQUIRED_FIXTURES)}"
        )
    if len(fixture_names) != len(set(fixture_names)):
        fail("duplicate fixture names")
    for fixture in fixtures:
        validate_fixture(repo_root, fixture, lock_mode)

    features = manifest["required_features"]
    if not isinstance(features, list) or not features:
        fail("required_features must be a non-empty list")
    by_feature = {feature.get("id"): feature for feature in features}
    if set(by_feature) != REQUIRED_FEATURES:
        fail(
            "required feature set mismatch "
            f"missing={sorted(REQUIRED_FEATURES - set(by_feature))} "
            f"extra={sorted(set(by_feature) - REQUIRED_FEATURES)}"
        )
    fixture_tags = {fixture["name"]: set(fixture["feature_tags"]) for fixture in fixtures}
    for feature_id, feature in sorted(by_feature.items()):
        extra = set(feature) - FEATURE_FIELDS
        missing = FEATURE_FIELDS - set(feature)
        if extra:
            fail(f"{feature_id} has unexpected fields {sorted(extra)}")
        if missing:
            fail(f"{feature_id} missing fields {sorted(missing)}")
        if feature.get("required") is not True:
            fail(f"{feature_id} must be required=true")
        covered_by = feature["covered_by"]
        if not isinstance(covered_by, list) or not covered_by:
            fail(f"{feature_id}.covered_by must be a non-empty list")
        for fixture_name in covered_by:
            if fixture_name not in fixture_tags:
                fail(f"{feature_id} covered_by unknown fixture {fixture_name}")
            if feature_id not in fixture_tags[fixture_name]:
                fail(f"{feature_id} covered_by {fixture_name} but fixture lacks feature tag")

    print(
        "PASS TTIR mixed acceptance coverage: "
        f"fixtures={len(fixtures)} required_features={len(features)} mode={'lock' if lock_mode else 'draft'}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--mode", choices=["draft", "lock"], default="draft")
    args = parser.parse_args()
    validate_manifest(args.repo_root.resolve(), load_json(args.manifest), args.mode == "lock")


if __name__ == "__main__":
    main()
