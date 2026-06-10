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
    "mixed_ttir_cf_loop_if_tail_vc4triton",
    "mixed_ttir_multi_axis_cf_tail_vc4triton",
    "mixed_ttir_mask_memory_cf_axes_b16_vc4triton",
    "mixed_ttir_strided_memory_axes_mask_cf_b16_vc4triton",
    "mixed_ttir_reduction_axes_mask_cf_strided_b16_vc4triton",
}

REQUIRED_FEATURES = {
    "real_ttir_input",
    "real_ttir_snapshot",
    "cpp_ttir_importer",
    "value_surface_verification",
    "value_to_vc4kernel",
    "value_scf_cf",
    "program_id_axis0",
    "arange_make_range_16",
    "masked_load_other_zero",
    "masked_store_tail",
    "tmu_load",
    "vdw_preserve_store",
    "tail_mask_clamp_overlaunch",
    "ttir_elementwise",
    "ttir_tail_mask",
    "ttir_scf_if",
    "ttir_scf_for_or_while",
    "ttir_cf_control_flow",
    "ttir_multi_axis_launch",
    "ttir_program_id_axis1",
    "ttir_program_id_axis2",
    "ttir_num_programs_axis1",
    "value_multi_axis_launch",
    "ttir_mask_tail",
    "ttir_mask_full",
    "ttir_mask_empty",
    "ttir_compute_mask_select",
    "ttir_control_flow",
    "ttir_multi_axis",
    "ttir_row_strided_memory",
    "value_mask_classifier",
    "value_strided_address",
    "no_sparse_memory_mask",
    "no_gather_lane_stride",
    "no_hidden_memref_descriptor",
    "row_padding_sentinels",
    "ttir_reduction_i32_add",
    "ttir_reduction_f32_finite_add",
    "ttir_scalar_reduction_store",
    "f32_finite_tree_policy",
    "no_dot_gemv",
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


def validate_ttir_inputs(name, paths, allow_reduce=False):
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
    if "tt.dot" in combined:
        fail(f"{name} TTIR inputs contain unsupported staged ops")
    if "tt.reduce" in combined and not allow_reduce:
        fail(f"{name} TTIR inputs contain reduction ops outside the Phase 12 reduction fixture")


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
    tags = set(fixture["feature_tags"])
    validate_ttir_inputs(name, inputs, allow_reduce="ttir_reduction_f32_finite_add" in tags or "ttir_reduction_i32_add" in tags)
    if name == "mixed_ttir_i32_f32_dual_kernel_tail_vc4triton" and len(inputs) < 2:
        fail(f"{name} must contain at least two real TTIR inputs")

    expected = load_json(expected_path)
    for field in fixture["required_result_fields"]:
        require_expected_field(expected, field, name, lock_mode)
    if expected.get("required", {}).get("active_qpus") != 12:
        fail(f"{name} expected.json must require active_qpus=12")
    if expected.get("required", {}).get("lanes") != 16:
        fail(f"{name} expected.json must require lanes=16")

    ttir_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in inputs)
    for required in ("real_ttir_input", "cpp_ttir_importer", "active_qpus_12", "sentinel_preserve", "nonzero_output_hash"):
        if required not in tags:
            fail(f"{name} missing required feature tag {required}")
    if "f32_cmp_select" in tags and ("arith.cmpf" not in ttir_text or "arith.select" not in ttir_text):
        fail(f"{name} claims f32_cmp_select but TTIR lacks cmpf/select")
    if "i32_cmp_select" in tags and ("arith.cmpi" not in ttir_text or "arith.select" not in ttir_text):
        fail(f"{name} claims i32_cmp_select but TTIR lacks cmpi/select")
    if "ttir_scf_if" in tags and "scf.if" not in ttir_text:
        fail(f"{name} claims ttir_scf_if but TTIR lacks scf.if")
    if "ttir_scf_for_or_while" in tags and "scf.for" not in ttir_text and "scf.while" not in ttir_text:
        fail(f"{name} claims ttir_scf_for_or_while but TTIR lacks scf.for/scf.while")
    if "ttir_cf_control_flow" in tags and ("scf.if" not in ttir_text or ("scf.for" not in ttir_text and "scf.while" not in ttir_text)):
        fail(f"{name} claims ttir_cf_control_flow but TTIR lacks mixed control-flow forms")
    if "ttir_multi_axis_launch" in tags:
        for marker in ("tt.get_program_id y", "tt.get_program_id z", "tt.get_num_programs y"):
            if marker not in ttir_text:
                fail(f"{name} claims ttir_multi_axis_launch but TTIR lacks {marker}")
    if "ttir_program_id_axis1" in tags and "tt.get_program_id y" not in ttir_text:
        fail(f"{name} claims ttir_program_id_axis1 but TTIR lacks tt.get_program_id y")
    if "ttir_program_id_axis2" in tags and "tt.get_program_id z" not in ttir_text:
        fail(f"{name} claims ttir_program_id_axis2 but TTIR lacks tt.get_program_id z")
    if "ttir_num_programs_axis1" in tags and "tt.get_num_programs y" not in ttir_text:
        fail(f"{name} claims ttir_num_programs_axis1 but TTIR lacks tt.get_num_programs y")
    if "ttir_num_programs_axis0" in tags and "tt.get_num_programs x" not in ttir_text:
        fail(f"{name} claims ttir_num_programs_axis0 but TTIR lacks tt.get_num_programs x")
    if "ttir_multi_axis" in tags and ("tt.get_program_id x" not in ttir_text or "tt.get_program_id y" not in ttir_text):
        fail(f"{name} claims ttir_multi_axis but TTIR lacks program_id x/y")
    if "ttir_mask_tail" in tags and "arith.cmpi slt" not in ttir_text:
        fail(f"{name} claims ttir_mask_tail but TTIR lacks canonical slt tail mask")
    if "ttir_compute_mask_select" in tags and ("arith.cmpf" not in ttir_text or "arith.select" not in ttir_text):
        fail(f"{name} claims ttir_compute_mask_select but TTIR lacks cmpf/select")
    if "ttir_row_strided_memory" in tags and ("tt.get_program_id y" not in ttir_text or not re.search(r"arith\.muli %[A-Za-z0-9_]+, %ld[oxy]", ttir_text)):
        fail(f"{name} claims ttir_row_strided_memory but TTIR lacks row*stride pointer arithmetic")
    if "value_mask_classifier" in tags and "saw_value_mask_classifier" not in (repo_root / fixture["path"] / "expected.json").read_text():
        fail(f"{name} claims value_mask_classifier but expected.json lacks saw_value_mask_classifier")
    if "value_strided_address" in tags and "saw_value_strided_address" not in (repo_root / fixture["path"] / "expected.json").read_text():
        fail(f"{name} claims value_strided_address but expected.json lacks saw_value_strided_address")
    if "no_sparse_memory_mask" in tags and "sparse" in ttir_text.lower():
        fail(f"{name} claims no_sparse_memory_mask but TTIR contains sparse marker")
    if "no_gather_lane_stride" in tags and re.search(r"arith\.muli %lanes, %", ttir_text):
        fail(f"{name} claims no_gather_lane_stride but TTIR contains lane-varying stride")
    if "no_hidden_memref_descriptor" in tags and "descriptor" in ttir_text.lower():
        fail(f"{name} claims no_hidden_memref_descriptor but TTIR contains descriptor marker")
    if "ttir_reduction_i32_add" in tags and ("tt.reduce" not in ttir_text or "arith.addi" not in ttir_text):
        fail(f"{name} claims ttir_reduction_i32_add but TTIR lacks i32 add reduction")
    if "ttir_reduction_f32_finite_add" in tags and ("tt.reduce" not in ttir_text or "arith.addf" not in ttir_text):
        fail(f"{name} claims ttir_reduction_f32_finite_add but TTIR lacks f32 add reduction")
    if "ttir_scalar_reduction_store" in tags and ("tt.store" not in ttir_text or "tt.reduce" not in ttir_text):
        fail(f"{name} claims ttir_scalar_reduction_store but TTIR lacks reduction scalar store")
    if "no_dot_gemv" in tags and ("tt.dot" in ttir_text or "gemv" in ttir_text.lower()):
        fail(f"{name} claims no_dot_gemv but TTIR contains dot/GEMV marker")


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
