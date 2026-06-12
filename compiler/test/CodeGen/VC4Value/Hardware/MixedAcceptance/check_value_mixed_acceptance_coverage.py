#!/usr/bin/env python3
"""Validate the VC4Value mixed acceptance manifest."""

import argparse
import json
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

FEATURE_FIELDS = {
    "id",
    "verification_kind",
    "covered_by",
    "required",
}

OPTIONAL_FEATURE_FIELDS = {
    "covered_by_isolation",
    "mixed_status",
}

REQUIRED_FIXTURES = {
    "mixed_value_saxpy_select_tail_vc4value",
    "mixed_value_i32_f32_dual_path_tail_vc4value",
    "mixed_value_cf_saxpy_loop_select_tail_vc4value",
    "mixed_value_cf_i32_f32_dual_path_tail_vc4value",
    "mixed_value_cf_completeness_loop_branch_tail_vc4value",
    "mixed_value_multi_axis_cf_tail_vc4value",
    "mixed_value_mask_memory_axis_cf_vc4value",
    "mixed_value_strided_ranked_memory_axis_mask_cf_vc4value",
    "mixed_value_reduction_axes_mask_cf_strided_vc4value",
    "mixed_value_gemv_row_dot_axes_mask_cf_strided_reduction_vc4value",
    "mixed_value_f16_storage_gemv_axes_mask_cf_reduction_vc4value",
}

REQUIRED_FEATURES = {
    "value_launch_identity",
    "value_vector_step_lane",
    "value_f32_transfer_read",
    "value_i32_transfer_read",
    "value_f32_transfer_write",
    "value_i32_transfer_write",
    "value_f32_alu",
    "value_i32_cmp",
    "value_f32_finite_cmp_select",
    "value_tail_masks",
    "value_vdw_inactive_preserve",
    "value_no_producer_dialect_in_verified_vc4kernel",
    "value_cf_loop",
    "value_cf_cond_br",
    "value_vector_block_arg",
    "value_scf_while",
    "value_nested_structured_cf",
    "value_tl_range_style_loop",
    "value_persistent_loop_skeleton",
    "value_vector_loop_carried",
    "value_multi_exit_reducible_cf",
    "value_scf_to_cf_boundary",
    "value_multi_axis_launch_identity",
    "value_program_id_axis1",
    "value_program_id_axis2",
    "value_num_programs_axis0",
    "value_num_programs_axis1",
    "value_mask_full",
    "value_mask_empty",
    "value_compute_mask_select",
    "value_transfer_read_inactive_zero",
    "value_transfer_write_inactive_preserve",
    "value_no_sparse_memory_mask",
    "value_rank1_flattened_stride",
    "value_rank2_row_slice_strided",
    "value_stride_args_metadata",
    "value_memref_dim_metadata",
    "value_no_gather_lane_stride",
    "value_no_hidden_memref_descriptor",
    "value_row_strided_memory",
    "value_vector_reduction_add_i32",
    "value_vector_reduction_add_f32_finite",
    "value_scalar_reduction_store",
    "value_f32_finite_tree_policy",
    "value_gemv_f32_row_dot",
    "value_gemv_partial_kblock",
    "value_no_tl_dot_tt_dot",
    "value_no_vector_contract",
    "value_no_multiblock_k_accumulation",
    "value_no_dot_gemv",
    "value_f16_storage_load",
    "value_f16_storage_store",
    "value_f32_compute_after_f16_load",
    "value_f16_storage_finite_policy",
    "value_no_native_f16_arithmetic",
    "value_no_bf16_fp8",
    "value_no_softmax_sfu",
}

FORBIDDEN_INPUT_MARKERS = (
    "vc4kernel.",
    "ssavc4.",
    "vc4.module",
    "vc4.qpu.",
    "tt.",
    "ttg.",
)


def fail(message):
    print(f"FAIL VC4Value mixed acceptance coverage: {message}", file=sys.stderr)
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


def require_expected_field(expected, field, fixture_name):
    if field == "status":
        if expected.get("status") != "PASS":
            fail(f"{fixture_name} expected.json must require top-level status=PASS")
        return
    required = expected.get("required")
    if not isinstance(required, dict):
        fail(f"{fixture_name} expected.json must contain required object")
    if field not in required:
        fail(f"{fixture_name} expected.json missing required result field {field}")
    if field == "output_hash" and required[field] == 0:
        fail(f"{fixture_name} expected.json must require nonzero output_hash")


def validate_fixture(repo_root, fixture):
    extra = set(fixture) - FIXTURE_FIELDS
    missing = FIXTURE_FIELDS - set(fixture)
    if extra:
        fail(f"{fixture.get('name', '<unnamed>')} has unexpected fixture fields {sorted(extra)}")
    if missing:
        fail(f"{fixture.get('name', '<unnamed>')} missing fixture fields {sorted(missing)}")

    name = fixture["name"]
    if fixture["dialect"] != "vc4value":
        fail(f"{name} dialect must be vc4value")
    if fixture["runner_kind"] != "vc4value_candidate":
        fail(f"{name} runner_kind must be vc4value_candidate")
    if fixture["status"] != "implemented":
        fail(f"{name} status must be implemented")
    require_string(fixture["real_world_role"], f"{name}.real_world_role")
    require_string_list(fixture["feature_tags"], f"{name}.feature_tags")
    require_string_list(fixture["required_result_fields"], f"{name}.required_result_fields")
    require_string_list(fixture["triage_feature_bands"], f"{name}.triage_feature_bands")

    fixture_dir = repo_root / fixture["path"]
    input_path = fixture_dir / "input.mlir"
    expected_path = fixture_dir / "expected.json"
    harness_path = fixture_dir / "candidate" / f"{name}_candidate_harness.c"
    for path in (fixture_dir, input_path, expected_path, harness_path):
        if not path.exists():
            fail(f"{name} missing required path {path}")

    mlir = input_path.read_text()
    if "vc4value.kernel" not in mlir:
        fail(f"{name} input must start from vc4value.kernel")
    for marker in FORBIDDEN_INPUT_MARKERS:
        if marker in mlir:
            fail(f"{name} value input contains forbidden marker {marker}")

    expected = load_json(expected_path)
    for field in fixture["required_result_fields"]:
        require_expected_field(expected, field, name)

    tags = set(fixture["feature_tags"])
    pattern_requirements = {
        "value_launch_identity": "vc4value.program_id",
        "value_vector_step_lane": "vector.step",
        "value_f32_transfer_read": "vector.transfer_read",
        "value_i32_transfer_read": "vector.transfer_read",
        "value_f32_transfer_write": "vector.transfer_write",
        "value_i32_transfer_write": "vector.transfer_write",
        "value_f32_alu": "arith.mulf",
        "value_i32_cmp": "arith.cmpi",
        "value_f32_finite_cmp_select": "arith.cmpf",
        "value_tail_masks": "vector.create_mask",
        "value_vdw_inactive_preserve": "vector.transfer_write",
        "value_cf_loop": "cf.br",
        "value_cf_cond_br": "cf.cond_br",
        "value_vector_block_arg": "vector<16x",
        "value_scf_while": "scf.while",
        "value_nested_structured_cf": "scf.if",
        "value_tl_range_style_loop": "scf.for",
        "value_persistent_loop_skeleton": "iter_args",
        "value_vector_loop_carried": "vector<16x",
        "value_multi_exit_reducible_cf": "^exit_early",
        "value_scf_to_cf_boundary": "scf.",
        "value_program_id_axis1": "vc4value.program_id {axis = 1",
        "value_program_id_axis2": "vc4value.program_id {axis = 2",
        "value_num_programs_axis0": "vc4value.num_programs {axis = 0",
        "value_num_programs_axis1": "vc4value.num_programs {axis = 1",
        "value_mask_full": "vector.transfer_write",
        "value_mask_empty": "vector.create_mask %c0_index",
        "value_compute_mask_select": "arith.select",
        "value_transfer_read_inactive_zero": "vector.transfer_read",
        "value_transfer_write_inactive_preserve": "vector.transfer_write",
        "value_rank1_flattened_stride": "%flat_idx = arith.addi %row_base_idx, %col",
        "value_rank2_row_slice_strided": "strided<[?, 1], offset: 0>",
        "value_stride_args_metadata": "vc4value.stride_args",
        "value_memref_dim_metadata": "memref.dim",
        "value_row_strided_memory": "strided<[?, 1], offset: 0>",
        "value_vector_reduction_add_i32": "vector.reduction <add>",
        "value_vector_reduction_add_f32_finite": "vector.reduction <add>",
        "value_scalar_reduction_store": "memref.store",
        "value_f32_finite_tree_policy": "vc4value.reduction_policy = \"finite_tree\"",
        "value_gemv_f32_row_dot": "arith.mulf",
        "value_gemv_partial_kblock": "%partial_index = arith.addi",
        "value_f16_storage_load": "vector.transfer_read %rank_f16",
        "value_f16_storage_store": "arith.truncf",
        "value_f32_compute_after_f16_load": "arith.extf",
        "value_f16_storage_finite_policy": "vc4value.f16_storage_policy = \"finite\"",
    }
    for tag, marker in pattern_requirements.items():
        if tag in tags and marker not in mlir:
            fail(f"{name} claims {tag} but input lacks {marker}")
    if "value_multi_axis_launch_identity" in tags:
        has_rank2_flatten = (
            "vc4value.grid_rank = 2" in mlir
            and "vc4value.program_id {axis = 1" in mlir
            and "vc4value.num_programs {axis = 0" in mlir
            and "%row_base = arith.muli %pid1, %num0" in mlir
        )
        has_rank3_flatten = (
            "vc4value.grid_rank = 3" in mlir
            and "vc4value.program_id {axis = 2" in mlir
            and "vc4value.num_programs {axis = 1" in mlir
        )
        if not has_rank2_flatten and not has_rank3_flatten:
            fail(f"{name} claims value_multi_axis_launch_identity without rank-2 or rank-3 flattening")
    if "value_f32_finite_cmp_select" in tags and 'vc4value.fp_domain = "finite"' not in mlir:
        fail(f"{name} uses f32 cmp/select without finite domain policy")
    if "value_i32_transfer_read" in tags and "memref<?xi32" not in mlir and "memref<?x?xi32" not in mlir:
        fail(f"{name} claims i32 transfer read but input has no i32 global memref")
    if "value_f32_transfer_read" in tags and "memref<?xf32" not in mlir and "memref<?x?xf32" not in mlir:
        fail(f"{name} claims f32 transfer read but input has no f32 global memref")
    if "value_no_gather_lane_stride" in tags:
        for marker in ("vector.gather", "vector.scatter", "strided<[?, ?]", "offs * stride"):
            if marker in mlir:
                fail(f"{name} claims no gather/lane stride but input uses {marker}")
    if "value_no_hidden_memref_descriptor" in tags:
        for marker in ("memref.extract_strided_metadata", "memref.reinterpret_cast"):
            if marker in mlir:
                fail(f"{name} claims no hidden memref descriptor but input uses {marker}")
    if "value_no_sparse_memory_mask" in tags:
        if "vector.create_mask" not in mlir:
            fail(f"{name} claims no sparse memory mask but input lacks canonical mask construction")
        sparse_memory_uses = (
            "vector.transfer_read %xi[%base], %zero_i32, %cond_i",
            "vector.transfer_read %xf[%base], %zero_f, %cond_f",
            "vector.transfer_write %store_f, %tail_out_f[%base], %cond",
            "vector.transfer_write %store_i, %tail_out_i[%base], %cond",
        )
        for marker in sparse_memory_uses:
            if marker in mlir:
                fail(f"{name} claims no sparse memory mask but input uses {marker}")
    if "value_no_dot_gemv" in tags:
        for marker in ("vector.contract", "linalg.", "dot", "gemv", "gemm"):
            if marker in mlir.lower():
                fail(f"{name} claims no dot/GEMV/GEMM but input uses {marker}")
    if "value_no_native_f16_arithmetic" in tags:
        for line in mlir.splitlines():
            if ("arith.addf" in line or "arith.mulf" in line) and "vector<16xf16>" in line:
                fail(f"{name} claims no native f16 arithmetic but input uses {line.strip()}")
    if "value_no_bf16_fp8" in tags:
        for marker in ("bf16", "fp8", "f8E", "f8e"):
            if marker in mlir:
                fail(f"{name} claims no bf16/fp8 but input uses {marker}")
    if "value_no_softmax_sfu" in tags:
        for marker in ("math.", "softmax", "exp", "log", "rsqrt", "recip"):
            if marker in mlir.lower():
                fail(f"{name} claims no softmax/SFU but input uses {marker}")


def validate_manifest(repo_root, manifest):
    if manifest.get("schema_version") != 1:
        fail("schema_version must be 1")
    if set(manifest) != TOP_LEVEL_FIELDS:
        fail(
            "top-level fields mismatch "
            f"missing={sorted(TOP_LEVEL_FIELDS - set(manifest))} "
            f"extra={sorted(set(manifest) - TOP_LEVEL_FIELDS)}"
        )
    if manifest.get("suite_name") != "vc4value_phase5_phase14_mixed_acceptance":
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
        validate_fixture(repo_root, fixture)

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
    staged_mixed = []
    for feature_id, feature in sorted(by_feature.items()):
        allowed_fields = FEATURE_FIELDS | OPTIONAL_FEATURE_FIELDS
        extra = set(feature) - allowed_fields
        missing = FEATURE_FIELDS - set(feature)
        if extra:
            fail(f"{feature_id} has unexpected fields {sorted(extra)}")
        if missing:
            fail(f"{feature_id} missing fields {sorted(missing)}")
        if feature.get("required") is not True:
            fail(f"{feature_id} must be required=true")
        covered_by = feature["covered_by"]
        if not isinstance(covered_by, list):
            fail(f"{feature_id}.covered_by must be a list")
        for fixture_name in covered_by:
            if fixture_name not in fixture_tags:
                fail(f"{feature_id} covered_by unknown fixture {fixture_name}")
            if feature_id != "value_no_producer_dialect_in_verified_vc4kernel":
                if feature_id not in fixture_tags[fixture_name]:
                    fail(f"{feature_id} covered_by {fixture_name} but fixture lacks feature tag")
        if not covered_by:
            isolation = feature.get("covered_by_isolation", [])
            mixed_status = feature.get("mixed_status", "")
            if not isolation or "staged" not in mixed_status:
                fail(f"{feature_id} has no mixed coverage and no staged isolation backstop")
            staged_mixed.append(feature_id)

    print(
        "PASS VC4Value mixed acceptance coverage: "
        f"fixtures={len(fixtures)} features={len(features)} "
        f"staged_mixed={len(staged_mixed)}"
    )
    if staged_mixed:
        print("STAGED_MIXED_ADDITIONS=" + ",".join(staged_mixed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--mode", default="phase5")
    args = parser.parse_args()
    if args.mode not in ("phase5", "phase9", "phase10", "phase11", "phase12", "phase13", "phase14"):
        fail(f"unsupported mode {args.mode!r}")
    validate_manifest(args.repo_root.resolve(), load_json(args.manifest))


if __name__ == "__main__":
    main()
