#!/usr/bin/env python3
"""Validate the VC4Kernel mixed acceptance manifest."""

import argparse
import json
import sys
from pathlib import Path


TOP_LEVEL_FIELDS = {
    "schema_version",
    "suite_name",
    "fixtures",
    "required_features",
    "deterministic_rejects",
    "future_phase_extension_points",
    "isolated_fixture_policy",
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
    "phase",
    "verification_kind",
    "covered_by",
    "required",
}

REQUIRED_FIXTURES = {
    "mixed_elementwise_tmu_alu_cmp_vdw_vc4kernel",
    "mixed_i32_control_reduce_scalar_address_vc4kernel",
    "mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel",
    "mixed_blocked_gemv_vpm_fullstack_vc4kernel",
    "mixed_blocked_gemm_vpm_fullstack_vc4kernel",
    "mixed_vertical_rect_vpm_vdw_preserve_vc4kernel",
    "mixed_cooperative_barrier_vpm_transpose_vc4kernel",
    "mixed_lower_half_spill_dma_branch_ssavc4",
    "mixed_subword_vpm_pack_unpack_roundtrip_vc4kernel",
    "mixed_quantized_gemv_subword_vpm_vc4kernel",
    "mixed_sfu_activation_tmu_vdw_vc4kernel",
    "mixed_sfu_norm_reduce_vpm_vc4kernel",
}

REQUIRED_FEATURES = {
    "p1_fragment_alu_f32",
    "p1_fragment_alu_i32_logic_shift_minmax_clz",
    "p1_fragment_alu_mul24_v8",
    "p2_fragment_bitcast_const",
    "p3_i32_cmp",
    "p3_f32_cmp_finite",
    "p4_i32_reduce",
    "p4_f32_reduce_finite_tree",
    "p5_scalar_i32_control_address",
    "p5_scalar_i1_bitcast_policy",
    "p6_memory_path_coherency",
    "p6_spill_coherency",
    "p7_tmu_safe_offset",
    "p8_vdw_preserve_register",
    "p8_vdw_preserve_vpm_rect",
    "p8_sparse_store_reject",
    "dynamic_vdr_vdw_horizontal",
    "dynamic_vdr_vdw_vertical",
    "blocked_gemv_runtime_dims",
    "blocked_gemm_runtime_dims",
    "cooperative_barrier_vpm",
    "lower_half_spill_dma_branch",
    "p9_fragment_pack",
    "p9_fragment_unpack",
    "p9_vpm_qpu_subword_modes",
    "p9_vdr_subword_dma",
    "p9_vdw_subword_dma",
    "p9_subword_unsupported_mode_rejects",
    "p9_mixed_subword_roundtrip",
    "p9_mixed_quantized_gemv_subword",
    "p10_fragment_sfu",
    "p10_sfu_approx_policy",
    "p10_sfu_latency_wait",
    "p10_exact_math_reject",
    "p10_mixed_sfu_activation",
    "p10_mixed_sfu_norm_reduce",
}

REQUIRED_REJECTS = {
    "sparse_vdw_store_reject",
    "old_tmu_signature_reject",
    "missing_memory_path_coherency_reject",
    "missing_vdw_inactive_store_reject",
    "f32_cmp_reduce_finite_policy_reject",
    "unsupported_numeric_casts_reject",
    "forbidden_producer_dialect_reject",
    "p9_subword_unsupported_mode_rejects",
    "p10_sfu_fastmath_policy_rejects",
}

FUTURE_EXTENSION_IDS = {
    "p11_dynamic_rotate_mixed",
    "p12_dynamic_vpm_coords_mixed",
    "p13_final_surface_lock",
}

AMBIGUOUS_TOKENS = ("TODO", "TBD", "unknown", "maybe", "??")
FORBIDDEN_VC4KERNEL_DIALECT_TERMS = (
    "memref.",
    "scf.",
    "linalg.",
    "gpu.",
    "tt.",
    "ttg.",
    "triton.",
    "nvgpu.",
    "nvvm.",
    "rocdl.",
    "spirv.",
    "iree.",
    "stablehlo.",
    "mhlo.",
)
FUTURE_PHASE_TAG_PREFIXES = ("p11_", "p12_")


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_strings(child)


def require_dict(value, context):
    if not isinstance(value, dict):
        fail(f"{context} must be an object")
    return value


def require_string(value, context):
    if not isinstance(value, str) or not value.strip():
        fail(f"{context} must be a non-empty string")


def require_string_list(value, context):
    if not isinstance(value, list) or not value:
        fail(f"{context} must be a non-empty list")
    for item in value:
        require_string(item, context)


def load_manifest(path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON: {error}")


def validate_no_ambiguous_text(manifest):
    for text in walk_strings(manifest):
        lowered = text.lower()
        for token in AMBIGUOUS_TOKENS:
            if token.lower() in lowered:
                fail(f'ambiguous token "{token}" appears in manifest text: {text}')


def validate_top_level(manifest):
    require_dict(manifest, "manifest")
    missing = TOP_LEVEL_FIELDS - manifest.keys()
    if missing:
        fail(f"missing top-level fields: {', '.join(sorted(missing))}")
    if manifest["schema_version"] != 1:
        fail("schema_version must be 1")
    if manifest["suite_name"] != "vc4kernel_surface_v2_mixed_acceptance":
        fail("suite_name must be vc4kernel_surface_v2_mixed_acceptance")


def validate_fixtures(manifest):
    fixtures = manifest["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        fail("fixtures must be a non-empty list")
    by_name = {}
    for fixture in fixtures:
        require_dict(fixture, "fixture")
        missing = FIXTURE_FIELDS - fixture.keys()
        if missing:
            ident = fixture.get("name", "<missing name>")
            fail(f"fixture {ident} missing fields: {', '.join(sorted(missing))}")
        name = fixture["name"]
        require_string(name, "fixture name")
        if name in by_name:
            fail(f"duplicate fixture name: {name}")
        by_name[name] = fixture
        if fixture["dialect"] not in {"vc4kernel", "ssavc4"}:
            fail(f"fixture {name} has invalid dialect")
        if fixture["runner_kind"] not in {"vc4kernel_candidate", "ssavc4_candidate"}:
            fail(f"fixture {name} has invalid runner_kind")
        if fixture["status"] not in {"planned", "implemented"}:
            fail(f"fixture {name} has invalid status")
        for field in ("path", "real_world_role", "expected_cases_summary"):
            require_string(fixture[field], f"fixture {name} field {field}")
        for field in ("feature_tags", "required_result_fields", "triage_feature_bands"):
            require_string_list(fixture[field], f"fixture {name} field {field}")
        for tag in fixture["feature_tags"]:
            if tag.startswith(FUTURE_PHASE_TAG_PREFIXES):
                fail(f"fixture {name} uses future phase tag {tag}")
    missing = REQUIRED_FIXTURES - by_name.keys()
    if missing:
        fail(f"missing required fixtures: {', '.join(sorted(missing))}")
    return by_name


def validate_rejects(manifest):
    rejects = manifest["deterministic_rejects"]
    if not isinstance(rejects, list) or not rejects:
        fail("deterministic_rejects must be a non-empty list")
    by_id = {}
    for reject in rejects:
        require_dict(reject, "deterministic reject")
        for field in ("id", "description", "verification_kind", "coverage_entries", "status"):
            if field not in reject:
                fail(f"deterministic reject missing field {field}")
        ident = reject["id"]
        require_string(ident, "deterministic reject id")
        if ident in by_id:
            fail(f"duplicate deterministic reject id: {ident}")
        by_id[ident] = reject
        if reject["verification_kind"] not in {"lit_negative", "audit", "matrix"}:
            fail(f"deterministic reject {ident} has invalid verification_kind")
        if reject["status"] != "implemented":
            fail(f"deterministic reject {ident} must be implemented")
        require_string_list(reject["coverage_entries"], f"deterministic reject {ident} coverage_entries")
    missing = REQUIRED_REJECTS - by_id.keys()
    if missing:
        fail(f"missing deterministic reject entries: {', '.join(sorted(missing))}")
    return by_id


def validate_features(manifest, fixture_by_name, reject_by_id, mode):
    features = manifest["required_features"]
    if not isinstance(features, list) or not features:
        fail("required_features must be a non-empty list")
    feature_by_id = {}
    valid_refs = set(fixture_by_name) | set(reject_by_id) | {"p8_vdw_store_policy_audit"}
    missing_coverage = 0
    for feature in features:
        require_dict(feature, "feature")
        missing = FEATURE_FIELDS - feature.keys()
        if missing:
            ident = feature.get("id", "<missing id>")
            fail(f"feature {ident} missing fields: {', '.join(sorted(missing))}")
        ident = feature["id"]
        require_string(ident, "feature id")
        if ident in feature_by_id:
            fail(f"duplicate feature id: {ident}")
        feature_by_id[ident] = feature
        if feature["verification_kind"] not in {"hardware_mixed", "lit_negative", "audit", "matrix"}:
            fail(f"feature {ident} has invalid verification_kind")
        if feature.get("required") is not True:
            fail(f"feature {ident} must be required")
        require_string_list(feature["covered_by"], f"feature {ident} covered_by")
        for ref in feature["covered_by"]:
            if ref not in valid_refs:
                fail(f"feature {ident} references missing coverage entry {ref}")
        if not feature["covered_by"]:
            missing_coverage += 1
        if mode == "lock" and feature.get("status") == "planned":
            fail(f"feature {ident} is still planned in lock mode")
    missing = REQUIRED_FEATURES - feature_by_id.keys()
    if missing:
        fail(f"missing required feature entries: {', '.join(sorted(missing))}")
    return feature_by_id, missing_coverage


def validate_future_extensions(manifest):
    entries = manifest["future_phase_extension_points"]
    if not isinstance(entries, list) or not entries:
        fail("future_phase_extension_points must be a non-empty list")
    seen = set()
    for entry in entries:
        require_dict(entry, "future extension")
        ident = entry.get("id")
        require_string(ident, "future extension id")
        seen.add(ident)
    missing = FUTURE_EXTENSION_IDS - seen
    if missing:
        fail(f"missing future extension points: {', '.join(sorted(missing))}")


def validate_lock_files(fixture_by_name, repo_root):
    for fixture in fixture_by_name.values():
        name = fixture["name"]
        if fixture["status"] == "planned":
            fail(f"fixture {name} is still planned in lock mode")
        fixture_dir = repo_root / fixture["path"]
        input_path = fixture_dir / "input.mlir"
        expected_path = fixture_dir / "expected.json"
        if not fixture_dir.is_dir():
            fail(f"fixture directory missing: {fixture_dir}")
        if not input_path.is_file():
            fail(f"fixture input.mlir missing: {input_path}")
        if not expected_path.is_file():
            fail(f"fixture expected.json missing: {expected_path}")
        text = input_path.read_text()
        if fixture["dialect"] == "vc4kernel":
            if "vc4kernel.kernel" not in text:
                fail(f"VC4Kernel fixture lacks vc4kernel.kernel: {name}")
            if "ssavc4." in text or " vc4." in text or "\nvc4." in text:
                fail(f"VC4Kernel fixture has source-authored lower-half dialect: {name}")
            lowered = text.lower()
            for term in FORBIDDEN_VC4KERNEL_DIALECT_TERMS:
                if term in lowered:
                    fail(f"VC4Kernel fixture {name} contains forbidden dialect term {term}")
        else:
            if "ssavc4.module" not in text:
                fail(f"SSAVC4 fixture lacks ssavc4.module: {name}")
            if "vc4kernel." in text or " vc4." in text or "\nvc4." in text:
                fail(f"SSAVC4 fixture has forbidden source dialect: {name}")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--mode", choices=("planned", "lock"), required=True)
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    repo_root = Path(args.repo_root)
    if not manifest_path.is_file():
        fail(f"manifest path does not exist: {manifest_path}")
    manifest = load_manifest(manifest_path)
    validate_top_level(manifest)
    validate_no_ambiguous_text(manifest)
    fixture_by_name = validate_fixtures(manifest)
    reject_by_id = validate_rejects(manifest)
    feature_by_id, missing_coverage = validate_features(
        manifest, fixture_by_name, reject_by_id, args.mode)
    validate_future_extensions(manifest)
    if args.mode == "lock":
        validate_lock_files(fixture_by_name, repo_root)

    print(
        "PASS VC4Kernel mixed acceptance coverage: "
        f"mode={args.mode} fixtures={len(fixture_by_name)} "
        f"features={len(feature_by_id)} "
        f"deterministic_rejects={len(reject_by_id)} "
        f"missing_coverage={missing_coverage}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
