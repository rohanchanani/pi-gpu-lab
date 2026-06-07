#!/usr/bin/env python3
"""Validate the VC4Kernel Surface v2 support matrix."""

import json
import sys
from collections import Counter
from pathlib import Path


TOP_LEVEL_FIELDS = {
    "schema_version",
    "generated_by",
    "phase_order",
    "locked_policies",
    "features",
}

FEATURE_FIELDS = {
    "id",
    "phase",
    "category",
    "current_status",
    "final_status_target",
    "current_surface",
    "target_surface",
    "vc4kernel_verifier_plan",
    "vc4kernel_to_ssavc4_plan",
    "ssavc4_to_vc4_plan",
    "hardware_or_reject_verification_plan",
    "upstream_use",
    "migration_or_removal",
    "dependencies",
    "acceptance_line",
    "proof_links",
    "reject_category",
}

REQUIRED_PHASES = [f"P{i}" for i in range(9)] + [
    "P8_5",
] + [f"P{i}" for i in range(9, 14)] + ["REJECTED_SPARSE_VDW_STORE"]

REQUIRED_POLICIES = {
    "generalize_not_special_case",
    "fastmath_opt_in",
    "vdw_sparse_store_reject",
    "no_tile_dsl",
    "only_vc4kernel_to_ssavc4",
    "hardware_or_deterministic_reject_required",
    "mixed_fixture_claim_contract",
}

REQUIRED_FEATURE_IDS = {
    "p1_general_fragment_add_alu",
    "p1_general_fragment_mul_alu",
    "p1_remove_fragment_add",
    "p1_remove_fragment_sub",
    "p1_remove_fragment_mul",
    "p1_remove_fragment_shl",
    "p2_fragment_bitcast",
    "p2_fragment_const_splat_zero_allones_lane_affine",
    "p3_fragment_cmp_signed_i32",
    "p3_fragment_cmp_ordered_f32",
    "p4_fragment_reduce_general",
    "p4_remove_add_only_reduce_specialness",
    "p5_scalar_arith_bitwise_shift_minmax_casts",
    "p6_memory_path_coherency_policy",
    "p7_tmu_load_explicit_safe_inactive_offset",
    "p7_remove_old_tmu_load_signature",
    "p8_vdw_store_inactive_preserve_full_tail_rect",
    "p8_sparse_vdw_store_deterministic_reject",
    "p8_5_mixed_acceptance_policy",
    "p9_fragment_pack",
    "p9_fragment_unpack",
    "p9_vpm_subword_w16_w8_modes",
    "p10_fragment_sfu_recip",
    "p10_fragment_sfu_rsqrt",
    "p10_fragment_sfu_exp",
    "p10_fragment_sfu_log",
    "p10_fastmath_approx_contract",
    "p10_sqrt_policy",
    "p11_fragment_rotate_dynamic_if_hardware",
    "p11_fragment_broadcast_lane_if_supported",
    "p12_dynamic_vpm_read_write_coordinates",
    "p12_dynamic_vdr_vdw_coordinates",
    "p13_surface_support_matrix_final",
    "p13f16_f16_storage_conversion",
    "p13f16_native_f16_arithmetic_reject",
    "p13f16_bf16_fp8_native_reject",
    "sparse_vdw_store_general_masks_reject",
}

SPECIAL_CASE_FEATURE_IDS = {
    "p1_remove_fragment_add",
    "p1_remove_fragment_sub",
    "p1_remove_fragment_mul",
    "p1_remove_fragment_shl",
    "p4_remove_add_only_reduce_specialness",
    "p7_remove_old_tmu_load_signature",
}

SPECIAL_CASE_ALLOWED_STATUSES = {
    "deterministic_reject",
}

FORBIDDEN_TILE_TERMS = {
    "tile_broadcast",
    "tile_dot",
    "tile_matmul",
    "tile_contract",
    "fragment_contract",
}

AMBIGUOUS_TOKENS = (
    "TBD",
    "TODO",
    "FIXME",
    "unknown",
    "maybe",
    "??",
    "implemented_pending_hardware",
    "pending_hardware",
    "pending final",
    "planned",
    "deferred",
    "unproven",
    " later ",
    " v1",
    "compatibility",
)

FINAL_STATUSES = {
    "accepted_hardware_proven",
    "deterministic_reject",
    "internal_only",
    "out_of_scope_non_compute_hardware",
    "producer_layer_future_work",
}

REJECT_CATEGORIES = {
    "hardware_forbidden",
    "static_surface_policy",
    "not_meaningful",
    "unsupported_source_layer_inside_vc4kernel",
    "out_of_scope_non_compute_hardware",
}


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


def load_matrix(path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON: {error}")


def validate_top_level(matrix):
    require_dict(matrix, "matrix")
    missing = TOP_LEVEL_FIELDS - matrix.keys()
    if missing:
        fail(f"missing top-level fields: {', '.join(sorted(missing))}")
    if matrix["schema_version"] != 1:
        fail("schema_version must be 1")
    if matrix["generated_by"] != "P13c_final_surface_lock":
        fail('generated_by must be "P13c_final_surface_lock"')
    phases = matrix["phase_order"]
    if not isinstance(phases, list):
        fail("phase_order must be a list")
    for phase in REQUIRED_PHASES:
        if phase not in phases:
            fail(f"phase_order missing {phase}")
    policies = require_dict(matrix["locked_policies"], "locked_policies")
    missing_policies = REQUIRED_POLICIES - policies.keys()
    if missing_policies:
        fail(f"missing locked policies: {', '.join(sorted(missing_policies))}")
    if policies["fastmath_opt_in"].get("opt_in_required") is not True:
        fail("fastmath policy is not opt-in")
    sparse_policy = policies["vdw_sparse_store_reject"]
    if sparse_policy.get("deterministic_reject") is not True:
        fail("sparse VDW store policy is not deterministic-reject")
    if policies["mixed_fixture_claim_contract"].get("required") is not True:
        fail("mixed fixture claim contract is not required")


def validate_no_ambiguous_text(matrix):
    matrix_without_features = dict(matrix)
    matrix_without_features["features"] = []
    for text in walk_strings(matrix_without_features):
        lowered = text.lower()
        for token in AMBIGUOUS_TOKENS:
            if token.lower() in lowered:
                fail(f'ambiguous token "{token}" appears in text: {text}')


def validate_feature(feature, phases):
    require_dict(feature, "feature")
    missing = FEATURE_FIELDS - feature.keys()
    if missing:
        ident = feature.get("id", "<missing id>")
        fail(f"feature {ident} missing fields: {', '.join(sorted(missing))}")
    for field in FEATURE_FIELDS - {"dependencies", "proof_links", "reject_category"}:
        require_string(feature[field], f"feature {feature['id']} field {field}")
    if feature["phase"] not in phases:
        fail(f"feature {feature['id']} uses phase not in phase_order: {feature['phase']}")
    if not isinstance(feature["dependencies"], list):
        fail(f"feature {feature['id']} dependencies must be a list")
    for dependency in feature["dependencies"]:
        require_string(dependency, f"feature {feature['id']} dependency")
    if feature["id"] in SPECIAL_CASE_FEATURE_IDS:
        if feature["current_status"] not in SPECIAL_CASE_ALLOWED_STATUSES:
            fail(
                f"current special-case {feature['id']} must be a deterministic reject"
            )
    text = "\n".join(walk_strings(feature))
    for feature_text in walk_strings(feature):
        lowered = feature_text.lower()
        for token in AMBIGUOUS_TOKENS:
            if token.lower() not in lowered:
                continue
            fail(
                f'ambiguous token "{token}" appears in feature '
                f"{feature['id']}: {feature_text}"
            )
    if any(term in text for term in FORBIDDEN_TILE_TERMS):
        if feature["current_status"] in {
            "accepted_hardware_proven",
            "implemented",
        }:
            fail(f"forbidden tile DSL feature appears planned or accepted: {feature['id']}")
    if feature["current_status"] not in FINAL_STATUSES:
        fail(f"feature {feature['id']} has non-final status {feature['current_status']}")
    if feature["final_status_target"] != feature["current_status"]:
        fail(f"feature {feature['id']} final_status_target must equal current_status")
    proof_links = require_dict(feature["proof_links"], f"feature {feature['id']} proof_links")
    if feature["current_status"] == "accepted_hardware_proven":
        for key in (
            "verifier_lit",
            "conversion_lit",
            "isolated_hardware_fixture",
            "mixed_hardware_fixture_or_feature",
            "mixed_claim_contract",
        ):
            require_string(proof_links.get(key), f"feature {feature['id']} proof_links {key}")
        if feature.get("reject_category"):
            fail(f"accepted feature {feature['id']} must not have reject_category")
    elif feature["current_status"] == "deterministic_reject":
        if feature.get("reject_category") not in REJECT_CATEGORIES:
            fail(f"feature {feature['id']} has invalid reject_category")
        if not proof_links:
            fail(f"deterministic reject {feature['id']} lacks proof_links")
    elif feature["current_status"] == "internal_only":
        if not proof_links:
            fail(f"internal feature {feature['id']} lacks proof_links")
    if "sparse_vdw" in feature["id"]:
        sparse_text = text.lower()
        if "deterministic-reject" not in sparse_text:
            fail(f"sparse VDW feature {feature['id']} lacks deterministic-reject policy")
        if feature["current_status"] != "deterministic_reject":
            fail(f"sparse VDW feature {feature['id']} has invalid current_status")
    if feature["id"] == "p11_fragment_broadcast_lane_if_supported":
        status = feature["current_status"]
        if status != "accepted_hardware_proven":
            fail("P11 lane broadcast row must be accepted_hardware_proven")
        broadcast_text = text.lower()
        for token in [
            "composite",
            "lane_range",
            "fragment_cmp",
            "fragment_select",
            "fragment_reduce",
            "fragment_bitcast",
            "finite numeric",
            "finite_tree",
            "nan/inf/signed-zero",
            "not the generic f32 broadcast",
            "no vc4kernel.fragment_broadcast_lane",
            "arbitrary shuffle",
            "deterministic reject",
        ]:
            if token not in broadcast_text:
                fail(f"P11 lane broadcast row must document {token}")


def validate_features(matrix):
    features = matrix["features"]
    if not isinstance(features, list) or not features:
        fail("features must be a non-empty list")
    phases = set(matrix["phase_order"])
    seen = set()
    for feature in features:
        validate_feature(feature, phases)
        ident = feature["id"]
        if ident in seen:
            fail(f"duplicate feature id: {ident}")
        seen.add(ident)
    missing = REQUIRED_FEATURE_IDS - seen
    if missing:
        fail(f"missing required feature ids: {', '.join(sorted(missing))}")
    if not any(feature["current_status"] == "accepted_hardware_proven" for feature in features):
        fail("matrix must include accepted_hardware_proven entries")
    return features


def print_summary(features):
    by_phase = Counter(feature["phase"] for feature in features)
    by_status = Counter(feature["current_status"] for feature in features)
    phase_summary = ", ".join(
        f"{phase}={by_phase[phase]}" for phase in sorted(by_phase)
    )
    status_summary = ", ".join(
        f"{status}={by_status[status]}" for status in sorted(by_status)
    )
    print(f"PASS VC4Kernel Surface v2 matrix: features={len(features)}")
    print(f"phase_counts: {phase_summary}")
    print(f"current_status_counts: {status_summary}")


def main(argv):
    if len(argv) not in {2, 4}:
        print("usage: check_vc4kernel_surface_v2_matrix.py MATRIX_JSON [--mode final]", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if len(argv) == 4 and argv[2:] != ["--mode", "final"]:
        print("usage: check_vc4kernel_surface_v2_matrix.py MATRIX_JSON [--mode final]", file=sys.stderr)
        return 2
    if not path.is_file():
        fail(f"matrix path does not exist: {path}")
    matrix = load_matrix(path)
    validate_top_level(matrix)
    validate_no_ambiguous_text(matrix)
    features = validate_features(matrix)
    print_summary(features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
