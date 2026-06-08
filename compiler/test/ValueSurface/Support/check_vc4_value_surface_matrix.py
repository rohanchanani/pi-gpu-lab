#!/usr/bin/env python3
"""Validate the Phase 3 VC4 value-surface support matrix."""

import argparse
import collections
import json
import pathlib
import sys


EXPECTED_FEATURE_IDS = [
    "value_kernel_wrapper_attrs",
    "vc4value_program_id_num_programs",
    "allowed_standard_dialects_boundary",
    "logical_memref_types_surface",
    "fixed_vector_types_surface",
    "vector_step_splat_mask_surface",
    "vector_transfer_read_write_surface",
    "arith_math_surface_boundary",
    "scf_cf_surface_boundary",
    "support_matrix_and_audit_contract",
    "value_to_vc4kernel_lowering_absent_guard",
    "triton_ttir_direct_ingestion_absent_guard",
    "forbidden_producer_dialects",
    "forbidden_target_dialects",
    "forbidden_vc4value_scope_creep",
    "forbidden_memref_side_effect_ops",
    "forbidden_sparse_store_ops",
    "unsupported_scalable_vectors_and_unranked_memrefs",
    "staged_vector_gather_memory_expansion",
    "staged_vector_reduction",
    "staged_vector_contract",
    "staged_math_sfu_policy",
    "staged_subword_f16_storage",
    "staged_shuffle_rotate_broadcast",
    "staged_vpm_tile_planning",
    "phase3_lock_doc",
]

ALLOWED_GENERATORS = {
    "Phase3b_value_surface_matrix_seed",
    "Phase3f_value_surface_audit_lock",
    "Phase3_5b_vector_abstraction_refinement",
}

REQUIRED_FIELDS = {
    "feature_id",
    "status",
    "phase_target",
    "source_layers",
    "value_surface_form",
    "required_vc4kernel_contract",
    "diagnostic_policy",
    "verification",
    "notes",
}

ALLOWED_STATUSES = {
    "accepted_surface_contract",
    "staged_future_surface_contract",
    "deterministic_reject_surface_policy",
    "internal_only",
}

EXPECTED_COUNTS = {
    "accepted_surface_contract": 9,
    "staged_future_surface_contract": 7,
    "deterministic_reject_surface_policy": 8,
    "internal_only": 2,
}

PHASE3_5_EQUIVALENTS = {
    "value.vector.fixed_rank1.width16_fragment_carriers": (
        "fixed_vector_types_surface",
        "accept",
        "lowerable_v1",
    ),
    "value.vector.fixed_rank1.non16_staged_split": (
        "fixed_vector_types_surface",
        "accept",
        "staged_split_required",
    ),
    "value.vector.fixed_rank2.staged_tile_contract": (
        "fixed_vector_types_surface",
        "accept",
        "staged_tile_or_contract_required",
    ),
    "value.vector.scalable_reject": (
        "unsupported_scalable_vectors_and_unranked_memrefs",
        "reject",
        "rejected",
    ),
    "value.vector.unsupported_element_reject": (
        "unsupported_scalable_vectors_and_unranked_memrefs",
        "reject",
        "rejected",
    ),
    "value.tensor_linalg.initial_reject": (
        "forbidden_producer_dialects",
        "reject",
        "rejected",
    ),
    "value.vector.f16_storage_not_native_arith": (
        "staged_subword_f16_storage",
        "accept",
        "staged_storage_only",
    ),
}

FORBIDDEN_VECTOR_LIMIT_PHRASES = [
    "vector<16> is the only legal value-layer vector shape",
    "vector<16xT> is the only legal value-layer vector shape",
    "only legal value-layer vector shape is vector<16>",
    "only legal value-layer vector shape is vector<16xT>",
]


def fail(message: str) -> None:
    print(f"FAIL VC4 value surface matrix: {message}", file=sys.stderr)
    sys.exit(1)


def get_feature(features: list[dict], feature_id: str) -> dict:
    for row in features:
        if row.get("feature_id") == feature_id:
            return row
    fail(f"missing feature row {feature_id}")


def lowering_behavior_for(row: dict, equivalent_id: str):
    behavior = row.get("phase5_lowering_behavior")
    if isinstance(behavior, dict):
        return behavior.get(equivalent_id)
    return behavior


def validate_phase3_5_vector_abstraction(data: dict, matrix_path: str) -> None:
    features = data["features"]
    resolved_matrix_path = pathlib.Path(matrix_path).resolve()
    for equivalent_id, (feature_id, surface_behavior, lowering_behavior) in (
        PHASE3_5_EQUIVALENTS.items()
    ):
        row = get_feature(features, feature_id)
        equivalents = row.get("phase3_5_equivalent_feature_ids")
        if not isinstance(equivalents, list) or equivalent_id not in equivalents:
            fail(f"{feature_id} missing Phase 3.5 equivalent {equivalent_id}")
        if row.get("surface_verifier_behavior") != surface_behavior:
            fail(
                f"{feature_id} must set surface_verifier_behavior="
                f"{surface_behavior!r}"
            )
        actual_lowering = lowering_behavior_for(row, equivalent_id)
        if actual_lowering != lowering_behavior:
            fail(
                f"{feature_id} must classify {equivalent_id} as "
                f"{lowering_behavior!r}, got {actual_lowering!r}"
            )

    fixed_row = get_feature(features, "fixed_vector_types_surface")
    fixed_text = json.dumps(fixed_row, sort_keys=True)
    if "not the global value-layer type limit" not in fixed_text:
        fail("fixed vector row must say vector<16> is not the global value-layer type limit")

    f16_row = get_feature(features, "staged_subword_f16_storage")
    f16_text = json.dumps(f16_row, sort_keys=True)
    if "native f16 arithmetic is not accepted" not in f16_text:
        fail("f16 storage row must explicitly reject native f16 arithmetic")

    matrix_text = resolved_matrix_path.read_text(encoding="utf-8")
    for phrase in FORBIDDEN_VECTOR_LIMIT_PHRASES:
        if phrase in matrix_text:
            fail(f"matrix claims forbidden global vector<16> limit: {phrase}")

    docs_root = resolved_matrix_path.parent
    for doc in docs_root.glob("vc4*phase3*.md"):
        text = doc.read_text(encoding="utf-8")
        if "READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES" in text:
            fail(f"{doc} claims value-to-vc4kernel lowering is ready")
        if "READY_FOR_TRITON=YES" in text:
            fail(f"{doc} claims Triton is ready")
        for phrase in FORBIDDEN_VECTOR_LIMIT_PHRASES:
            if phrase in text:
                fail(f"{doc} claims forbidden global vector<16> limit: {phrase}")

    repo_root = docs_root.parent.parent
    for root in [repo_root / "compiler/docs", repo_root / "compiler/test"]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "Support" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for phrase in FORBIDDEN_VECTOR_LIMIT_PHRASES:
                if phrase in text:
                    fail(f"{path} claims forbidden global vector<16> limit: {phrase}")
            if "audit" in path.name or path.name.startswith("invalid-"):
                continue
            if "READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES" in text:
                fail(f"{path} claims value-to-vc4kernel lowering is ready")
            if "READY_FOR_TRITON=YES" in text:
                fail(f"{path} claims Triton is ready")


def validate_matrix(matrix_path: str, mode: str) -> dict:
    if mode not in {"phase3-lock", "phase3_5"}:
        fail(f"unsupported mode {mode!r}")

    try:
        with open(matrix_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        fail(f"could not parse JSON: {exc}")

    if data.get("schema_version") != 1:
        fail("schema_version must be 1")
    if data.get("generated_by") not in ALLOWED_GENERATORS:
        fail(f"unexpected generated_by {data.get('generated_by')!r}")
    if data.get("phase") != "Phase3":
        fail("phase must be Phase3")

    features = data.get("features")
    if not isinstance(features, list):
        fail("features must be a list")
    if len(features) != 26:
        fail(f"expected 26 features, got {len(features)}")

    ids = [row.get("feature_id") for row in features]
    if ids != EXPECTED_FEATURE_IDS:
        fail("feature IDs/order do not match Phase 3 contract")
    if len(ids) != len(set(ids)):
        fail("feature_id values must be unique")

    counts = collections.Counter()
    for row in features:
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            fail(f"{row.get('feature_id')} missing fields: {sorted(missing)}")
        status = row.get("status")
        if status not in ALLOWED_STATUSES:
            fail(f"{row.get('feature_id')} has invalid status {status!r}")
        counts[status] += 1
        if status == "deterministic_reject_surface_policy":
            diag = row.get("diagnostic_policy")
            if not isinstance(diag, str) or not diag.strip():
                fail(f"{row.get('feature_id')} must have diagnostic_policy")
        if status == "staged_future_surface_contract":
            if row.get("phase_target") == "Phase3":
                fail(f"{row.get('feature_id')} staged row must target post-Phase3")

    if dict(counts) != EXPECTED_COUNTS:
        fail(f"unexpected status counts {dict(counts)}")

    validate_phase3_5_vector_abstraction(data, matrix_path)

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    parser.add_argument("--mode", default="phase3-lock")
    args = parser.parse_args()

    validate_matrix(args.matrix, args.mode)
    print(
        f"PASS VC4 value surface matrix: mode={args.mode} features=26 "
        "accepted_surface_contract=9 staged_future_surface_contract=7 "
        "deterministic_reject_surface_policy=8 internal_only=2"
    )


if __name__ == "__main__":
    main()
