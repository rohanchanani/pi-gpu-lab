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

PHASE4_ABI_EQUIVALENTS = {
    "value_kernel_func_wrapper_abi": (
        "value_kernel_wrapper_attrs",
        "accepted_public_abi_contract",
    ),
    "vc4value_global_memory_space_attr": (
        "logical_memref_types_surface",
        "accepted_public_abi_contract",
    ),
    "public_arg_name_attr_schema": (
        "value_kernel_wrapper_attrs",
        "accepted_public_abi_contract",
    ),
    "public_memref_direction_attr_schema": (
        "logical_memref_types_surface",
        "accepted_public_abi_contract",
    ),
    "public_scalar_role_attr_schema": (
        "value_kernel_wrapper_attrs",
        "accepted_public_abi_contract",
    ),
    "public_memref_shape_args_metadata": (
        "logical_memref_types_surface",
        "accepted_public_abi_contract",
    ),
    "public_memref_stride_args_metadata": (
        "logical_memref_types_surface",
        "accepted_public_abi_contract",
    ),
    "rank1_rank2_global_memref_i8_i16_i32_f16_f32_abi": (
        "logical_memref_types_surface",
        "abi_admissible_with_staged_subsets",
    ),
    "public_memref_dim_metadata_only": (
        "logical_memref_types_surface",
        "accepted_metadata_only",
    ),
    "public_vector_args_reject": (
        "fixed_vector_types_surface",
        "reject_public_kernel_argument_only_body_vectors_remain_surface_admissible",
    ),
    "public_tensor_args_reject": (
        "forbidden_producer_dialects",
        "reject_public_kernel_argument_and_initial_value_surface_tensor_policy",
    ),
    "hidden_memref_descriptor_reject": (
        "forbidden_memref_side_effect_ops",
        "deterministic_reject_no_hidden_descriptor_abi",
    ),
    "phase5_v1_i32_f32_rank1_contiguous_lowerable_subset": (
        "logical_memref_types_surface",
        "future_phase5_v1_lowerable_candidate",
    ),
}

FORBIDDEN_VECTOR_LIMIT_PHRASES = [
    "vector<16> is the only legal value-layer vector shape",
    "vector<16xT> is the only legal value-layer vector shape",
    "only legal value-layer vector shape is vector<16>",
    "only legal value-layer vector shape is vector<16xT>",
]

PHASE4_SCOPED_LOWERING_READINESS = (
    "READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES_FOR_PHASE5_ELEMENTWISE_V1"
)

PHASE8_CF_BEHAVIOR = {
    "cf.br": "accepted_executable_phase8_pending_hardware",
    "cf.cond_br": "accepted_executable_phase8_pending_hardware",
    "value.block_args.index": "lowerable_phase8_v1",
    "value.block_args.i1": "lowerable_phase8_v1",
    "value.block_args.i32": "lowerable_phase8_v1",
    "value.block_args.f32": "lowerable_phase8_v1",
    "value.block_args.vector16xi1": "lowerable_phase8_v1",
    "value.block_args.vector16xindex": "lowerable_phase8_v1",
    "value.block_args.vector16xi32": "lowerable_phase8_v1",
    "value.block_args.vector16xf32": "lowerable_phase8_v1",
    "scf.if": "surface_admissible_canonicalization_required",
    "scf.for": "surface_admissible_canonicalization_required",
    "scf.while": "surface_admissible_canonicalization_required_phase8r",
    "nested_structured_cf": "support_now_phase8r",
    "tl_range_style_loop_skeleton": "support_now_value_level_phase8r",
    "persistent_loop_skeleton": "support_now_value_level_phase8r",
    "scf_inside_vc4kernel": "deterministic_reject",
    "ttir_control_flow": "staged_future_ttir_import",
    "vector_valued_branch_conditions": "deterministic_reject_as_cfg_use_masks",
    "memref_block_args": "deterministic_reject_until_proven",
    "cf.switch": "staged_with_proof_phase8r",
    "scf.index_switch": "staged_with_proof_phase8r",
    "scf.parallel": "staged_reject_parallel_semantics",
    "scf.forall": "staged_reject_parallel_semantics",
    "scf.reduce": "staged_reject_reduction_semantics",
    "irreducible_cfg": "probe_not_required_for_sane_triton",
}


def has_forbidden_lowering_ready_claim(text: str) -> bool:
    """Reject broad YES claims while allowing the scoped Phase 4 handoff line."""
    return (
        "READY_FOR_VALUE_TO_VC4KERNEL_LOWERING=YES"
        in text.replace(PHASE4_SCOPED_LOWERING_READINESS, "")
    )


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


def phase4_behavior_for(row: dict, equivalent_id: str):
    behavior = row.get("phase4_abi_behavior")
    if isinstance(behavior, dict):
        return behavior.get(equivalent_id)
    return None


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
        if has_forbidden_lowering_ready_claim(text):
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
            if has_forbidden_lowering_ready_claim(text):
                fail(f"{path} claims value-to-vc4kernel lowering is ready")
            if "READY_FOR_TRITON=YES" in text:
                fail(f"{path} claims Triton is ready")


def validate_phase4_abi_lock(data: dict, matrix_path: str) -> None:
    features = data["features"]
    resolved_matrix_path = pathlib.Path(matrix_path).resolve()

    forbidden_status_words = {"unknown", "planned", "deferred", "unclassified"}
    for row in features:
        status = row.get("status")
        if any(word in status for word in forbidden_status_words):
            fail(f"{row.get('feature_id')} has nonfinal ABI status {status!r}")

    for equivalent_id, (feature_id, expected_behavior) in PHASE4_ABI_EQUIVALENTS.items():
        row = get_feature(features, feature_id)
        equivalents = row.get("phase4_abi_equivalent_feature_ids")
        if not isinstance(equivalents, list) or equivalent_id not in equivalents:
            fail(f"{feature_id} missing Phase 4 ABI equivalent {equivalent_id}")
        actual_behavior = phase4_behavior_for(row, equivalent_id)
        if actual_behavior != expected_behavior:
            fail(
                f"{feature_id} must classify {equivalent_id} as "
                f"{expected_behavior!r}, got {actual_behavior!r}"
            )

    memref_row = get_feature(features, "logical_memref_types_surface")
    memref_text = json.dumps(memref_row, sort_keys=True)
    for phrase in ["i8/i16/i32/f16/f32", "no hidden descriptors", "memref.dim as metadata only"]:
        if phrase not in memref_text:
            fail(f"logical memref row must document Phase 4 ABI phrase: {phrase}")
    if (
        phase4_behavior_for(
            memref_row, "rank1_rank2_global_memref_i8_i16_i32_f16_f32_abi"
        )
        == phase4_behavior_for(
            memref_row, "phase5_v1_i32_f32_rank1_contiguous_lowerable_subset"
        )
    ):
        fail("memref ABI row must distinguish ABI-admissible forms from Phase 5 V1 candidates")

    memref_text_lower = memref_text.lower()
    if "i8/i16/i32/f16/f32" in memref_text and "staged" not in memref_text_lower:
        fail("i8/i16/f16/rank2 memref ABI row must include staged subset language")

    f16_row = get_feature(features, "staged_subword_f16_storage")
    f16_text = json.dumps(f16_row, sort_keys=True)
    if "native f16 arithmetic is not accepted" not in f16_text:
        fail("f16 row must reject native f16 arithmetic")
    if "f32 compute" not in f16_text:
        fail("f16 row must state f16 storage uses future f32 compute path")

    hidden_row = get_feature(features, "forbidden_memref_side_effect_ops")
    if hidden_row.get("status") != "deterministic_reject_surface_policy":
        fail("hidden memref descriptor equivalent must live on a deterministic reject row")
    vector_row = get_feature(features, "fixed_vector_types_surface")
    if vector_row.get("status") != "accepted_surface_contract":
        fail("public vector arg rejection must not make body vector values rejected")
    tensor_row = get_feature(features, "forbidden_producer_dialects")
    if tensor_row.get("status") != "deterministic_reject_surface_policy":
        fail("public tensor args must remain deterministic rejects")

    matrix_text = resolved_matrix_path.read_text(encoding="utf-8")
    if "READY_FOR_TRITON=YES" in matrix_text:
        fail("matrix claims Triton readiness")
    if "native f16 arithmetic is accepted" in matrix_text:
        fail("matrix claims native f16 arithmetic")

    docs_root = resolved_matrix_path.parent
    for doc in docs_root.glob("vc4*phase4*.md"):
        text = doc.read_text(encoding="utf-8")
        if "READY_FOR_TRITON=YES" in text:
            fail(f"{doc} claims Triton is ready")


def validate_phase8_control_flow_policy(data: dict, matrix_path: str) -> None:
    features = data["features"]
    control_row = get_feature(features, "scf_cf_surface_boundary")
    behavior = control_row.get("phase8_control_flow_behavior")
    if not isinstance(behavior, dict):
        fail("scf/cf row missing phase8_control_flow_behavior")
    for key, expected in PHASE8_CF_BEHAVIOR.items():
        if behavior.get(key) != expected:
            fail(
                f"scf/cf row must classify {key} as {expected!r}, "
                f"got {behavior.get(key)!r}"
            )

    control_text = json.dumps(control_row, sort_keys=True)
    for phrase in [
        "scf-to-cf",
        "scf.while",
        "tl.range-style loop skeleton",
        "persistent-loop skeleton",
        "vector branch conditions are rejected as CFG",
        "irreducible CFG is probe/classify only",
        "Executable value-to-VC4Kernel lowering consumes cf, not raw scf",
        "TTIR control flow is staged_future_ttir_import",
        "READY_FOR_TRITON remains NO",
    ]:
        if phrase not in control_text:
            fail(f"scf/cf row missing Phase 8 phrase: {phrase}")
    if "vc4value." in control_text and "program_id" not in control_text:
        fail("scf/cf row must not assign control-flow semantics to vc4value ops")

    docs_root = pathlib.Path(matrix_path).resolve().parent
    for doc_name in [
        "vc4_value_surface_spec.md",
        "vc4_value_to_vc4kernel_planning.md",
    ]:
        doc = docs_root / doc_name
        text = doc.read_text(encoding="utf-8")
        for phrase in [
            "Phase 8 is a value-layer control-flow phase",
            "consumes `cf`, not raw `scf`",
            "scf-to-cf",
            "Verified VC4Kernel output contains no producer dialect operations",
            "TTIR control flow",
            "READY_FOR_TRITON remains NO",
        ]:
            if phrase not in text:
                fail(f"{doc_name} missing Phase 8 control-flow phrase: {phrase}")
        if "READY_FOR_TRITON=YES" in text:
            fail(f"{doc_name} claims Triton readiness")


def validate_matrix(matrix_path: str, mode: str) -> dict:
    if mode not in {"phase3-lock", "phase3_5", "phase4-abi-lock"}:
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

    validate_phase8_control_flow_policy(data, matrix_path)
    validate_phase3_5_vector_abstraction(data, matrix_path)
    if mode == "phase4-abi-lock":
        validate_phase4_abi_lock(data, matrix_path)

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
