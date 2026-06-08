#!/usr/bin/env python3
"""Validate the Phase 3 VC4 value-surface support matrix."""

import argparse
import collections
import json
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


def fail(message: str) -> None:
    print(f"FAIL VC4 value surface matrix: {message}", file=sys.stderr)
    sys.exit(1)


def validate_matrix(matrix_path: str, mode: str) -> dict:
    if mode != "phase3-lock":
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

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix")
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()

    validate_matrix(args.matrix, args.mode)
    print(
        "PASS VC4 value surface matrix: mode=phase3-lock features=26 "
        "accepted_surface_contract=9 staged_future_surface_contract=7 "
        "deterministic_reject_surface_policy=8 internal_only=2"
    )


if __name__ == "__main__":
    main()
