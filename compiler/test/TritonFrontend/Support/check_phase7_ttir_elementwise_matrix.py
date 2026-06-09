#!/usr/bin/env python3
"""Validate the Phase 7 TTIR elementwise V1 support matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {
    "accepted_hardware_proven",
    "accepted_static_proven",
    "staged_future_work",
    "deterministic_reject_surface_policy",
    "deterministic_reject_hardware_forbidden",
    "internal_only",
}

REQUIRED_ACCEPTED_HARDWARE = {
    "real_ttir_parse_and_import",
    "ttir_to_value_cpp_conversion",
    "ttir_program_id_axis0_to_vc4value",
    "ttir_make_range_0_16_to_vector_step",
    "ttir_contiguous_addptr_to_value_transfer",
    "ttir_masked_load_other_zero_to_transfer_read",
    "ttir_masked_store_tail_to_transfer_write",
    "ttir_f32_add_mul_cmp_select_elementwise",
    "ttir_i32_add_cmp_select_elementwise",
    "ttir_tail_mask_overlaunch_active_qpus12",
    "ttir_to_value_to_vc4kernel_to_hardware",
    "ttir_mixed_acceptance_claim_audit",
}

REQUIRED_POLICY_ROWS = {
    "ttir_direct_to_lower_half_paths",
    "ttgir_nvidia_inputs_rejected",
    "phase7_python_semantic_importer_retired",
}

REQUIRED_STAGED = {
    "ttir_reduce",
    "ttir_dot",
    "block_pointers",
    "gather_load",
    "scatter_store",
    "axis1_axis2_program_ids",
    "block_size_not_16",
    "f16_bf16_int8_memory",
    "math_sfu",
    "scf_control_flow",
}

FORBIDDEN_PHASE7_ACCEPTED_STATUSES = {
    "planned",
    "deferred",
    "unknown",
    "implemented_pending_hardware",
}


def fail(message: str) -> None:
    print(f"check_phase7_ttir_elementwise_matrix.py: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"failed to read JSON {path}: {exc}")


def require_repo_path(repo_root: Path, rel: str, context: str) -> None:
    path = repo_root / rel
    if not path.exists():
        fail(f"{context} path does not exist: {rel}")


def validate_matrix(repo_root: Path, matrix_path: Path, mode: str) -> None:
    data = load_json(matrix_path)
    required_top = {
        "schema_version",
        "phase",
        "matrix_name",
        "ready_for_triton",
        "accepted_input_snapshots",
        "required_path",
        "status_values",
        "features",
    }
    if set(data) != required_top:
        fail(
            "top-level fields mismatch "
            f"missing={sorted(required_top - set(data))} "
            f"extra={sorted(set(data) - required_top)}"
        )
    if data["schema_version"] != 1:
        fail("schema_version must be 1")
    if data["ready_for_triton"] != "NO":
        fail("ready_for_triton must remain NO")
    if data["required_path"] != ["ttir", "vc4value", "vc4kernel", "ssavc4", "scheduled_vc4", "hardware"]:
        fail("required_path must preserve TTIR -> value -> vc4kernel -> ssavc4 -> scheduled_vc4 -> hardware")
    if set(data["status_values"]) != ALLOWED_STATUSES:
        fail("status_values does not match the locked Phase 7 status vocabulary")

    snapshots = data["accepted_input_snapshots"]
    if not isinstance(snapshots, list) or len(snapshots) != 3:
        fail("accepted_input_snapshots must list the three Phase 6 real TTIR snapshots")
    for rel in snapshots:
        require_repo_path(repo_root, rel, "accepted_input_snapshots")

    features = data["features"]
    if not isinstance(features, list) or not features:
        fail("features must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in features:
        if set(row) != {"id", "status", "summary", "proof_paths", "hardware_fixtures"}:
            fail(f"feature row has invalid fields: {row.get('id', '<missing>')}")
        feature_id = row["id"]
        if feature_id in by_id:
            fail(f"duplicate feature id: {feature_id}")
        by_id[feature_id] = row
        status = row["status"]
        if status not in ALLOWED_STATUSES:
            fail(f"{feature_id} has invalid status {status!r}")
        if status in FORBIDDEN_PHASE7_ACCEPTED_STATUSES:
            fail(f"{feature_id} uses forbidden pending status {status}")
        if not isinstance(row["summary"], str) or not row["summary"].strip():
            fail(f"{feature_id} summary must be non-empty")
        proof_paths = row["proof_paths"]
        if not isinstance(proof_paths, list) or not proof_paths:
            fail(f"{feature_id} proof_paths must be non-empty")
        for rel in proof_paths:
            require_repo_path(repo_root, rel, f"{feature_id}.proof_paths")
        hardware_fixtures = row["hardware_fixtures"]
        if not isinstance(hardware_fixtures, list):
            fail(f"{feature_id} hardware_fixtures must be a list")
        for rel in hardware_fixtures:
            require_repo_path(repo_root, rel, f"{feature_id}.hardware_fixtures")

    missing_accepted = REQUIRED_ACCEPTED_HARDWARE - set(by_id)
    missing_policy = REQUIRED_POLICY_ROWS - set(by_id)
    missing_staged = REQUIRED_STAGED - set(by_id)
    if missing_accepted:
        fail(f"missing accepted hardware rows: {sorted(missing_accepted)}")
    if missing_policy:
        fail(f"missing required policy rows: {sorted(missing_policy)}")
    if missing_staged:
        fail(f"missing staged rows: {sorted(missing_staged)}")

    for feature_id in REQUIRED_ACCEPTED_HARDWARE:
        row = by_id[feature_id]
        if row["status"] != "accepted_hardware_proven":
            fail(f"{feature_id} must be accepted_hardware_proven")
        if mode in ("lock", "final") and not row["hardware_fixtures"]:
            fail(f"{feature_id} must reference Phase 7 hardware fixtures in {mode} mode")
        for fixture in row["hardware_fixtures"]:
            if "Triton/Hardware/Run/" not in fixture:
                fail(f"{feature_id} hardware fixture must be a Phase 7 TTIR fixture: {fixture}")
        row_text = json.dumps(row, sort_keys=True)
        for legacy in ("vc4-triton-import", "lower-elementwise-v1"):
            if legacy in row_text:
                fail(f"{feature_id} accepted hardware row still cites retired Python semantic path {legacy!r}")

    for feature_id in REQUIRED_POLICY_ROWS:
        if by_id[feature_id]["status"] != "deterministic_reject_surface_policy":
            fail(f"{feature_id} must be deterministic_reject_surface_policy")

    for feature_id in REQUIRED_STAGED:
        if by_id[feature_id]["status"] != "staged_future_work":
            fail(f"{feature_id} must remain staged_future_work in Phase 7")

    matrix_text = matrix_path.read_text(encoding="utf-8")
    for forbidden in FORBIDDEN_PHASE7_ACCEPTED_STATUSES:
        if forbidden in matrix_text:
            fail(f"matrix contains forbidden unresolved status text: {forbidden}")

    print(
        "PHASE7_TTIR_ELEMENTWISE_MATRIX_CHECK=PASS "
        f"mode={mode} features={len(features)} accepted_hardware={len(REQUIRED_ACCEPTED_HARDWARE)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--mode", choices=["draft", "lock", "final"], default="draft")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    validate_matrix(repo_root, args.matrix, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
