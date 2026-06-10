#!/usr/bin/env python3
"""Audit Phase85 TTIR control-flow contract docs and taxonomy."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_CLASSIFICATIONS = {
    "TTIR_TL_RANGE_LOOP_SKELETON_POLICY": "SCF_CF",
    "TTIR_PERSISTENT_LOOP_SKELETON_POLICY": "SCF_CF",
    "TTIR_STATIC_RANGE_POLICY": "STATIC_SPECIALIZED_NO_RUNTIME_CF",
    "TTIR_SCALAR_IF_POLICY": "SCF_CF",
    "TTIR_WHILE_POLICY": "SCF_CF",
}

REQUIRED_FEATURE_IDS = {
    "ttir_control_flow_runtime_loop_skeleton",
    "ttir_control_flow_persistent_loop_skeleton",
    "ttir_control_flow_static_range_specialized",
    "ttir_control_flow_scalar_if",
    "ttir_control_flow_while",
    "ttir_control_flow_vector_branch_condition",
    "ttir_control_flow_backend_dialect",
    "ttir_control_flow_loop_body_dot",
    "ttir_control_flow_loop_body_reduce",
    "ttir_control_flow_loop_body_block_pointer",
    "ttir_control_flow_loop_attrs_num_stages_unroll_flatten",
    "ttir_control_flow_loop_attrs_warp_specialize",
}

REQUIRED_STATUSES = {
    "lowerable_phase85",
    "static_specialized_no_runtime_cf",
    "staged_by_body_feature",
    "deterministic_reject_target_profile",
    "deterministic_reject_source_boundary",
    "not_emitted_by_real_triton",
    "frontend_rejected",
    "inventory_only",
}

SNAPSHOTS = (
    "tl_range_loop_skeleton.ttir.mlir",
    "persistent_loop_skeleton.ttir.mlir",
    "static_range_probe.ttir.mlir",
    "scalar_if_probe.ttir.mlir",
    "while_probe.ttir.mlir",
)


def fail(message: str) -> None:
    print(f"PHASE85_TTIR_CF_DOC_AUDIT=FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def read(path: Path) -> str:
    if not path.exists():
        fail(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label} missing {needle!r}")


def main(argv: list[str]) -> int:
    repo = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd().resolve()

    contract_rel = "compiler/docs/vc4_ttir_control_flow_bridge.md"
    contract = read(repo / contract_rel)
    require(contract, "PHASE85_TTIR_CONTROL_FLOW_CONTRACT=ACTIVE", contract_rel)
    require(contract, "REAL_TTIR_SNAPSHOTS_SOURCE_CONTROLLED=YES", contract_rel)
    require(contract, "ANY_REAL_RUNTIME_TTIR_CF_LOWERABLE_NOW=YES", contract_rel)
    require(contract, "TTIR_VECTOR_BRANCH_CONDITION_POLICY=DETERMINISTIC_REJECT_AS_CFG_USE_MASKS", contract_rel)
    require(contract, "TTIR_UNSUPPORTED_LOOP_BODY_POLICY=STAGE_BY_BODY_FEATURE_NOT_CF", contract_rel)
    require(contract, "TTIR_BACKEND_DIALECT_CF_POLICY=DETERMINISTIC_REJECT_SOURCE_BOUNDARY", contract_rel)
    require(contract, "READY_FOR_PHASE85C_IMPORTER_REGION_FRAMEWORK=YES", contract_rel)
    require(contract, "READY_FOR_TRITON=NO", contract_rel)
    if "SUPPORT_NOW" in contract:
        fail("contract must not claim SUPPORT_NOW for TTIR control-flow import")
    for key, value in REQUIRED_CLASSIFICATIONS.items():
        require(contract, f"{key}={value}", contract_rel)

    profile = read(repo / "compiler/docs/vc4_ttir_target_profile.md")
    reject = read(repo / "compiler/docs/vc4_ttir_reject_proof_policy.md")
    staged = read(repo / "compiler/docs/codegen/VC4_VECTOR_TRITON_STAGED_PLAN_REPORT.md")
    for label, text in (
        ("compiler/docs/vc4_ttir_target_profile.md", profile),
        ("compiler/docs/vc4_ttir_reject_proof_policy.md", reject),
        ("compiler/docs/codegen/VC4_VECTOR_TRITON_STAGED_PLAN_REPORT.md", staged),
    ):
        require(text, "PHASE85_TTIR_CONTROL_FLOW_CONTRACT=ACTIVE", label)
        require(text, "READY_FOR_TRITON=NO", label)

    if "READY_FOR_PHASE9_MASK_CLASSIFIER_AND_RICHER_MEMORY_LEGALITY=YES" in contract + profile + staged:
        fail("Phase 9 must not be active before Phase 8.5 importer-region framework/final lock")

    snap_root = repo / "compiler/test/CodeGen/Triton/Snapshots/ControlFlow"
    for subdir in ("sources", "ttir", "manifests"):
        if not (snap_root / subdir).is_dir():
            fail(f"missing source-controlled snapshot directory: {snap_root / subdir}")
    for snapshot in SNAPSHOTS:
        if not (snap_root / "ttir" / snapshot).is_file():
            fail(f"missing source-controlled TTIR snapshot: {snapshot}")
    parse_test = snap_root / "control-flow-snapshots-parse.test"
    parse_text = read(parse_test)
    for snapshot in SNAPSHOTS:
        require(parse_text, snapshot, str(parse_test))

    taxonomy_path = repo / "compiler/docs/vc4_value_ttir_feature_taxonomy.json"
    taxonomy = json.loads(read(taxonomy_path))
    statuses = set(taxonomy.get("status_definitions", {}))
    missing_statuses = sorted(REQUIRED_STATUSES - statuses)
    if missing_statuses:
        fail(f"taxonomy missing statuses: {missing_statuses}")
    feature_ids = {row.get("feature_id") for row in taxonomy.get("feature_families", [])}
    missing_features = sorted(REQUIRED_FEATURE_IDS - feature_ids)
    if missing_features:
        fail(f"taxonomy missing feature ids: {missing_features}")

    if "READY_FOR_TRITON=YES" in contract + profile + reject + staged:
        fail("tracked docs must not claim READY_FOR_TRITON=YES")

    print("PHASE85_TTIR_CF_DOC_AUDIT=PASS")
    print("PHASE85_TTIR_CONTROL_FLOW_CONTRACT=ACTIVE")
    print("REAL_TTIR_SNAPSHOTS_SOURCE_CONTROLLED=YES")
    print("ANY_REAL_RUNTIME_TTIR_CF_LOWERABLE_NOW=YES")
    print("READY_FOR_PHASE85C_IMPORTER_REGION_FRAMEWORK=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
