#!/usr/bin/env python3
"""Audit the Phase 15.7 repair contract after the failed static bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_DOC_LINES = (
    "PHASE15_7_REPAIR_CONTRACT=LOCKED",
    "ACCEPTED_PHASE15_TTIR_SNAPSHOTS_DO_NOT_REQUIRE_SCALAR_TT_LOAD=YES",
    "ACCEPTED_PHASE15_TTIR_SNAPSHOTS_DO_NOT_REQUIRE_SCALAR_MEMREF_LOAD=YES",
    "SCALAR_TT_LOAD_STAGED=YES",
    "SCALAR_TT_LOAD_NONZERO_OTHER_STAGED=YES",
    "TTIR_EXACT_DEFAULT_NO_POLICY_REJECT_RECLASSIFIED=YES",
    "VALUE_EXACT_DEFAULT_MATH_REJECT_REMAINS_REQUIRED=YES",
    "TTIR_EXACT_VS_APPROX_REQUIRES_STRUCTURAL_TARGET_PROFILE_INPUT=YES",
    "NO_NAME_PATH_MANIFEST_SEMANTICS=YES",
    "READY_FOR_PHASE15_7R1_CONTROLLED_FIXTURE_REPAIR=YES",
    "READY_FOR_TRITON=NO",
)


def fail(message: str) -> None:
    print(f"PHASE15_7_REPAIR_CONTRACT_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"missing {label}: {needle}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root)

    contract = repo_root / "compiler/docs/vc4_vector_triton_phase15_7_repair_contract.md"
    if not contract.exists():
        fail(f"missing repair contract doc: {contract}")
    contract_text = contract.read_text(encoding="utf-8")
    for line in REQUIRED_DOC_LINES:
        require(contract_text, line, "repair contract line")

    value_lowering = (
        repo_root / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"
    ).read_text(encoding="utf-8")
    require(value_lowering, "scalar memref.load is staged", "value scalar-load staging")

    importer = (
        repo_root / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"
    ).read_text(encoding="utf-8")
    require(importer, "tt.load nonzero other value", "TTIR nonzero-other staging")

    top_half = (repo_root / "compiler/docs/vc4_top_half_robustness_lock.md").read_text(
        encoding="utf-8"
    )
    for phrase in (
        "fixture/path/name special cases",
        "raw TTIR text parsing",
        "regex/substring semantic classification",
        "direct lower-half output",
    ):
        require(top_half, phrase, "top-half robustness lock")

    phase15_report = repo_root / ".vc4_auto/vector_triton_phase15_7_ttir_importer_static/REPORT.md"
    if phase15_report.exists():
        phase15_text = phase15_report.read_text(encoding="utf-8")
        require(phase15_text, "PHASE15_7_RESULT=FAILURE", "Phase 15.7 failure report")
        require(phase15_text, "scalar `tt.load`", "Phase 15.7 scalar-load evidence")
        require(phase15_text, "structurally", "Phase 15.7 exact/default evidence")

    manifest_path = repo_root / "examples/triton/phase15_sfu_softmax/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = {entry["name"]: entry for entry in manifest.get("snapshots", [])}
    mixed = snapshots.get("mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16")
    if mixed is None:
        fail("missing mixed Phase 15 TTIR snapshot entry")
    if mixed.get("expected_classification") != "ACCEPTED_LOWERABLE":
        fail("Phase 15.7R0 expected to record pre-repair accepted mixed mismatch")

    mixed_ttir = (
        repo_root
        / "examples/triton/phase15_sfu_softmax/generated/mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16.ttir.mlir"
    ).read_text(encoding="utf-8")
    require(mixed_ttir, "tt.load", "mixed scalar-load evidence")
    require(mixed_ttir, "!tt.ptr<f32>", "mixed scalar pointer evidence")
    require(mixed_ttir, "1.000000e+00", "mixed nonzero-other evidence")

    exact = snapshots.get("ttir_exact_math_no_policy_reject_b16")
    if exact is None:
        fail("missing exact/default TTIR snapshot entry")
    if exact.get("expected_classification") != "STAGED_EXACT_DEFAULT_MATH":
        fail("Phase 15.7R0 expected to record pre-repair exact/default mismatch")

    print("PHASE15_7_REPAIR_CONTRACT_AUDIT=PASS")
    print("PHASE15_7_FAILURE_CLASSIFICATION=FIXTURE_AND_CONTRACT_MISMATCH")
    print("SCALAR_TT_LOAD_STAGED=YES")
    print("SCALAR_TT_LOAD_NONZERO_OTHER_STAGED=YES")
    print("TTIR_EXACT_DEFAULT_NO_POLICY_REJECT_RECLASSIFIED=YES")
    print("NO_NAME_PATH_MANIFEST_SEMANTICS=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
