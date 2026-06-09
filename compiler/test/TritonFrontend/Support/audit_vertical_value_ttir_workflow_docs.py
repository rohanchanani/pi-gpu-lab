#!/usr/bin/env python3
"""Audit the persistent vertical value/TTIR workflow documentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"VERTICAL_VALUE_TTIR_WORKFLOW_DOCS_AUDIT=FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def read(repo: Path, rel: str) -> str:
    path = repo / rel
    if not path.exists():
        fail(f"missing required file: {rel}")
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, rel: str) -> None:
    if needle not in text:
        fail(f"{rel} missing {needle!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()

    agents = read(repo, "AGENTS.md")
    staged = read(repo, "compiler/docs/codegen/VC4_VECTOR_TRITON_STAGED_PLAN_REPORT.md")
    phase8 = read(repo, "compiler/docs/vc4_vector_triton_phase8_control_flow_lock.md")
    ttir_profile = read(repo, "compiler/docs/vc4_ttir_target_profile.md")

    require(agents, "Vertical value/TTIR increment workflow", "AGENTS.md")
    require(agents, "Phase 8.5 TTIR control-flow bridge", "AGENTS.md")
    require(agents, "Phase 9.5 TTIR", "AGENTS.md")
    require(agents, "READY_FOR_TRITON` remains `NO", "AGENTS.md")

    require(
        staged,
        "Current Phase-8.25 Vertical Workflow Note",
        "compiler/docs/codegen/VC4_VECTOR_TRITON_STAGED_PLAN_REPORT.md",
    )
    require(
        staged,
        "corresponding TTIR bridge/support/reject lock",
        "compiler/docs/codegen/VC4_VECTOR_TRITON_STAGED_PLAN_REPORT.md",
    )
    require(
        staged,
        "Phase 8.5 — TTIR control-flow bridge/support/reject lock",
        "compiler/docs/codegen/VC4_VECTOR_TRITON_STAGED_PLAN_REPORT.md",
    )
    require(
        staged,
        "Phase 9.5 — TTIR mask/memory-legality bridge/support/reject lock",
        "compiler/docs/codegen/VC4_VECTOR_TRITON_STAGED_PLAN_REPORT.md",
    )
    require(
        staged,
        "Phase 10.5 — TTIR gather/strided memory bridge",
        "compiler/docs/codegen/VC4_VECTOR_TRITON_STAGED_PLAN_REPORT.md",
    )

    require(phase8, "ACTIVE_NEXT_STEP=PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE", "compiler/docs/vc4_vector_triton_phase8_control_flow_lock.md")
    require(phase8, "READY_FOR_PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE=YES", "compiler/docs/vc4_vector_triton_phase8_control_flow_lock.md")
    require(
        phase8,
        "READY_FOR_PHASE9_MASK_CLASSIFIER_AND_RICHER_MEMORY_LEGALITY=NO_PENDING_PHASE8_5_TTIR_CONTROL_FLOW_BRIDGE",
        "compiler/docs/vc4_vector_triton_phase8_control_flow_lock.md",
    )
    require(phase8, "READY_FOR_TRITON=NO", "compiler/docs/vc4_vector_triton_phase8_control_flow_lock.md")

    phase9_rel = "compiler/docs/vc4_vector_triton_phase9_mask_memory_legality_lock.md"
    phase9_path = repo / phase9_rel
    phase9_present = phase9_path.exists()
    if phase9_present:
        phase9 = phase9_path.read_text(encoding="utf-8")
        require(phase9, "READY_FOR_PHASE9_5_TTIR_MASK_MEMORY_LEGALITY=YES", phase9_rel)
        require(
            phase9,
            "READY_FOR_PHASE10_MEMORY_EXPANSION_GATHER_STRIDED=NO_PENDING_PHASE9_5_TTIR_MASK_MEMORY_LEGALITY",
            phase9_rel,
        )
        phase9_status = "YES"
    else:
        phase9_status = "NOT_PRESENT"

    require(ttir_profile, "## 9.5 TTIR control-flow bridge status", "compiler/docs/vc4_ttir_target_profile.md")
    require(ttir_profile, "frontend_specialization", "compiler/docs/vc4_ttir_target_profile.md")
    require(ttir_profile, "lowerable_if_matches_value_cf_subset", "compiler/docs/vc4_ttir_target_profile.md")
    require(ttir_profile, "READY_FOR_TRITON=NO", "compiler/docs/vc4_ttir_target_profile.md")

    tracked_docs = [agents, staged, phase8, ttir_profile]
    if any("READY_FOR_TRITON=YES" in text for text in tracked_docs):
        fail("tracked vertical workflow docs must not claim READY_FOR_TRITON=YES")

    print("VERTICAL_VALUE_TTIR_WORKFLOW_DOCS_AUDIT=PASS")
    print("PHASE8_ACTIVE_NEXT_STEP_IS_PHASE8_5=YES")
    print(f"PHASE9_ACTIVE_NEXT_STEP_IS_PHASE9_5={phase9_status}")
    print("STAGED_PLAN_MENTIONS_VALUE_LED_VERTICAL_INCREMENTS=YES")
    print("AGENTS_MENTIONS_TTIR_BRIDGE_PHASES=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
