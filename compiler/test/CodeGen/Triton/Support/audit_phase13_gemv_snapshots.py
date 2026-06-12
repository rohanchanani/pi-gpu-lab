#!/usr/bin/env python3
"""Audit Phase 13 controlled GEMV row-dot snapshot manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: audit_phase13_gemv_snapshots.py REPO_ROOT")
    repo = Path(sys.argv[1])
    manifest = repo / "examples/triton/phase13_gemv_rowwise_dot/manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))

    if data.get("READY_FOR_TRITON") != "NO":
        raise SystemExit("READY_FOR_TRITON must remain NO")
    snapshots = {snap["name"]: snap for snap in data.get("snapshots", [])}
    accepted = [
        "ttir_gemv_row_dot_f32_b16",
        "ttir_gemv_row_dot_tail_f32_b16",
        "ttir_gemv_partial_kblock_f32_b16",
        "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16",
        "ttir_gemv_row_dot_i32_b16",
    ]
    for name in accepted:
        snap = snapshots.get(name)
        if not snap:
            raise SystemExit(f"missing snapshot {name}")
        if snap.get("expected_classification") != "ACCEPTED_LOWERABLE":
            raise SystemExit(f"{name} must be ACCEPTED_LOWERABLE")
        ttir_path = repo / snap["generated_ttir"]
        source_path = repo / snap["source"]
        if not source_path.is_file():
            raise SystemExit(f"missing source {source_path}")
        if not ttir_path.is_file():
            raise SystemExit(f"missing TTIR {ttir_path}")

    staged = {
        "ttir_tl_dot_reject_b16": "STAGED_TL_DOT_TT_DOT",
        "ttir_gemv_loop_kblocks_reject_b16": "STAGED_MULTIBLOCK_K_ACCUMULATION",
    }
    for name, classification in staged.items():
        snap = snapshots.get(name)
        if not snap:
            raise SystemExit(f"missing staged snapshot {name}")
        if snap.get("expected_classification") != classification:
            raise SystemExit(f"{name} expected {classification}")

    staged_without_snapshot = data.get("staged_without_snapshot", [])
    if not any(
        item.get("expected_classification") == "STAGED_VECTOR_CONTRACT"
        for item in staged_without_snapshot
    ):
        raise SystemExit("missing STAGED_VECTOR_CONTRACT marker")

    print("PHASE13_GEMV_MANIFEST_AUDIT=PASS")
    print("PHASE13_ACCEPTED_SNAPSHOTS=5")
    print("PHASE13_STAGED_SNAPSHOTS=2")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
