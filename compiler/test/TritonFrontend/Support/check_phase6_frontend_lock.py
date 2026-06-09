#!/usr/bin/env python3
"""Static Phase 6 frontend lock checker.

This checker intentionally does not import Triton. It verifies that the
source-controlled Phase 6 lock, corpus, tools, and artifact boundaries are in
the expected pending or final state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PENDING_LINES = {
    "VC4_TRITON_PHASE6_FRONTEND_LOCKED": "PENDING",
    "VC4_TRITON_PHASE6_PINNED_TRITON_VERSION": "3.7.0",
    "VC4_TRITON_PHASE6_REAL_TTIR_CORPUS": "YES",
    "VC4_TRITON_PHASE6_TTIR_GENERATION_TOOL": "YES",
    "VC4_TRITON_PHASE6_TTIR_PARSE_INVENTORY": "YES",
    "VC4_TRITON_PHASE6_IMPORTER_SKELETON": "YES",
    "VC4_TRITON_PHASE6_NO_TTIR_TO_VALUE_SEMANTIC_LOWERING": "YES",
    "VC4_TRITON_PHASE6_NO_CORE_TRITON_DEPENDENCY": "YES",
    "READY_FOR_PHASE7_REAL_TTIR_ELEMENTWISE_SMOKE": "PENDING",
    "READY_FOR_TRITON": "NO",
}

FINAL_LINES = {
    **PENDING_LINES,
    "VC4_TRITON_PHASE6_FRONTEND_LOCKED": "YES",
    "READY_FOR_PHASE7_REAL_TTIR_ELEMENTWISE_SMOKE": "YES",
}

REQUIRED_ELEMENTWISE = [
    "vector_add_b16",
    "saxpy_select_b16",
    "i32_add_select_b16",
]

FORBIDDEN_SUFFIXES = {
    ".ttgir",
    ".ptx",
    ".cubin",
    ".hsaco",
    ".ll",
    ".llvm",
}

FORBIDDEN_NAME_PARTS = [
    ".ttgir.",
    ".ptx.",
    ".cubin.",
    ".hsaco.",
]


def fail(message: str) -> "None":
    print(f"check_phase6_frontend_lock.py: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"failed to read JSON {path}: {exc}")


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"missing required file: {path}")


def require_executable(path: Path) -> None:
    require_file(path)
    if not os.access(path, os.X_OK):
        fail(f"required tool is not executable: {path}")


def parse_readiness_lines(text: str) -> dict[str, str]:
    lines: dict[str, str] = {}
    for raw in text.splitlines():
        stripped = raw.strip()
        if "=" not in stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split("=", 1)
        if key.startswith("VC4_TRITON_PHASE6_") or key.startswith("READY_FOR_"):
            lines[key] = value
    return lines


def check_readiness(lock_doc: Path, mode: str) -> None:
    expected = PENDING_LINES if mode == "pending" else FINAL_LINES
    found = parse_readiness_lines(lock_doc.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if found.get(key) != value:
            fail(f"{lock_doc} expected {key}={value}, found {found.get(key)!r}")


def check_required_corpus(repo_root: Path) -> None:
    manifest_path = repo_root / "examples/triton/phase6/manifest.json"
    require_file(manifest_path)
    manifest = load_json(manifest_path)
    kernels = {entry.get("name"): entry for entry in manifest.get("kernels", [])}
    for name in REQUIRED_ELEMENTWISE:
        entry = kernels.get(name)
        if not isinstance(entry, dict):
            fail(f"manifest missing required elementwise kernel {name}")
        if entry.get("status") != "generated_ttir_parse_checked":
            fail(f"{name} must have generated_ttir_parse_checked status")
        if entry.get("phase7_candidate") is not True:
            fail(f"{name} must be marked phase7_candidate=true")

        ttir = repo_root / str(entry.get("ttir"))
        metadata = repo_root / str(entry.get("metadata"))
        require_file(ttir)
        require_file(metadata)
        metadata_json = load_json(metadata)
        version = str(metadata_json.get("triton_version", ""))
        if not version.startswith("3.7.0"):
            fail(f"{metadata} must record triton_version starting with 3.7.0, found {version!r}")


def check_no_forbidden_snapshots(repo_root: Path) -> None:
    corpus_root = repo_root / "examples/triton/phase6"
    for path in corpus_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"forbidden target-specific artifact checked in: {path}")
        if any(part in name for part in FORBIDDEN_NAME_PARTS):
            fail(f"forbidden target-specific artifact checked in: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--mode", choices=["pending", "final"], required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    lock_doc = repo_root / "compiler/docs/vc4_vector_triton_phase6_frontend_lock.md"
    require_file(lock_doc)
    check_readiness(lock_doc, args.mode)

    require_file(repo_root / "compiler/docs/vc4_ttir_frontend_inventory_matrix.json")
    require_file(repo_root / "compiler/docs/vc4_real_ttir_inventory.md")
    check_required_corpus(repo_root)
    check_no_forbidden_snapshots(repo_root)

    require_executable(repo_root / "tools/vc4_emit_ttir.py")
    require_executable(repo_root / "tools/vc4_parse_ttir.py")
    require_executable(repo_root / "tools/vc4-triton-import")

    print("PHASE6_FRONTEND_LOCK_CHECK=PASS")
    print(f"MODE={args.mode}")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
