#!/usr/bin/env python3
"""Check the Phase 6 real TTIR corpus inventory.

This is an inventory checker only. It parses TTIR through Triton's parser via
tools/vc4_parse_ttir.py, then validates op-name summaries after parse succeeds.
It does not import, lower, or translate TTIR semantics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def fail(message: str) -> "None":
    print(f"check_phase6_ttir_inventory.py: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        fail(f"failed to read JSON {path}: {exc}")


def resolve_repo_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


def parse_with_triton(
    repo_root: Path,
    python_exe: Path,
    ttir_path: Path,
    target: str,
    summary_path: Path,
) -> dict[str, Any]:
    cmd = [
        str(python_exe),
        str(repo_root / "tools" / "vc4_parse_ttir.py"),
        str(ttir_path),
        "--target",
        target,
        "--summary-json",
        str(summary_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        fail(
            "TTIR parse failed for "
            f"{ttir_path} with exit {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    summary = load_json(summary_path)
    if summary.get("parse_ttir") != "PASS":
        fail(f"parse summary for {ttir_path} did not report PASS")
    return summary


def matrix_entries(matrix: dict[str, Any]) -> dict[str, Any]:
    entries = matrix.get("entries")
    if not isinstance(entries, dict):
        fail("matrix JSON must contain an object field named 'entries'")
    return entries


def required_matrix_ops(entries: dict[str, Any]) -> dict[str, list[str]]:
    required: dict[str, list[str]] = {}
    for family, entry in entries.items():
        if entry.get("status_for_phase7") != "required":
            continue
        if not entry.get("required_in_phase7_candidates", False):
            continue
        ops = entry.get("ttir_ops", [])
        if not isinstance(ops, list) or not ops:
            fail(f"required matrix entry {family} must list ttir_ops")
        required[family] = [str(op) for op in ops]
    return required


def has_any_op(inventory: dict[str, Any], ops: list[str]) -> bool:
    return any(int(inventory.get(op, 0)) > 0 for op in ops)


def check_required_kernel(
    kernel: dict[str, Any],
    inventory: dict[str, Any],
    matrix_required: dict[str, list[str]],
) -> list[str]:
    missing: list[str] = []
    for op in kernel.get("required_ops", []):
        if int(inventory.get(op, 0)) <= 0:
            missing.append(f"manifest required op {op}")
    for family, ops in matrix_required.items():
        if not has_any_op(inventory, ops):
            missing.append(f"matrix required family {family} ({', '.join(ops)})")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--triton-python", default=None)
    parser.add_argument("--target", default="cuda:80:32")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    repo_root = repo_root_from_script()
    manifest_path = resolve_repo_path(repo_root, args.manifest)
    matrix_path = resolve_repo_path(repo_root, args.matrix)
    python_exe = Path(args.triton_python) if args.triton_python else Path(sys.executable)
    if not python_exe.is_absolute():
        python_exe = repo_root / python_exe
    if not python_exe.exists():
        fail(f"requested Triton Python does not exist: {python_exe}")

    manifest = load_json(manifest_path)
    matrix = load_json(matrix_path)
    entries = matrix_entries(matrix)
    matrix_required = required_matrix_ops(entries)
    kernels = manifest.get("kernels", [])
    if not isinstance(kernels, list) or not kernels:
        fail("manifest must contain a non-empty 'kernels' list")

    parsed: dict[str, Any] = {}
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vc4_phase6_ttir_inventory_") as temp_dir:
        temp_root = Path(temp_dir)
        for kernel in kernels:
            name = str(kernel.get("name", "<unnamed>"))
            status = kernel.get("status")
            phase7_candidate = bool(kernel.get("phase7_candidate", False))
            ttir_text = kernel.get("ttir")

            if status == "source_only_generation_failed_recorded" and not phase7_candidate:
                parsed[name] = {
                    "status": status,
                    "phase7_candidate": phase7_candidate,
                    "parse": "SKIPPED_RECORDED_SOURCE_ONLY_FAILURE",
                }
                continue

            ttir_path = resolve_repo_path(repo_root, str(ttir_text))
            if phase7_candidate and status != "generated_ttir_parse_checked":
                failures.append(f"{name}: Phase 7 candidate status is {status!r}")
            if not ttir_path.is_file():
                if phase7_candidate:
                    failures.append(f"{name}: required TTIR missing: {ttir_path}")
                else:
                    parsed[name] = {
                        "status": status,
                        "phase7_candidate": phase7_candidate,
                        "parse": "SKIPPED_NO_TTIR",
                    }
                continue

            summary_path = temp_root / f"{name}.parse.json"
            summary = parse_with_triton(repo_root, python_exe, ttir_path, args.target, summary_path)
            inventory = summary.get("op_inventory", {})
            if not isinstance(inventory, dict):
                fail(f"{name}: parse summary lacks op_inventory object")

            missing = []
            if phase7_candidate:
                missing = check_required_kernel(kernel, inventory, matrix_required)
                failures.extend(f"{name}: missing {item}" for item in missing)

            parsed[name] = {
                "status": status,
                "phase7_candidate": phase7_candidate,
                "parse": "PASS",
                "entry_func": summary.get("entry_func"),
                "op_inventory": inventory,
                "missing_required": missing,
            }

    if failures:
        for failure in failures:
            print("FAIL: " + failure, file=sys.stderr)
        fail(f"inventory check failed with {len(failures)} failure(s)")

    result = {
        "phase": "VC4 Vector/Triton Phase 6d",
        "result": "PASS",
        "manifest": str(manifest_path),
        "matrix": str(matrix_path),
        "target": args.target,
        "triton_python": str(python_exe),
        "required_matrix_families": matrix_required,
        "kernels": parsed,
    }
    if args.summary_json:
        summary_json = resolve_repo_path(repo_root, args.summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("PHASE6_TTIR_INVENTORY=PASS")
    print("REQUIRED_PHASE7_CANDIDATES_PARSED=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
