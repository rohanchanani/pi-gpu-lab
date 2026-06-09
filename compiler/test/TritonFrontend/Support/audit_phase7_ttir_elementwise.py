#!/usr/bin/env python3
"""Audit Phase 7 TTIR elementwise layer boundaries and lock claims."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SNAPSHOTS = [
    "examples/triton/phase6/generated/vector_add_b16.ttir.mlir",
    "examples/triton/phase6/generated/saxpy_select_b16.ttir.mlir",
    "examples/triton/phase6/generated/i32_add_select_b16.ttir.mlir",
]

VALUE_REQUIRED = [
    "func.func",
    "vc4value.kernel",
    "vc4value.program_id",
    "vector.step",
    "vector.create_mask",
    "vector.transfer_read",
    "vector.transfer_write",
]

VALUE_FORBIDDEN = [
    "tt.",
    "ttg.",
    "gpu.",
    "nvgpu.",
    "nvvm.",
    "rocdl.",
    "llvm.",
    "vc4kernel.",
    "ssavc4.",
    "vc4.qpu.",
    "vc4.module",
]

VC4KERNEL_FORBIDDEN = [
    "tt.",
    "ttg.",
    "gpu.",
    "func.",
    "vector.",
    "memref.",
    "vc4value.",
    "ssavc4.",
    "vc4.qpu.",
    "vc4.module",
]


def fail(message: str) -> None:
    print(f"audit_phase7_ttir_elementwise.py: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        fail(
            "command failed with exit "
            f"{proc.returncode}: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def find_tool(repo_root: Path, name: str) -> str:
    candidate = repo_root / "compiler" / "build" / "bin" / name
    if candidate.exists():
        return str(candidate)
    proc = subprocess.run(["/bin/sh", "-lc", f"command -v {name}"], text=True, stdout=subprocess.PIPE)
    if proc.returncode == 0 and proc.stdout.strip():
        return proc.stdout.strip()
    fail(f"could not find required tool: {name}")


def triton_python(repo_root: Path) -> str:
    override = os.environ.get("VC4_TRITON_PYTHON")
    if override:
        return override
    candidate = repo_root / ".vc4_auto" / "triton_phase6_venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def contains_op_marker(text: str, marker: str) -> bool:
    if marker.endswith("."):
        return re.search(rf'(?<![A-Za-z0-9_])"?{re.escape(marker)}', text) is not None
    return marker in text


def require_markers(text: str, required: list[str], path: Path) -> None:
    for marker in required:
        if marker not in text:
            fail(f"{path} missing required marker {marker}")


def reject_markers(text: str, forbidden: list[str], path: Path) -> None:
    for marker in forbidden:
        if contains_op_marker(text, marker):
            fail(f"{path} contains forbidden marker {marker}")


def audit_direct_ttir_claims(repo_root: Path) -> None:
    allowed_negative_words = (
        "not ",
        "no direct",
        "must not",
        "forbidden",
        "reject",
        "rejected",
        "or direct",
        "without a direct",
        "no ttir-to-vc4kernel",
    )
    patterns = [
        re.compile(r"TTIR\s*-?>\s*vc4kernel", re.IGNORECASE),
        re.compile(r"TTIR-to-VC4Kernel", re.IGNORECASE),
        re.compile(r"TTIR to VC4Kernel", re.IGNORECASE),
    ]
    roots = [repo_root / "compiler" / "docs", repo_root / "compiler" / "test", repo_root / "tools"]
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".o", ".a"}:
                continue
            if "build" in path.parts or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(repo_root).as_posix()
            if "/Support/" in rel and ("audit" in path.name or "check" in path.name):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lines = text.splitlines()
            for lineno, line in enumerate(lines, 1):
                if any(pattern.search(line) for pattern in patterns):
                    context = " ".join(lines[max(0, lineno - 3) : min(len(lines), lineno + 2)])
                    lowered = context.lower()
                    if not any(word in lowered for word in allowed_negative_words):
                        fail(f"{path}:{lineno}: possible forbidden direct TTIR-to-lower-half claim: {line}")


def audit_ready_for_triton(repo_root: Path) -> None:
    for root in [repo_root / "compiler", repo_root / "tools", repo_root / "examples"]:
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or "build" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "READY_FOR_TRITON=YES" not in text:
                continue
            rel = path.relative_to(repo_root).as_posix()
            if "/Support/" in rel and ("audit" in path.name or "check" in path.name):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if "READY_FOR_TRITON=YES" in line:
                    lowered = line.lower()
                    if not any(token in lowered for token in ("forbidden", "negative", "audit", "needle", "must not")):
                        fail(f"{path}:{lineno}: READY_FOR_TRITON=YES is not an audit/negative needle")


def audit_generated_boundaries(repo_root: Path) -> None:
    vc4_opt = find_tool(repo_root, "vc4-opt")
    importer = repo_root / "tools" / "vc4-triton-import"
    py = triton_python(repo_root)
    with tempfile.TemporaryDirectory(prefix="phase7_ttir_audit_") as tmp_text:
        tmp = Path(tmp_text)
        for rel in SNAPSHOTS:
            snapshot = repo_root / rel
            if not snapshot.exists():
                fail(f"missing required TTIR snapshot: {rel}")
            stem = snapshot.name.replace(".ttir.mlir", "")
            value = tmp / f"{stem}.value.mlir"
            vc4kernel = tmp / f"{stem}.vc4kernel.mlir"
            run(
                [
                    py,
                    str(importer),
                    str(snapshot),
                    "--mode",
                    "lower-elementwise-v1",
                    "--target",
                    "cuda:80:32",
                    "-o",
                    str(value),
                ],
                repo_root,
            )
            value_text = value.read_text(encoding="utf-8")
            require_markers(value_text, VALUE_REQUIRED, value)
            reject_markers(value_text, VALUE_FORBIDDEN, value)
            run([vc4_opt, str(value), "--vc4-verify-value-surface"], repo_root)
            run(
                [
                    vc4_opt,
                    str(value),
                    "--vc4-verify-value-surface",
                    "--convert-vc4-value-to-vc4kernel",
                    "--verify-vc4kernel",
                    "-o",
                    str(vc4kernel),
                ],
                repo_root,
            )
            vc4kernel_text = vc4kernel.read_text(encoding="utf-8")
            if "vc4kernel.kernel" not in vc4kernel_text:
                fail(f"{vc4kernel} missing vc4kernel.kernel")
            reject_markers(vc4kernel_text, VC4KERNEL_FORBIDDEN, vc4kernel)


def audit_mixed_claims(repo_root: Path) -> None:
    manifest = repo_root / "compiler/test/CodeGen/Triton/Hardware/MixedAcceptance/mixed_acceptance_manifest.json"
    claims = repo_root / "compiler/test/CodeGen/Triton/Hardware/MixedAcceptance/mixed_fixture_claims.json"
    coverage = repo_root / "compiler/test/CodeGen/Triton/Hardware/MixedAcceptance/check_mixed_acceptance_coverage.py"
    audit = repo_root / "compiler/test/CodeGen/Triton/Hardware/MixedAcceptance/audit_mixed_fixture_claims.py"
    for path in (manifest, claims, coverage, audit):
        if not path.exists():
            fail(f"missing mixed acceptance audit path: {path.relative_to(repo_root)}")
    run([sys.executable, str(coverage), "--manifest", str(manifest), "--repo-root", str(repo_root), "--mode", "lock"], repo_root)
    run([sys.executable, str(audit), "--claims", str(claims), "--manifest", str(manifest), "--repo-root", str(repo_root)], repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--mode", choices=["draft", "lock"], default="draft")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    audit_direct_ttir_claims(repo_root)
    audit_ready_for_triton(repo_root)
    audit_generated_boundaries(repo_root)
    audit_mixed_claims(repo_root)
    print(f"PHASE7_TTIR_ELEMENTWISE_AUDIT=PASS mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
