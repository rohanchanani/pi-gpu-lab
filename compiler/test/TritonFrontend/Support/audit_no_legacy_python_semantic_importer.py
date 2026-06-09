#!/usr/bin/env python3
"""Audit that accepted paths do not use retired Python TTIR semantic lowering."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ALLOWED_LOWER_ELEMENTWISE = {
    "tools/vc4-triton-import",
    "compiler/test/TritonFrontend/importer-lower-elementwise-v1-retired.test",
    "compiler/test/TritonFrontend/Support/audit_no_legacy_python_semantic_importer.py",
    "compiler/test/CodeGen/Triton/Support/audit_accepted_ttir_cpp_pipeline.py",
    "compiler/test/Conversion/TritonToVC4Value/Support/audit_conversion_suite.py",
    "compiler/test/TritonFrontend/Support/check_phase7_ttir_elementwise_matrix.py",
}

ALLOWED_DOCS = {
    "AGENTS.md",
    "compiler/docs/codegen/VC4_VECTOR_TRITON_STAGED_PLAN_REPORT.md",
    "compiler/docs/vc4_triton_cpp_frontend_toolchain.md",
    "compiler/docs/vc4_triton_importer_skeleton.md",
    "compiler/docs/vc4_ttir_elementwise_v1_support_matrix.json",
    "compiler/docs/vc4_ttir_target_profile.md",
    "compiler/docs/vc4_vector_triton_phase7_5_toolchain_lane_lock.md",
    "compiler/docs/vc4_vector_triton_phase1_lock.md",
    "compiler/docs/vc4_vector_triton_phase6_frontend_lock.md",
    "compiler/docs/vc4_vector_triton_phase7_5_cpp_frontend_lock.md",
    "compiler/docs/vc4_vector_triton_phase7_ttir_elementwise_hardware_lock.md",
}

ALLOWED_INVENTORY_USES = {
    "compiler/test/TritonFrontend/importer-bad-ttir.test",
    "compiler/test/TritonFrontend/importer-inventory-saxpy.test",
    "compiler/test/TritonFrontend/importer-inventory-vector-add.test",
    "compiler/test/TritonFrontend/Support/check_phase6_frontend_lock.py",
}


def fail(message: str) -> None:
    print(f"NO_LEGACY_PYTHON_SEMANTIC_IMPORTER_AUDIT=FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def git_files(repo: Path) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        fail(proc.stderr.strip() or "git ls-files failed")
    return [repo / line for line in proc.stdout.splitlines() if line]


def main() -> int:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    repo = repo.resolve()
    violations: list[str] = []

    for path in git_files(repo):
        if not path.is_file():
            continue
        rel = path.relative_to(repo).as_posix()
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        for index, line in enumerate(lines, start=1):
            if "lower-elementwise-v1" in line or "--mode lower" in line:
                if rel in ALLOWED_LOWER_ELEMENTWISE or rel in ALLOWED_DOCS:
                    continue
                violations.append(f"{rel}:{index}: retired semantic lowering reference")

            if "vc4-triton-import" in line:
                if rel in ALLOWED_DOCS or rel in ALLOWED_INVENTORY_USES or rel in ALLOWED_LOWER_ELEMENTWISE:
                    continue
                if rel == "tools/vc4-triton-import":
                    continue
                if rel.endswith(".test") and "--mode inventory" in line:
                    continue
                violations.append(f"{rel}:{index}: vc4-triton-import outside inventory/doc context")

            if (
                rel.endswith((".test", ".sh", ".py"))
                and ".vc4value.mlir" in line
                and "vc4-triton-import" in line
            ):
                violations.append(f"{rel}:{index}: Python-generated value IR in accepted path")

    if violations:
        fail("; ".join(violations))

    print("NO_LEGACY_PYTHON_SEMANTIC_IMPORTER_AUDIT=PASS")
    print("ACCEPTED_TTIR_TO_VALUE_PATH_IS_CPP=YES")
    print("PYTHON_TTIR_GENERATION_AND_INVENTORY_ALLOWED=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
