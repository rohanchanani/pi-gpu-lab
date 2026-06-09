#!/usr/bin/env python3
"""Audit accepted TTIR static tests for C++ importer provenance."""

from __future__ import annotations

import sys
from pathlib import Path


ACCEPTED_TESTS = (
    "compiler/test/CodeGen/Triton/Emit/triton-vector-add-b16-static-pipeline.test",
    "compiler/test/CodeGen/Triton/Emit/triton-saxpy-select-b16-static-pipeline.test",
    "compiler/test/CodeGen/Triton/Emit/triton-i32-add-select-b16-static-pipeline.test",
)

RUNNER_TEST = "compiler/test/CodeGen/Triton/runner-shape.test"
RUNNER_SCRIPT = "compiler/test/CodeGen/Triton/Support/run_ttir_candidate_codegen_test.sh"

FORBIDDEN_ACCEPTED = (
    "vc4-triton-import",
    "--mode lower-elementwise-v1",
    "lower-elementwise-v1",
    "%vc4_triton_python",
    ".py\" \"%vc4_repo_root/tools/vc4-triton-import",
)

REQUIRED_ACCEPTED = (
    ".ttir.mlir",
    "--convert-triton-to-vc4-value",
    ".vc4value.mlir",
    "--vc4-verify-value-surface",
    "--convert-vc4-value-to-vc4kernel",
    "--verify-vc4kernel",
)

LEGACY_TESTS = (
    "compiler/test/TritonFrontend/real-vector-add-b16-to-vc4kernel-static.test",
    "compiler/test/TritonFrontend/real-i32-add-select-b16-to-value.test",
    "compiler/test/TritonFrontend/real-vector-add-b16-to-value.test",
    "compiler/test/TritonFrontend/importer-rejects-phase6-lowering-disabled-message-removed.test",
    "compiler/test/TritonFrontend/importer-phase7c-rejects-staged-forms.test",
    "compiler/test/TritonFrontend/real-saxpy-select-b16-to-value.test",
    "compiler/test/TritonFrontend/real-extended-elementwise-to-vc4kernel-static.test",
)


def fail(message: str) -> None:
    print(f"ACCEPTED_TTIR_CPP_PIPELINE_AUDIT=FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"{path}: could not read: {exc}")


def audit_accepted(repo: Path) -> None:
    for rel in ACCEPTED_TESTS:
        path = repo / rel
        text = read(path)
        for needle in FORBIDDEN_ACCEPTED:
            if needle in text:
                fail(f"{rel}: accepted TTIR static path uses forbidden {needle!r}")
        for needle in REQUIRED_ACCEPTED:
            if needle not in text:
                fail(f"{rel}: accepted TTIR static path is missing {needle!r}")
    runner = read(repo / RUNNER_TEST)
    if "run_ttir_candidate_codegen_test.sh" not in runner:
        fail(f"{RUNNER_TEST}: runner shape test does not exercise the TTIR candidate runner")
    for needle in FORBIDDEN_ACCEPTED:
        if needle in runner:
            fail(f"{RUNNER_TEST}: runner shape test uses forbidden {needle!r}")
    support = read(repo / RUNNER_SCRIPT)
    for needle in ("vc4-triton-opt", "--convert-triton-to-vc4-value",
                   "--vc4-verify-value-surface",
                   "--convert-vc4-value-to-vc4kernel", "--verify-vc4kernel"):
        if needle not in support:
            fail(f"{RUNNER_SCRIPT}: runner support path is missing {needle!r}")
    for needle in FORBIDDEN_ACCEPTED:
        if needle in support:
            fail(f"{RUNNER_SCRIPT}: runner support path uses forbidden {needle!r}")


def audit_legacy(repo: Path) -> None:
    for rel in LEGACY_TESTS:
        path = repo / rel
        text = read(path)
        if "lower-elementwise-v1" not in text:
            fail(f"{rel}: legacy test no longer contains expected audit needle")
        if "UNSUPPORTED: legacy-python-semantic-lowering" not in text:
            fail(f"{rel}: legacy Python semantic lowering test is not retired")


def main() -> int:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    repo = repo.resolve()
    audit_accepted(repo)
    audit_legacy(repo)
    print("ACCEPTED_TTIR_CPP_PIPELINE_AUDIT=PASS")
    print("ACCEPTED_TTIR_STATIC_PIPELINE_USES_CPP_IMPORTER=YES")
    print("PYTHON_SEMANTIC_LOWER_ELEMENTWISE_RETIRED_FROM_ACCEPTED_TESTS=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
