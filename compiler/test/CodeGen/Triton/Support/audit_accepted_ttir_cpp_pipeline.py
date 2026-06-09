#!/usr/bin/env python3
"""Audit accepted TTIR static tests and C++ importer boundary rules."""

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
IMPORTER_SOURCES = (
    "compiler/include/vc4/Conversion/TritonToVC4Value/TritonToVC4Value.h",
    "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp",
    "compiler/tools/vc4-triton-opt/vc4-triton-opt.cpp",
)
TRITON_ADAPTER = "compiler/cmake/VC4TritonFrontendDeps.cmake"

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

KERNEL_NAME_NEEDLES = (
    "saxpy_select_b16_kernel",
    "vector_add_b16_kernel",
    "i32_add_select_b16_kernel",
)

LOWER_HALF_CREATION_NEEDLES = (
    '"vc4kernel.',
    '"ssavc4.',
    '"vc4.',
    "OperationState state(loc, \"vc4kernel.",
    "OperationState state(loc, \"ssavc4.",
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
    legacy_retirement_test = (
        repo / "compiler/test/TritonFrontend/importer-lower-elementwise-v1-retired.test"
    )
    text = read(legacy_retirement_test)
    if "Python semantic TTIR-to-VC4Value lowering is retired" not in text:
        fail("legacy lowering retirement test is missing the retirement diagnostic")
    if "--mode lower-elementwise-v1" not in text:
        fail("legacy lowering retirement test no longer exercises the retired mode")


def audit_importer_sources(repo: Path) -> None:
    for rel in IMPORTER_SOURCES:
        text = read(repo / rel)
        for needle in KERNEL_NAME_NEEDLES:
            if needle in text:
                fail(f"{rel}: C++ importer/tool source contains kernel-name dispatch needle {needle!r}")
        for needle in ("std::regex", "regex"):
            if needle in text:
                fail(f"{rel}: C++ importer/tool source contains regex semantic importer needle {needle!r}")
        for needle in LOWER_HALF_CREATION_NEEDLES:
            if needle in text:
                fail(f"{rel}: C++ importer/tool source appears to create lower-half op {needle!r}")


def audit_triton_closure_adapter(repo: Path) -> None:
    for cmake in (repo / "compiler").rglob("CMakeLists.txt"):
        text = read(cmake)
        rel = cmake.relative_to(repo).as_posix()
        if "VC4_TRITON_CPP_OBJECTS" in text or "VC4_TRITON_CPP_LIBRARIES" in text:
            # The top-level file may declare cache variables, but raw closure
            # consumption must stay in VC4TritonFrontendDeps.cmake.
            if rel != "compiler/CMakeLists.txt":
                fail(f"{rel}: raw Triton closure variable outside adapter")
            for forbidden in ("target_link_libraries", "add_library", "add_executable"):
                if forbidden in text and ("VC4_TRITON_CPP_OBJECTS" in text or
                                          "VC4_TRITON_CPP_LIBRARIES" in text):
                    fail(f"{rel}: raw Triton closure variable used in build target outside adapter")
    adapter = read(repo / TRITON_ADAPTER)
    for needle in ("VC4_TRITON_CPP_OBJECTS", "VC4_TRITON_CPP_LIBRARIES",
                   "add_library(VC4TritonFrontendDeps INTERFACE)"):
        if needle not in adapter:
            fail(f"{TRITON_ADAPTER}: missing adapter closure marker {needle!r}")


def main() -> int:
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    repo = repo.resolve()
    audit_accepted(repo)
    audit_legacy(repo)
    audit_importer_sources(repo)
    audit_triton_closure_adapter(repo)
    print("ACCEPTED_TTIR_CPP_PIPELINE_AUDIT=PASS")
    print("ACCEPTED_TTIR_STATIC_PIPELINE_USES_CPP_IMPORTER=YES")
    print("PYTHON_SEMANTIC_LOWER_ELEMENTWISE_RETIRED_FROM_ACCEPTED_TESTS=YES")
    print("NO_KERNEL_NAME_DISPATCH=YES")
    print("NO_REGEX_SEMANTIC_IMPORTER=YES")
    print("NO_DIRECT_TTIR_TO_VC4KERNEL=YES")
    print("TRITON_CLOSURE_ONLY_IN_VC4TRITONFRONTENDDEPS=YES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
