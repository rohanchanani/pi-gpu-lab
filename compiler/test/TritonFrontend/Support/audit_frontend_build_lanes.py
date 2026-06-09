#!/usr/bin/env python3
"""Audit VC4 Triton frontend build-lane isolation."""

from __future__ import annotations

import argparse
from pathlib import Path


def fail(message: str) -> None:
    print(f"FRONTEND_BUILD_LANES_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        fail(f"{path}: could not read: {exc}")


def require_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label}: missing {needle!r}")


def audit_source(repo: Path) -> None:
    top = read(repo / "compiler/CMakeLists.txt")
    require_contains(top, "option(VC4_ENABLE_TRITON_CPP_FRONTEND", "compiler/CMakeLists.txt")
    require_contains(top, "OFF)", "compiler/CMakeLists.txt")
    require_contains(top, "if(VC4_ENABLE_TRITON_CPP_FRONTEND)", "compiler/CMakeLists.txt")
    require_contains(top, "cmake/VC4TritonFrontendDeps.cmake", "compiler/CMakeLists.txt")
    require_contains(top, "add_subdirectory(tools/vc4-triton-opt)", "compiler/CMakeLists.txt")

    guarded = top.split("if(VC4_ENABLE_TRITON_CPP_FRONTEND)", 1)[1].split("endif()", 1)[0]
    for needle in (
        "cmake/VC4TritonFrontendDeps.cmake",
        "add_subdirectory(lib/Conversion/TritonToVC4Value)",
        "add_subdirectory(tools/vc4-triton-opt)",
    ):
        if needle not in guarded:
            fail(f"compiler/CMakeLists.txt: {needle!r} is not inside the optional Triton guard")

    before_guard = top.split("if(VC4_ENABLE_TRITON_CPP_FRONTEND)", 1)[0]
    for forbidden in (
        "add_subdirectory(tools/vc4-triton-opt)",
        "add_subdirectory(lib/Conversion/TritonToVC4Value)",
        "VC4TritonFrontendDeps.cmake",
    ):
        if forbidden in before_guard:
            fail(f"compiler/CMakeLists.txt: optional Triton frontend marker appears before guard: {forbidden}")

    tool_cmake = read(repo / "compiler/tools/vc4-triton-opt/CMakeLists.txt")
    require_contains(tool_cmake, "add_mlir_tool(vc4-triton-opt", "vc4-triton-opt/CMakeLists.txt")
    require_contains(tool_cmake, "VC4TritonFrontendDeps", "vc4-triton-opt/CMakeLists.txt")

    adapter = read(repo / "compiler/cmake/VC4TritonFrontendDeps.cmake")
    require_contains(adapter, "if(NOT VC4_ENABLE_TRITON_CPP_FRONTEND)", "VC4TritonFrontendDeps.cmake")
    require_contains(adapter, "message(FATAL_ERROR", "VC4TritonFrontendDeps.cmake")
    require_contains(adapter, "add_library(VC4TritonFrontendDeps INTERFACE)", "VC4TritonFrontendDeps.cmake")
    require_contains(adapter, "VC4_TRITON_CPP_OBJECTS", "VC4TritonFrontendDeps.cmake")
    require_contains(adapter, "VC4_TRITON_CPP_LIBRARIES", "VC4TritonFrontendDeps.cmake")


def audit_build_dirs(repo: Path, require_optional_build: bool) -> None:
    default_tool = repo / "compiler/build/bin/vc4-triton-opt"
    if default_tool.exists():
        fail("default compiler/build unexpectedly contains vc4-triton-opt")

    optional_tool = repo / "compiler/build-triton-llvm/bin/vc4-triton-opt"
    if require_optional_build and not optional_tool.exists():
        fail("optional compiler/build-triton-llvm vc4-triton-opt is missing")


def audit_raw_closure(repo: Path) -> None:
    adapter = repo / "compiler/cmake/VC4TritonFrontendDeps.cmake"
    for path in (repo / "compiler").rglob("CMakeLists.txt"):
        rel = path.relative_to(repo).as_posix()
        text = read(path)
        if "VC4_TRITON_CPP_OBJECTS" not in text and "VC4_TRITON_CPP_LIBRARIES" not in text:
            continue
        if rel == "compiler/CMakeLists.txt":
            continue
        fail(f"{rel}: raw Triton closure variable outside {adapter.relative_to(repo)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--require-optional-build", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    audit_source(repo)
    audit_build_dirs(repo, args.require_optional_build)
    audit_raw_closure(repo)

    print("FRONTEND_BUILD_LANES_AUDIT=PASS")
    print("DEFAULT_BUILD_TRITON_FREE=YES")
    print("OPTIONAL_CPP_FRONTEND_LANE_ISOLATED=YES")
    print("TRITON_CLOSURE_ONLY_IN_VC4TRITONFRONTENDDEPS=YES")
    print("VC4_TRITON_OPT_ONLY_IN_OPTIONAL_BUILD=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
