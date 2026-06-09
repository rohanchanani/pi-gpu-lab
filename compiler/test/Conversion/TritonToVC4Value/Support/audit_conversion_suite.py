#!/usr/bin/env python3
"""Audit the dedicated TritonToVC4Value conversion lit suite."""

from __future__ import annotations

from pathlib import Path


SUITE = Path(__file__).resolve().parents[1]
SUCCESS_TESTS = (
    "ttir-vector-add-b16-to-vc4value-valid.test",
    "ttir-saxpy-select-b16-to-vc4value-valid.test",
    "ttir-i32-add-select-b16-to-vc4value-valid.test",
)

FORBIDDEN_ANYWHERE = (
    "vc4-triton-import",
    "lower-elementwise-v1",
    "%vc4_triton_python",
    "--convert-vc4kernel-to-ssavc4",
    "--convert-ssavc4-to-vc4",
    "vc4-codegen",
)

REQUIRED_SUCCESS = (
    ".ttir.mlir",
    "%vc4_triton_opt",
    "--convert-triton-to-vc4-value",
    "check_ttir_to_vc4value_output.py",
    "--vc4-verify-value-surface",
    "--convert-vc4-value-to-vc4kernel",
    "--verify-vc4kernel",
)


def fail(message: str) -> None:
    print(f"TRITON_TO_VC4VALUE_CONVERSION_SUITE_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    tests = sorted(SUITE.glob("*.test"))
    for path in tests:
      text = path.read_text(encoding="utf-8")
      rel = path.relative_to(SUITE).as_posix()
      for needle in FORBIDDEN_ANYWHERE:
        if needle in text:
          fail(f"{rel}: forbidden accepted-path marker {needle!r}")

    for name in SUCCESS_TESTS:
      path = SUITE / name
      if not path.exists():
        fail(f"missing success test {name}")
      text = path.read_text(encoding="utf-8")
      for needle in REQUIRED_SUCCESS:
        if needle not in text:
          fail(f"{name}: missing required marker {needle!r}")
      if "FileCheck" not in text or "vc4kernel.kernel" not in text:
        fail(f"{name}: missing VC4Kernel output checks")

    manifest = (SUITE / "test_manifest.json").read_text(encoding="utf-8")
    for name in SUCCESS_TESTS:
      if name not in manifest:
        fail(f"manifest missing {name}")

    print("TRITON_TO_VC4VALUE_CONVERSION_SUITE_AUDIT=PASS")
    print("CPP_IMPORTER_CONVERSION_SUITE_USES_CPP_TOOL=YES")
    print("PYTHON_SEMANTIC_LOWERING_USED=NO")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
