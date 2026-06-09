#!/usr/bin/env python3
"""Audit TTIR-to-VC4Value pointer type classification for string semantics."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"

FORBIDDEN_NEEDLES = (
    "!tt.ptr<",
    "containsTritonPointerType",
    "parseSupportedTritonPointerElement",
    "getSupportedPointerElement",
    "getTensorPointerElement",
    "StringRef(text).contains(\"!tt.ptr",
    "find(\"!tt.ptr",
)


def fail(message: str) -> None:
    print(f"TTIR_TYPE_SEMANTICS_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    for needle in FORBIDDEN_NEEDLES:
        if needle in text:
            fail(f"forbidden semantic pointer type string pattern remains: {needle}")

    if "stringify(Type" in text:
        fail("Type printing helper remains in importer implementation")
    if "llvm::dyn_cast<mlir::triton::PointerType>" not in text:
        fail("typed Triton PointerType classification was not found")
    if ".getPointeeType()" not in text:
        fail("PointerType::getPointeeType() use was not found")
    if ".getAddressSpace()" not in text:
        fail("PointerType::getAddressSpace() use was not found")

    print("TTIR_TYPE_SEMANTICS_AUDIT=PASS")
    print("POINTER_TYPE_CLASSIFICATION_TYPED=YES")
    print("PRINTED_TYPE_SEMANTIC_MATCHING=NO")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
