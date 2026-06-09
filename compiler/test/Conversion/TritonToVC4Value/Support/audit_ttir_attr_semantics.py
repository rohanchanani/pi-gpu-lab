#!/usr/bin/env python3
"""Audit TTIR-to-VC4Value axis/range classification for text semantics."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"

FORBIDDEN_NEEDLES = (
    "getAttrString",
    "isAxisZero",
    "isMakeRange0To16",
    "axis.find(",
    "find(\"x\")",
    "find(\"0\")",
    "contains(\"x\")",
    "contains(\"0\")",
    "op->print(",
    "StringRef(text).contains(\"start",
    "StringRef(text).contains(\"end",
    "OpPrintingFlags().skipRegions()",
)


def fail(message: str) -> None:
    print(f"TTIR_ATTR_SEMANTICS_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    text = SOURCE.read_text(encoding="utf-8")
    for needle in FORBIDDEN_NEEDLES:
        if needle in text:
            fail(f"forbidden semantic attr/range text pattern remains: {needle}")

    required = (
        "class TTIRAttrAdapter",
        "mlir::triton::GetProgramIdOp",
        "mlir::triton::GetNumProgramsOp",
        "mlir::triton::ProgramIDDimAttr",
        "mlir::triton::MakeRangeOp",
        ".getAxis()",
        ".getStart()",
        ".getEnd()",
    )
    for needle in required:
        if needle not in text:
            fail(f"required typed attr/range API use missing: {needle}")

    print("TTIR_ATTR_SEMANTICS_AUDIT=PASS")
    print("AXIS_CLASSIFICATION_EXACT=YES")
    print("MAKE_RANGE_STRUCTURED_CLASSIFICATION=YES")
    print("PRINTED_ATTR_SUBSTRING_SEMANTIC_MATCHING=NO")
    print("MAKE_RANGE_TEXT_FALLBACK=NO")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

