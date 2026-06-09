#!/usr/bin/env python3
"""Audit TTIR-to-VC4Value semantic string matching.

This checks implementation source and accepted conversion tests. It deliberately
does not scan docs-only text, and it does not treat exact MLIR op/attr names or
source-location provenance names as semantic string matching.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"
TEST_ROOT = REPO_ROOT / "compiler/test/Conversion/TritonToVC4Value"

SOURCE_FORBIDDEN = (
    "stringify(Type",
    "stringify(Attribute",
    "containsTritonPointerType",
    "parseSupportedTritonPointerElement",
    "getSupportedPointerElement",
    "getTensorPointerElement",
    "\"!tt.ptr<\"",
    "StringRef(text).contains(\"!tt.ptr",
    "find(\"!tt.ptr",
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
    "raw_string_ostream",
    "isLikelySizeArgName",
    "isSizeLike",
    "ends_with(\"_n\")",
    "ends_with(\"_size\")",
    "ends_with(\"_elements\")",
)

TEST_FORBIDDEN = (
    "lower-elementwise-v1",
    "vc4-triton-import",
    "%vc4_triton_python",
)

REQUIRED_SOURCE = (
    "class TTIRTypeAdapter",
    "class TTIRAttrAdapter",
    "enum class TTIRArgumentRole",
    "TTIRArgumentRole::ScalarTailBound",
    "TTIRArgumentRole::ScalarI32",
    "TTIRArgumentRole::ScalarF32",
    "mlir::triton::PointerType",
    "mlir::triton::GetProgramIdOp",
    "mlir::triton::MakeRangeOp",
    "collectNameLocStrings",
    "inferSourceArgName",
    "kVC4ValueArgNameAttr",
)


def fail(message: str) -> None:
    print(f"TTIR_SEMANTIC_STRING_MATCHING_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")
    for needle in SOURCE_FORBIDDEN:
        if needle in source:
            fail(f"forbidden semantic string pattern remains in importer: {needle}")

    for needle in REQUIRED_SOURCE:
        if needle not in source:
            fail(f"required structural/provenance marker missing: {needle}")

    suspicious_role_patterns = (
        "role = inferSourceArgName",
        "role = arg.valueName",
        "role = info.valueName",
        "ScalarTailBound" + "Name",
        "ShapeArgs" + "NameHeuristic",
    )
    for needle in suspicious_role_patterns:
        if needle in source:
            fail(f"source-name semantic role assignment remains: {needle}")

    if "vc4kernel." in source and "forbidden dialect" not in source:
        fail("vc4kernel marker appears outside the defensive boundary check")

    for path in TEST_ROOT.rglob("*.test"):
        text = path.read_text(encoding="utf-8")
        for needle in TEST_FORBIDDEN:
            if needle in text:
                fail(f"forbidden accepted conversion path in {path.relative_to(REPO_ROOT)}: {needle}")

    print("TTIR_SEMANTIC_STRING_MATCHING_AUDIT=PASS")
    print("ARG_NAME_SEMANTIC_HEURISTIC=NO")
    print("STRUCTURAL_ARGUMENT_ROLE_CLASSIFICATION=YES")
    print("PRINTED_TYPE_SEMANTIC_MATCHING=NO")
    print("PRINTED_ATTR_SUBSTRING_SEMANTIC_MATCHING=NO")
    print("MAKE_RANGE_TEXT_FALLBACK=NO")
    print("DIRECT_LOWER_HALF_EMISSION=NO")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
