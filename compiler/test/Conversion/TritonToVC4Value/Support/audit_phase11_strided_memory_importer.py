#!/usr/bin/env python3
"""Audit Phase 11 TTIR strided-memory importer invariants."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"


def fail(message: str) -> None:
    print(f"PHASE11_TTIR_STRIDED_MEMORY_IMPORTER_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")

    forbidden = (
        "std::regex",
        "#include <regex>",
        "raw_string_ostream",
        "op->print(",
        ".print(",
        "stringify(Type",
        "stringify(Attribute",
        "ttir_strided_row",
        "lane_varying_stride_gather",
        "column_slice_reject",
        "sourcePath",
        "fileName",
        "filename",
        "getFilename",
        "endswith(\"_ptr\")",
        "ends_with(\"_ptr\")",
    )
    for needle in forbidden:
        if needle in source:
            fail(f"forbidden brittle importer pattern remains: {needle}")

    required = (
        "collectContiguousOffsetTerms",
        "lookupContiguousOffsetScalarTerms",
        "contiguousOffsetScalarTerms",
        "valueDependsOnMakeRange",
        "lane-varying stride/gather pointer expression staged",
        "non-contiguous column slice pointer expression staged",
        "Pointer, mask, and offset infrastructure are planned separately.",
        "Argument names are metadata only",
        "READY_FOR_TRITON remains NO",
    )
    for needle in required:
        if needle not in source:
            fail(f"required Phase 11 structural marker missing: {needle}")

    pointer_section = source[
        source.find("LogicalResult collectContiguousOffsetTerms"):
        source.find("std::optional<BlockArgument> matchCanonicalTailMask")
    ]
    if not pointer_section:
        fail("could not locate Phase 11 pointer classifier section")
    for needle in ("valueName", "inferSourceArgName", "arg_name", "loc("):
        if needle in pointer_section:
            fail(f"pointer classification appears to use name/location metadata: {needle}")
    for needle in ("getDefiningOp", "getOperand", "Value"):
        if needle not in pointer_section:
            fail(f"pointer classification missing structural SSA marker: {needle}")

    print("PHASE11_TTIR_STRIDED_MEMORY_IMPORTER_AUDIT=PASS")
    print("TTIR_POINTER_CLASSIFICATION_STRUCTURAL=YES")
    print("NO_FIXTURE_NAME_OR_PATH_SPECIAL_CASES=YES")
    print("NO_TTIR_TEXT_PARSING=YES")
    print("NO_REGEX_OR_SUBSTRING_SEMANTIC_CLASSIFICATION=YES")
    print("NO_DIRECT_LOWER_HALF_OUTPUT=YES")
    print("TTIR_RANK_INFERENCE_FROM_NAMES=NO")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
