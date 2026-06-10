#!/usr/bin/env python3
"""Phase 11 value strided/ranked memory static lowering source audit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def fail(message: str) -> None:
    raise SystemExit(f"PHASE11_STRIDED_RANKED_SOURCE_AUDIT=FAIL: {message}")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"missing {label}: {needle}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        fail(f"forbidden {label}: {needle}")


def function_body(text: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*\{{", text)
    if not match:
        fail(f"missing function {name}")
    depth = 1
    index = match.end()
    while index < len(text) and depth:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1
    if depth != 0:
        fail(f"could not parse function body for {name}")
    return text[match.end() : index - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    source_path = (
        root
        / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"
    )
    text = strip_comments(read(source_path))

    for needle in (
        "struct MemoryAddressPlan",
        "Rank1ContiguousOrScalarComputed",
        "Rank2RowSliceIdentity",
        "Rank2RowSliceStridedOuterDynamic",
        "buildMemoryAddressPlan",
        "classifyTransferMask",
        "lowerMemRefDim",
        "lookupShapeArgForDim",
        "lookupOuterStrideArg",
    ):
        require(text, needle, "central Phase 11 memory planner component")

    read_classifier = function_body(text, "classifyTransferRead")
    write_classifier = function_body(text, "classifyTransferWrite")
    require(read_classifier, "classifyTransferMask", "read mask classifier reuse")
    require(write_classifier, "classifyTransferMask", "write mask classifier reuse")
    require(read_classifier, "buildMemoryAddressPlan", "read address planner call")
    require(write_classifier, "buildMemoryAddressPlan", "write address planner call")

    lower_read = function_body(text, "lowerTransferRead")
    lower_write = function_body(text, "lowerTransferWrite")
    forbid(lower_read, "getPermutationMap", "ad hoc read map inspection")
    forbid(lower_write, "getPermutationMap", "ad hoc write map inspection")
    require(lower_read, "legality->addressPlan", "read central plan consumption")
    require(lower_write, "legality->addressPlan", "write central plan consumption")

    planner = function_body(text, "buildMemoryAddressPlan")
    require(planner, "isRank2InnermostRowSliceTransferMap", "rank-2 row-slice map helper")
    require(planner, "lookupShapeArgForDim", "identity row stride from shape arg")
    require(planner, "lookupOuterStrideArg", "strided row stride from stride arg")
    require(planner, "column slices, transposes, and gather-like maps are", "staged map diagnostic")

    for needle in (
        "vector::GatherOp",
        "vector.gather",
        "vector::ScatterOp",
        "vector.scatter",
        "memref.extract_strided_metadata",
        "extract_strided_metadata",
        "block_pointer",
        "rank2 tile",
        "vector<16x16",
    ):
        forbid(text, needle, "out-of-scope Phase 11 static acceptance")

    for needle in (
        "fixture",
        "candidate",
        "phase11_strided",
        "value_rank2_row_slice",
    ):
        forbid(text, needle, "fixture/path/public-name semantic special case")

    print("PHASE11_STRIDED_RANKED_SOURCE_AUDIT=PASS")
    print("VALUE_STRIDED_RANKED_ADDRESS_PLANNER=YES")
    print("PHASE10_MASK_CLASSIFIER_REUSED=YES")
    print("GATHER_LANE_STRIDE_STAGED=YES")
    print("HIDDEN_MEMREF_DESCRIPTOR_REJECTED=YES")
    print("RANK2_TILE_VECTOR_ACCEPTANCE=NO")


if __name__ == "__main__":
    main()
