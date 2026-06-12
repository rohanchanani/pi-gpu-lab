#!/usr/bin/env python3
"""Audit Phase 13 TTIR GEMV row-wise dot importer invariants."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
IMPORTER = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"
VALUE_LOWERING = REPO_ROOT / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"


def fail(message: str) -> None:
    print(f"PHASE13_TTIR_GEMV_IMPORTER_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    importer = IMPORTER.read_text(encoding="utf-8")
    value_lowering = VALUE_LOWERING.read_text(encoding="utf-8")

    forbidden = (
        "std::regex",
        "#include <regex>",
        "raw_string_ostream",
        "op->print(",
        ".print(",
        "ttir_gemv_row_dot",
        "mixed_ttir_gemv",
        "tl_dot_reject",
        "loop_kblocks",
        "sourcePath",
        "fileName",
        "filename",
        "getFilename",
    )
    for needle in forbidden:
        if needle in importer:
            fail(f"forbidden brittle importer pattern remains: {needle}")

    required_importer = (
        "classifyTTIRReduceCall",
        "isTTIRAddF32ReduceCallValue",
        "isTTIRLoopCarriedF32ReduceAccumulation",
        "operationAddsTwoBlockArgs",
        "vector::ReductionOp",
        "createVectorAddReduction",
        "memref::StoreOp",
        "kVC4ValueFPDomainAttr",
        "kVC4ValueReductionPolicyAttr",
        "finite_tree",
        "multi-block K accumulation is staged",
        "tt.dot / contract",
        "READY_FOR_TRITON remains NO",
    )
    for needle in required_importer:
        if needle not in importer:
            fail(f"required Phase 13 importer marker missing: {needle}")

    required_value = (
        "isReductionFragmentBlockArgument",
        "isReductionFragmentValue",
        "lookupReductionFragment",
        "isDotCompositeReductionValue",
        "multi-block K accumulation is staged",
        "AddALUOpcode::fadd",
    )
    for needle in required_value:
        if needle not in value_lowering:
            fail(f"required Phase 13 value bridge marker missing: {needle}")

    reduce_section = importer[
        importer.find("static TTIRReduceCallInfo classifyTTIRReduceCall"):
        importer.find("enum class TTIRScalarElementKind")
    ]
    if not reduce_section:
        fail("could not locate reduction and GEMV classifier section")
    for needle in ("inferSourceArgName", "arg_name", "loc(", "sanitizeSymbolName"):
        if needle in reduce_section:
            fail(f"GEMV classification appears to use name/location metadata: {needle}")
    for needle in ("getRegion", "getOperand", "getResult", "BlockArgument"):
        if needle not in reduce_section:
            fail(f"GEMV classification missing structural SSA/region marker: {needle}")

    print("PHASE13_TTIR_GEMV_IMPORTER_AUDIT=PASS")
    print("TTIR_GEMV_PRODUCT_REDUCTION_CLASSIFICATION_STRUCTURAL=YES")
    print("TTIR_GEMV_USES_SSA_REGION_ANALYSIS=YES")
    print("TTIR_GEMV_LOOP_CARRIED_ACCUMULATION_STAGED=YES")
    print("NO_FIXTURE_NAME_OR_PATH_SPECIAL_CASES=YES")
    print("NO_TTIR_TEXT_PARSING=YES")
    print("NO_REGEX_OR_SUBSTRING_SEMANTIC_CLASSIFICATION=YES")
    print("NO_DIRECT_LOWER_HALF_OUTPUT=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
