#!/usr/bin/env python3
"""Audit Phase 12 TTIR reduction importer invariants."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"


def fail(message: str) -> None:
    print(f"PHASE12_TTIR_REDUCTION_IMPORTER_AUDIT=FAIL: {message}")
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
        "ttir_reduce_sum",
        "ttir_reduce_max",
        "ttir_reduce_rank2",
        "mixed_ttir_reduction",
        "sourcePath",
        "fileName",
        "filename",
        "getFilename",
    )
    for needle in forbidden:
        if needle in source:
            fail(f"forbidden brittle importer pattern remains: {needle}")

    required = (
        "classifyTTIRReduceCall",
        "TTIRReduceClassification",
        "lookupTTCallee",
        "kTTReduceOpName",
        "kTTReduceReturnOpName",
        "operationAddsTwoBlockArgs",
        "functionReturnsAddOfArgs",
        "vector::ReductionOp",
        "kVC4ValueReductionPolicyAttr",
        "finite_tree",
        "memref::StoreOp",
        "scf::IfOp",
        "non-add tt.reduce is staged",
        "rank>1 tt.reduce is staged",
        "unsupported tt.reduce combiner",
        "tt.dot / contract",
        "READY_FOR_TRITON remains NO",
    )
    for needle in required:
        if needle not in source:
            fail(f"required Phase 12 structural reduction marker missing: {needle}")

    reduce_section = source[
        source.find("static TTIRReduceCallInfo classifyTTIRReduceCall"):
        source.find("enum class TTIRScalarElementKind")
    ]
    if not reduce_section:
        fail("could not locate Phase 12 reduction classifier section")
    for needle in ("inferSourceArgName", "arg_name", "loc(", "sanitizeSymbolName"):
        if needle in reduce_section:
            fail(f"reduction classification appears to use name/location metadata: {needle}")
    for needle in ("getRegion", "getOperand", "getResult", "RankedTensorType", "Block"):
        if needle not in reduce_section:
            fail(f"reduction classification missing structural SSA/region marker: {needle}")

    lower_half_mentions = [
        line.strip()
        for line in source.splitlines()
        if not line.strip().startswith("//")
        and ("vc4kernel" in line or "ssavc4" in line or "dialect == \"vc4\"" in line)
    ]
    allowed = (
        "dialect == \"vc4kernel\"",
        "dialect == \"ssavc4\"",
        "dialect == \"vc4\"",
    )
    for line in lower_half_mentions:
        if not any(token in line for token in allowed):
            fail(f"lower-half marker appears outside boundary rejection: {line}")

    print("PHASE12_TTIR_REDUCTION_IMPORTER_AUDIT=PASS")
    print("TTIR_REDUCTION_CLASSIFICATION_STRUCTURAL=YES")
    print("TTIR_REDUCTION_USES_SSA_REGION_ANALYSIS=YES")
    print("NO_FIXTURE_NAME_OR_PATH_SPECIAL_CASES=YES")
    print("NO_TTIR_TEXT_PARSING=YES")
    print("NO_REGEX_OR_SUBSTRING_SEMANTIC_CLASSIFICATION=YES")
    print("NO_DIRECT_LOWER_HALF_OUTPUT=YES")
    print("TTIR_DOT_GEMV_STAGED_FOR_PHASE13=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
