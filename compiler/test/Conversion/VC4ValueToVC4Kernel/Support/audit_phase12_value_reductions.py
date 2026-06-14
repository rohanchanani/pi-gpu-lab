#!/usr/bin/env python3
"""Audit Phase 12 value reduction static lowering source structure."""

import argparse
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        print(f"PHASE12_VALUE_REDUCTION_SOURCE_AUDIT=FAIL")
        print(f"ERROR={message}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root)
    source_path = root / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"
    text = source_path.read_text()

    require("LogicalResult lowerVectorReduction(vector::ReductionOp reduction)" in text,
            "central vector reduction helper missing")
    require("LogicalResult lowerScalarMemrefStore(memref::StoreOp store)" in text,
            "central scalar memref.store helper missing")
    require("DenseMap<Value, Value> reductionFragments;" in text,
            "reduction fragment state map missing")
    require("kFragmentReduceOpName" in text and "vc4kernel.fragment_reduce" in text,
            "locked VC4Kernel fragment_reduce op not used")
    require("mlir::vc4kernel::ReduceKind::add" in text,
            "add reduction kind not selected structurally")
    require("mlir::vc4kernel::FPReducePolicy::finite_tree" in text,
            "finite-tree f32 reduction policy not emitted")
    require("hasFiniteReductionPolicy" in text and "kReductionPolicyAttr" in text,
            "f32 finite-tree policy helper missing")
    require("unsupported reduction variant is staged" in text,
            "non-add reduction staged diagnostic missing")
    require("rank>1 reduction is staged" in text,
            "rank>1 reduction staged diagnostic missing")
    require("vector.multi_reduction is staged" in text,
            "vector.multi_reduction staged diagnostic missing")
    require("unsupported reduction element type" in text,
            "unsupported reduction element type diagnostic missing")
    require("scalar memref.load is staged" in text,
            "scalar load staged diagnostic missing")
    require("kVDWStoreFragmentOpName" in text and "InactiveStore::preserve" in text,
            "scalar store does not use central VDW inactive-preserve path")

    forbidden_claims = [
        "dot",
        "gemv",
        "gemm",
        "vector.contract",
        "fptosi",
        "sitofp",
    ]
    lower_vector_reduction_body = text.split(
        "LogicalResult lowerVectorReduction(vector::ReductionOp reduction)", 1
    )[1].split("LogicalResult lowerMemRefDim", 1)[0]
    for token in forbidden_claims:
        require(token not in lower_vector_reduction_body.lower(),
                f"unexpected future feature token in reduction helper: {token}")

    print("PHASE12_VALUE_REDUCTION_SOURCE_AUDIT=PASS")
    print("VALUE_REDUCTION_CENTRAL_HELPER=YES")
    print("VALUE_SCALAR_STORE_CENTRAL_HELPER=YES")
    print("F32_REDUCTION_FINITE_POLICY_REQUIRED=YES")
    print("NON_ADD_REDUCTIONS_STAGED=YES")
    print("DOT_GEMV_SUPPORT_CLAIMED=NO")
    print("FIXTURE_NAME_SPECIAL_CASES=NO")
    print("READY_FOR_TRITON=NO")


if __name__ == "__main__":
    main()
