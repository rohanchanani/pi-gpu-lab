#!/usr/bin/env python3
"""Audit Phase 13 value GEMV static lowering boundaries."""

import argparse
import pathlib
import sys


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"missing {label}: {needle}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        fail(f"forbidden {label}: {needle}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    repo = pathlib.Path(args.repo_root).resolve()
    source_path = (
        repo
        / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"
    )
    planning_doc = repo / "compiler/docs/vc4_value_to_vc4kernel_planning.md"
    fixtures_doc = (
        repo
        / "compiler/docs/vc4_vector_triton_phase13_gemv_rowwise_dot_fixtures.md"
    )
    if not source_path.exists():
        fail(f"missing source: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    docs = planning_doc.read_text(encoding="utf-8") + "\n" + fixtures_doc.read_text(
        encoding="utf-8"
    )

    require(source, "isDotCompositeProduct", "central dot-composite helper")
    require(source, "kFragmentALUMulOpName", "existing fragment mul path")
    require(source, "kFragmentReduceOpName", "existing reduction path")
    require(source, "lowerScalarMemrefStore", "existing scalar store path")
    require(source, "exact f32 dot requires unsupported exact reduction policy",
            "exact f32 dot diagnostic")
    require(source, "vector.contract is staged for Phase 15/contract",
            "vector.contract staged diagnostic")
    require(source, "tt.dot is staged", "tt.dot staged diagnostic")
    require(source, "multi-block K accumulation is staged",
            "multi-block K staged diagnostic")
    require(source, "unsupported GEMV element type",
            "unsupported GEMV element type diagnostic")
    require(source, 'vc4value.i32_mul_policy = \\"mul24_safe\\"',
            "i32 multiply policy diagnostic")
    require(source, "i32 dot is staged by current i32 multiply policy",
            "i32 dot staged policy diagnostic")

    for forbidden in [
        "vc4kernel.dot",
        "vc4kernel.gemv",
        "kDotOpName",
        "kGEMVOpName",
        "gemv-row-dot-f32-lowers",
        "value_gemv_row_dot_f32_tail_vc4value",
        "mixed_value_gemv_row_dot_axes_mask_cf_strided_reduction_vc4value",
    ]:
        forbid(source, forbidden, "target dot op or fixture-name special case")

    for line in [
        "VALUE_GEMV_ROWWISE_DOT_STATIC=PASS",
        "VALUE_GEMV_F32_ROW_DOT_STATIC=PASS",
        "VALUE_GEMV_I32_ROW_DOT_STATUS=STAGED_BY_I32_POLICY",
        "VALUE_GEMV_PARTIAL_KBLOCK_STATIC=PASS",
        "F32_DOT_FINITE_TREE_POLICY=YES",
        "TL_DOT_TT_DOT_STAGED=YES",
        "VECTOR_CONTRACT_STAGED=YES",
        "MULTIBLOCK_K_ACCUMULATION_STAGED=YES",
        "READY_FOR_PHASE13_5_VALUE_HARDWARE_ISOLATION=YES",
        "READY_FOR_TRITON=NO",
    ]:
        require(docs, line, "Phase 13.4 doc readiness line")

    print("PHASE13_VALUE_GEMV_SOURCE_AUDIT=PASS")
    print("NO_NEW_TARGET_DOT_OP=YES")
    print("F32_DOT_USES_FINITE_TREE_POLICY=YES")
    print("VECTOR_CONTRACT_STAGED=YES")
    print("TT_DOT_STAGED=YES")
    print("MULTIBLOCK_K_ACCUMULATION_STAGED=YES")
    print("NO_FIXTURE_NAME_SPECIAL_CASES=YES")
    print("NO_HIDDEN_EXACT_F32_DOT_CLAIM=YES")
    print("F32_DOT_COMPOSITE_EXISTING_MUL_REDUCTION_STORE=YES")
    print("READY_FOR_TRITON=NO")


if __name__ == "__main__":
    main()
