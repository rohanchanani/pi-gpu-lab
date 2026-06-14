#!/usr/bin/env python3
"""Audit Phase 15B value natural math static lowering boundaries."""

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
    verifier_path = repo / "compiler/lib/Transforms/ValueSurface/ValueSurfaceVerification.cpp"
    docs_paths = [
        repo / "compiler/docs/vc4_vector_triton_phase15_base2_sfu_semantics.md",
        repo / "compiler/docs/vc4_vector_triton_phase15_sfu_softmax_fixtures.md",
        repo / "compiler/docs/vc4_value_to_vc4kernel_planning.md",
    ]
    source = source_path.read_text(encoding="utf-8")
    verifier = verifier_path.read_text(encoding="utf-8")
    docs = "\n".join(path.read_text(encoding="utf-8") for path in docs_paths)

    for needle, label in [
        ("constexpr double kLog2E", "named LOG2E constant"),
        ("constexpr double kLn2", "named LN2 constant"),
        ("createF32FragmentSplatConstant", "central f32 fragment constant helper"),
        ("Public math.exp is natural exp", "natural exp comment"),
        ("exp2(x * log2(e))", "natural exp lowering comment"),
        ("Public math.log is natural log", "natural log comment"),
        ("log2(x) * ln(2)", "natural log lowering comment"),
        ("Public math.sqrt is sqrt", "sqrt comment"),
        ("x * rsqrt(x)", "sqrt lowering comment"),
        ("llvm::isa<math::ExpOp, math::LogOp, math::RsqrtOp, math::SqrtOp>", "central math dispatch"),
        ("mlir::vc4kernel::SFUKind::exp", "target exp2 mode spelling"),
        ("mlir::vc4kernel::SFUKind::log", "target log2 mode spelling"),
        ("mlir::vc4kernel::SFUKind::rsqrt", "target rsqrt mode spelling"),
        ("exact/default math requires explicit approximate-SFU policy", "exact/default reject diagnostic"),
        ("NaN/Inf exact math semantics are staged", "NaN/Inf reject diagnostic"),
    ]:
        require(source, needle, label)

    require(verifier, 'name == "math.sqrt"', "value-surface sqrt admission")

    for needle in [
        "VALUE_NATURAL_EXP_STATIC=PASS",
        "VALUE_NATURAL_LOG_STATIC=PASS",
        "VALUE_SQRT_STATIC=PASS",
        "NATURAL_EXP_LOWERING=EXP2_X_LOG2E",
        "NATURAL_LOG_LOWERING=LOG2_X_LN2",
        "SQRT_LOWERING=RSQRT_TIMES_X",
        "SOFTMAX_USES_NATURAL_EXP=YES",
        "EXACT_DEFAULT_MATH_REJECTED=YES",
        "READY_FOR_PHASE15B_2_VALUE_HARDWARE_EXP_LOG_SQRT_REPROOF=YES",
        "READY_FOR_TRITON=NO",
    ]:
        require(docs, needle, "Phase 15B.1 doc readiness line")

    for forbidden in [
        "value_sfu_natural_exp_f32_b16_vc4value",
        "value_sfu_natural_log_f32_b16_vc4value",
        "value_sfu_sqrt_f32_b16_vc4value",
        "value_softmax_stable_f32_b16_vc4value",
        "mixed_value_sfu_softmax_axes_mask_cf_f16_storage_vc4value",
        "READY_FOR_TRITON=YES",
    ]:
        forbid(source, forbidden, "fixture-name/path special case or forbidden readiness")

    print("VALUE_NATURAL_MATH_SOURCE_AUDIT=PASS")
    print("NATURAL_EXP_LOWERING=EXP2_X_LOG2E")
    print("NATURAL_LOG_LOWERING=LOG2_X_LN2")
    print("SQRT_LOWERING=RSQRT_TIMES_X")
    print("EXACT_DEFAULT_MATH_REJECTED=YES")
    print("NO_FIXTURE_NAME_OR_PATH_SPECIAL_CASES=YES")
    print("NO_HIDDEN_EXACT_MATH_CLAIMS=YES")
    print("READY_FOR_TRITON=NO")


if __name__ == "__main__":
    main()
