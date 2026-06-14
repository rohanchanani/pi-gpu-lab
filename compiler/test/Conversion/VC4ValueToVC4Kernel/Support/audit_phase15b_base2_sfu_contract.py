#!/usr/bin/env python3
"""Audit Phase 15B base-2 SFU and natural public math contract docs."""

import argparse
import sys
from pathlib import Path


REQUIRED_CONTRACT_LINES = [
    "PHASE15_BASE2_SFU_SEMANTICS_CONTRACT=LOCKED",
    "TARGET_SFU_EXP_IS_EXP2=YES",
    "TARGET_SFU_LOG_IS_LOG2=YES",
    "TARGET_SFU_RSQRT_IS_RECIPROCAL_SQRT=YES",
    "TARGET_SFU_RECIP_IS_RECIPROCAL=YES",
    "VALUE_MATH_EXP_IS_NATURAL_EXP=YES",
    "VALUE_MATH_LOG_IS_NATURAL_LOG=YES",
    "VALUE_MATH_SQRT_IS_SQRT=YES",
    "TTIR_TL_EXP_IS_NATURAL_EXP=YES",
    "TTIR_TL_LOG_IS_NATURAL_LOG=YES",
    "TTIR_TL_SQRT_IS_SQRT=YES",
    "NATURAL_EXP_LOWERING=EXP2_X_LOG2E",
    "NATURAL_LOG_LOWERING=LOG2_X_LN2",
    "SQRT_LOWERING=RSQRT_TIMES_X_POSITIVE_FINITE_DOMAIN",
    "LOG2E_CONSTANT_REQUIRED=YES",
    "LN2_CONSTANT_REQUIRED=YES",
    "APPROX_MATH_POLICY=EXPLICIT",
    "EXACT_DEFAULT_MATH_REJECTED=YES",
    "PHASE15_5_EXP2_ORACLE_IF_PRESENT_REQUIRES_REPAIR=YES",
    "READY_FOR_PHASE15B_1_VALUE_NATURAL_MATH_STATIC_REPAIR=YES",
    "READY_FOR_TRITON=NO",
]

FORBIDDEN_DOC_MARKERS = [
    "public math.exp equals target exp2",
    "public `math.exp` equals target exp2",
    "value math.exp equals target exp2",
    "`math.exp` equals target exp2",
    "public tl.exp equals target exp2",
    "public `tl.exp` equals target exp2",
    "`tl.exp` equals target exp2",
    "public math.log equals target log2",
    "public `math.log` equals target log2",
    "value math.log equals target log2",
    "`math.log` equals target log2",
    "public tl.log equals target log2",
    "public `tl.log` equals target log2",
    "`tl.log` equals target log2",
]


def fail(message: str) -> None:
    print(f"FAIL Phase 15B base-2 SFU contract audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_marker(text: str, marker: str, context: str) -> None:
    if marker not in text:
        fail(f"{context} missing marker {marker!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root)

    contract_path = repo_root / "compiler/docs/vc4_vector_triton_phase15_base2_sfu_semantics.md"
    if not contract_path.exists():
        fail(f"missing contract doc {contract_path}")
    contract_text = contract_path.read_text()
    for line in REQUIRED_CONTRACT_LINES:
        require_marker(contract_text, line, contract_path.name)

    required_phrases = [
        "target SFU `exp` means exp2",
        "target SFU `log` means log2",
        "value `math.exp` and TTIR `tl.exp` are natural exp",
        "value `math.log` and TTIR `tl.log` are natural log",
        "value `math.sqrt` and TTIR `tl.sqrt` are sqrt",
        "natural exp(x) = target_exp2(x * log2(e))",
        "natural log(x) = target_log2(x) * ln(2)",
        "sqrt(x)        = x * target_rsqrt(x), for positive finite x",
    ]
    for phrase in required_phrases:
        require_marker(contract_text, phrase, contract_path.name)

    doc_paths = [
        contract_path,
        repo_root / "compiler/docs/vc4_vector_triton_phase15_sfu_softmax_fixtures.md",
        repo_root / "compiler/docs/vc4_value_to_vc4kernel_planning.md",
        repo_root / "compiler/docs/vc4_ttir_target_profile.md",
        repo_root / "compiler/docs/vc4kernel_surface_v2_final_lock.md",
    ]
    combined = "\n".join(path.read_text() for path in doc_paths if path.exists())
    for marker in FORBIDDEN_DOC_MARKERS:
        if marker in combined:
            fail(f"forbidden public/target semantic equivalence marker {marker!r}")

    for marker in (
        "NATURAL_EXP_LOWERING=EXP2_X_LOG2E",
        "NATURAL_LOG_LOWERING=LOG2_X_LN2",
        "SQRT_LOWERING=RSQRT_TIMES_X_POSITIVE_FINITE_DOMAIN",
        "PHASE15_5_EXP2_ORACLE_IF_PRESENT_REQUIRES_REPAIR=YES",
    ):
        require_marker(combined, marker, "Phase 15 docs")

    for static_test, diagnostic in {
        "invalid-exact-math-no-approx-policy.mlir": "exact/default math requires explicit approximate-SFU policy",
        "invalid-generic-divf-no-approx-policy.mlir": "generic division without approximate reciprocal policy is staged",
    }.items():
        path = repo_root / "compiler/test/Conversion/VC4ValueToVC4Kernel" / static_test
        if not path.exists():
            fail(f"missing static reject test {static_test}")
        require_marker(path.read_text(), diagnostic, static_test)

    exp_harness = repo_root / (
        "compiler/test/CodeGen/VC4Value/Hardware/Run/"
        "value_sfu_exp_f32_b16_vc4value/candidate/"
        "value_sfu_exp_f32_b16_vc4value_candidate_harness.c"
    )
    softmax_harness = repo_root / (
        "compiler/test/CodeGen/VC4Value/Hardware/Run/"
        "value_softmax_stable_f32_b16_vc4value/candidate/"
        "value_softmax_stable_f32_b16_vc4value_candidate_harness.c"
    )
    for path in (exp_harness, softmax_harness):
        if not path.exists():
            fail(f"missing Phase 15.5 harness {path}")
        require_marker(path.read_text(), "exp2_integer_ref", path.name)
    require_marker(contract_text, "Phase 15.5 hardware fixtures remain valid proofs of the current target SFU path", contract_path.name)

    print("PASS Phase 15B base-2 SFU contract audit")
    print("PHASE15_BASE2_SFU_SEMANTICS_CONTRACT_AUDIT=PASS")


if __name__ == "__main__":
    main()
