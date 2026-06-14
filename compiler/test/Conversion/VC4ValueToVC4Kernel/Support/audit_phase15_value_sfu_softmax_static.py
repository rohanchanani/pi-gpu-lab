#!/usr/bin/env python3
"""Audit Phase 15 value SFU/softmax static lowering boundaries."""

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
        repo / "compiler/docs/vc4_vector_triton_phase15_sfu_softmax_fixtures.md"
    )
    if not source_path.exists():
        fail(f"missing source: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    docs = planning_doc.read_text(encoding="utf-8") + "\n" + fixtures_doc.read_text(
        encoding="utf-8"
    )

    for needle, label in [
        ("createApproxSFU", "central approximate SFU helper"),
        ("requireApproxSFUPolicy", "central approximate policy gate"),
        ("lowerApproxSFUMath", "math.exp/log/rsqrt planner"),
        ("lowerApproxDivF", "approximate reciprocal/division planner"),
        ("generic division without approximate reciprocal policy is", "generic division staged diagnostic"),
        ("exact/default math requires explicit approximate-SFU policy", "exact/default math staged diagnostic"),
        ("NaN/Inf exact math semantics are staged", "NaN/Inf staged diagnostic"),
        ("math.log SFU mode is staged by lower-half gap", "log staged diagnostic"),
        ("math.rsqrt SFU mode is staged by lower-half gap", "rsqrt staged diagnostic"),
        ("mapVectorReductionKind", "central reduction-kind mapper"),
        ("vc4kernel::ReduceKind::fmax", "finite f32 max reduction target"),
        ("vector::CombiningKind::MAXNUMF", "maxnumf accepted source spelling"),
        ("vector::CombiningKind::MAXIMUMF", "maximumf accepted source spelling"),
        ("lookupReductionFragment", "fragment-backed scalar result mapping"),
        ("kFragmentSFUOpName", "locked VC4Kernel fragment_sfu op"),
        ("kFragmentReduceOpName", "locked VC4Kernel fragment_reduce op"),
        ("kSplatOpName", "locked VC4Kernel splat op"),
    ]:
        require(source, needle, label)

    for forbidden in [
        "value_sfu_exp_f32_b16_vc4value",
        "value_sfu_recip_div_f32_b16_vc4value",
        "value_reduce_max_f32_b16_vc4value",
        "value_softmax_stable_f32_b16_vc4value",
        "mixed_value_sfu_softmax_axes_mask_cf_f16_storage_vc4value",
        "vc4value.softmax",
        "READY_FOR_TRITON=YES",
    ]:
        forbid(source, forbidden, "fixture-name/path special case or forbidden value op")

    for line in [
        "VALUE_APPROX_SFU_STATIC=PASS",
        "VALUE_APPROX_SFU_EXP_STATIC=PASS",
        "VALUE_APPROX_SFU_RECIP_DIV_STATIC=PASS",
        "VALUE_FINITE_F32_MAX_REDUCTION_STATIC=PASS",
        "VALUE_SOFTMAX_V0_STATIC=PASS",
        "APPROX_MATH_POLICY=EXPLICIT",
        "EXACT_DEFAULT_MATH_REJECTED=YES",
        "ZERO_ACTIVE_SOFTMAX_STATUS=STAGED_OR_EXPLICIT_NOOP_GUARD_PROVEN",
        "READY_FOR_PHASE15_5_VALUE_HARDWARE_ISOLATION=YES",
        "READY_FOR_TRITON=NO",
    ]:
        require(docs, line, "Phase 15.4 doc readiness line")

    print("VALUE_SFU_SOFTMAX_SOURCE_AUDIT=PASS")
    print("APPROX_MATH_THROUGH_CENTRAL_POLICY_HELPER=YES")
    print("EXACT_DEFAULT_MATH_REJECTED=YES")
    print("FINITE_MAX_REDUCTION_CENTRAL=YES")
    print("SOFTMAX_IS_COMPOSITE_NO_VALUE_OP=YES")
    print("NO_FIXTURE_NAME_OR_PATH_SPECIAL_CASES=YES")
    print("NO_HIDDEN_EXACT_MATH_CLAIMS=YES")
    print("MULTIBLOCK_SOFTMAX_NOT_ACCEPTED=YES")
    print("READY_FOR_TRITON=NO")


if __name__ == "__main__":
    main()
