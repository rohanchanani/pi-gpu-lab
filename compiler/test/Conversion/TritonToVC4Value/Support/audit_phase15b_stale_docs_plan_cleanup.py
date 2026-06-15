#!/usr/bin/env python3
"""Audit Phase 15B stale-doc and active-roadmap cleanup invariants."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
DOCS = REPO_ROOT / "compiler/docs"
LADDER = DOCS / "vc4_vector_triton_ladder_plan.md"
BASE2 = DOCS / "vc4_vector_triton_phase15_base2_sfu_semantics.md"
FIXTURES = DOCS / "vc4_vector_triton_phase15_sfu_softmax_fixtures.md"
VALUE_PLAN = DOCS / "vc4_value_to_vc4kernel_planning.md"
VALUE_SURFACE = DOCS / "vc4_value_surface_spec.md"
TTIR_PROFILE = DOCS / "vc4_ttir_target_profile.md"
VC4KERNEL_SURFACE = DOCS / "vc4kernel_surface_v2_final_lock.md"
TTIR_IMPORTER = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"
VALUE_LOWERING = REPO_ROOT / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"


def fail(message: str) -> None:
    print(f"PHASE15B_STALE_DOCS_PLAN_CLEANUP_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"missing {label}: {needle}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        fail(f"stale {label}: {needle}")


def main() -> int:
    ladder = read(LADDER)
    base2 = read(BASE2)
    fixtures = read(FIXTURES)
    value_plan = read(VALUE_PLAN)
    value_surface = read(VALUE_SURFACE)
    ttir_profile = read(TTIR_PROFILE)
    vc4kernel_surface = read(VC4KERNEL_SURFACE)
    ttir_importer = read(TTIR_IMPORTER)
    value_lowering = read(VALUE_LOWERING)

    active_docs = "\n".join(
        (
            ladder,
            base2,
            fixtures,
            value_plan,
            value_surface,
            ttir_profile,
            vc4kernel_surface,
        )
    )

    required_ladder = (
        "CURRENT_LADDER_PLAN_UPDATED_AFTER_PHASE15B=YES",
        "PHASE15_BASE2_SFU_NATURAL_MATH_REPAIR_RECORDED=YES",
        "Phase 13   VALUE_AND_TTIR_GEMV_ROWWISE_DOT",
        "Phase 14   VALUE_AND_TTIR_ML_STORAGE_NUMERIC_CONVERSION_POLICY",
        "Phase 15   BASE2_SFU_BACKED_APPROX_NATURAL_MATH_AND_SOFTMAX",
        "Phase 16 = softmax-apply / attention-apply v0 over precomputed scores",
        "Phase 18 = vector.contract / tl.dot / tt.dot",
        "Phase 31 = global Triton readiness gate",
        "READY_FOR_TRITON=NO",
    )
    for needle in required_ladder:
        require(ladder, needle, "current ladder marker")

    stale_active_doc_phrases = (
        "Phase 13 math",
        "Phase 14 math smoke",
        "Phase 15 subword",
        "Phase 16 subword smoke",
        "future SFU",
    )
    for needle in stale_active_doc_phrases:
        forbid(active_docs, needle, "active phase mapping")

    required_base_semantics = (
        "TARGET_SFU_EXP_IS_EXP2=YES",
        "TARGET_SFU_LOG_IS_LOG2=YES",
        "VALUE_MATH_EXP_IS_NATURAL_EXP=YES",
        "VALUE_MATH_LOG_IS_NATURAL_LOG=YES",
        "VALUE_MATH_SQRT_IS_SQRT=YES",
        "TTIR_TL_EXP_IS_NATURAL_EXP=YES",
        "TTIR_TL_LOG_IS_NATURAL_LOG=YES",
        "TTIR_TL_SQRT_IS_SQRT=YES",
        "NATURAL_EXP_LOWERING=EXP2_X_LOG2E",
        "NATURAL_LOG_LOWERING=LOG2_X_LN2",
        "SQRT_LOWERING=RSQRT_TIMES_X",
        "SOFTMAX_USES_NATURAL_EXP=YES",
        "TTIR_SOFTMAX_USES_NATURAL_EXP=YES",
        "EXACT_DEFAULT_MATH_REJECTED=YES",
    )
    for needle in required_base_semantics:
        require(active_docs, needle, "base-2/natural math semantic marker")

    ambiguous_or_wrong_claims = (
        "math.exp is exp2",
        "math.exp equals exp2",
        "tl.exp is exp2",
        "tl.exp equals exp2",
        "math.log is log2",
        "math.log equals log2",
        "tl.log is log2",
        "tl.log equals log2",
    )
    for needle in ambiguous_or_wrong_claims:
        forbid(active_docs, needle, "public math base claim")

    for doc_name, text in (
        ("ladder", ladder),
        ("base2", base2),
        ("fixtures", fixtures),
        ("value_plan", value_plan),
        ("ttir_profile", ttir_profile),
    ):
        require(text, "READY_FOR_TRITON=NO", f"{doc_name} READY_FOR_TRITON marker")

    required_source_comments = (
        "only the VC4 standard value layer",
        "No vc4kernel/ssavc4/scheduled-vc4 operations are emitted here",
        "emits only standard\n// value-layer IR",
        "Public math.exp/log/sqrt are natural value\n//     semantics",
        "target base-2/rsqrt SFU modes",
    )
    source_text = ttir_importer + "\n" + value_lowering
    for needle in required_source_comments:
        require(source_text, needle, "updated source comment")

    stale_source_phrases = (
        "Phase 8.5 limitations",
        "Phase 7.5 elementwise subset",
        "lowering remains Phase 7.5",
        "not Phase 8.5 C++ TTIR-to-VC4Value lowerable",
        "top-level operation beside tt.func is not Phase 7.5",
    )
    for needle in stale_source_phrases:
        forbid(source_text, needle, "active source phase-limited wording")

    print("PHASE15B_STALE_DOCS_PLAN_CLEANUP_AUDIT=PASS")
    print("CURRENT_LADDER_PLAN_UPDATED_AFTER_PHASE15B=YES")
    print("ACTIVE_SFU_DOCS_BASE2_NATURAL_SEMANTICS=YES")
    print("ACTIVE_SOURCE_COMMENTS_UPDATED=YES")
    print("EXACT_DEFAULT_MATH_REJECTED_DOCS=YES")
    print("NO_DIRECT_TTIR_TO_VC4KERNEL_PATH_IN_SOURCE_COMMENTS=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
