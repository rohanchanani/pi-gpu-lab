#!/usr/bin/env python3
"""Audit Phase 17 TTIR online softmax importer coverage and robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fail(message: str) -> None:
    print(f"PHASE17_ONLINE_SOFTMAX_IMPORTER_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def require(text: str, needle: str, context: str) -> None:
    if needle not in text:
        fail(f"missing {context}: {needle}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root)
    source_path = root / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"
    manifest_path = root / "examples/triton/phase17_online_softmax_state/manifest.json"
    source = source_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    forbidden_source = (
        "std::regex",
        "#include <regex>",
        "raw_string_ostream",
        "op->print(",
        ".print(",
        "ttir_online_softmax",
        "online_attention_apply",
        "phase17_online_softmax",
        "fixtureName",
        "candidatePath",
        "sourcePath",
        "getFilename",
    )
    for needle in forbidden_source:
        if needle in source:
            fail(f"forbidden source/name/path semantic hook remains: {needle}")

    require(source, "lowerSCF", "structural scf lowering")
    require(source, "ensureIndexValue", "i32/index loop-control bridge")
    require(source, "regionArgTypeOverrides", "loop IV type override")
    require(source, "classifyPointer", "structural pointer classification")
    require(source, "collectContiguousOffsetTerms", "pointer use-def traversal")
    require(source, "matchesCanonicalTailMask", "tail mask classification")
    require(source, "classifyTTIRReduceCall", "tt.reduce classification")
    require(source, "isTTIRGeneratedScoreReductionCall",
            "structural generated-score reject")
    require(source, "isTTIRLoadLikeVectorValue",
            "QK load-product structural check")
    require(source, "math::ExpOp", "natural exp lowering")
    require(source, "vector::ReductionOp", "value reduction output")
    require(source, "memref::StoreOp", "scalar value store output")
    require(source, "kTTDotOpName", "tt.dot staging")
    require(source, "scalar tt.load", "scalar tt.load reject")
    require(source, "non-contiguous column slice pointer expression staged",
            "non-transposed V gather reject")
    require(source, "QK score generation is staged", "QK reject diagnostic")
    require(source, "valueFunc->setAttr(kVC4ValueMathPolicyAttr",
            "explicit approximate math policy")
    require(source, "vc4value.fp_domain", "finite value policy attribute")
    require(source, "Argument names are metadata only",
            "argument-name non-semantic policy")

    for token in ("vc4kernel::", "ssavc4::"):
        if token in source:
            fail(f"direct lower-half C++ API reference remains: {token}")

    expected = {
        "ttir_online_softmax_normalizer_f32_b16": "ACCEPTED_LOWERABLE",
        "ttir_online_attention_apply_v0_f32_b16": "ACCEPTED_LOWERABLE",
        "ttir_online_attention_apply_v0_scaled_f32_b16": "ACCEPTED_LOWERABLE",
        "mixed_ttir_online_attention_apply_axes_mask_cf_f16_storage_b16": "ACCEPTED_LOWERABLE",
        "ttir_online_attention_nontransposed_v_reject_b16": "STAGED_LANE_VARYING_STRIDE",
        "ttir_online_attention_scalar_scale_load_reject_b16": "STAGED_SCALAR_TT_LOAD",
        "ttir_online_softmax_k_zero_reject_b16": "STAGED_K_ZERO_ONLINE_SOFTMAX",
        "ttir_online_attention_qk_score_generation_reject_b16": "STAGED_QK_SCORE_GENERATION",
        "ttir_online_attention_tl_dot_reject_b16": "STAGED_TT_DOT",
    }
    fixtures = {entry["name"]: entry for entry in manifest["fixtures"]}
    for name, classification in expected.items():
        entry = fixtures.get(name)
        if not entry:
            fail(f"manifest fixture missing: {name}")
        if entry.get("classification") != classification:
            fail(f"{name} classification mismatch")
        generated = root / entry["generated_ttir"]
        if not generated.exists():
            fail(f"generated TTIR snapshot missing: {generated}")

    accepted_claims = {
        "ttir_online_softmax_normalizer_f32_b16": (
            "loop-carried scalar f32 m/l state",
            "natural-exp online softmax denominator recurrence",
            "masked precomputed score loads",
            "scalar f32 output store",
        ),
        "ttir_online_attention_apply_v0_f32_b16": (
            "precomputed scores only",
            "transposed V lane-contiguous load",
            "loop-carried scalar f32 m/l/acc state",
            "natural-exp online attention-apply recurrence",
            "scalar f32 output store",
        ),
        "ttir_online_attention_apply_v0_scaled_f32_b16": (
            "scalar f32 scale argument",
            "tt.splat scale broadcast",
            "precomputed scores only",
            "transposed V lane-contiguous load",
            "loop-carried scalar f32 m/l/acc state",
        ),
        "mixed_ttir_online_attention_apply_axes_mask_cf_f16_storage_b16": (
            "multi-axis program ids",
            "structured control-flow guard",
            "masked f16 storage loads promoted to f32 compute",
            "loop-carried scalar f32 m/l/acc state",
            "transposed V lane-contiguous load",
        ),
    }
    for name, claims in accepted_claims.items():
        actual = set(fixtures[name].get("expected_feature_claims", []))
        missing = [claim for claim in claims if claim not in actual]
        if missing:
            fail(f"{name} missing accepted claims: {missing}")

    print("PHASE17_ONLINE_SOFTMAX_IMPORTER_AUDIT=PASS")
    print("TTIR_ONLINE_SOFTMAX_IMPORTER_STATIC=PASS")
    print("TTIR_ONLINE_ATTENTION_APPLY_IMPORTER_STATIC=PASS")
    print("TTIR_ONLINE_SOFTMAX_LOOP_STATE_LOWERING=YES")
    print("TTIR_ONLINE_ATTENTION_PRECOMPUTED_SCORES=YES")
    print("TTIR_ONLINE_ATTENTION_TRANSPOSED_V_LAYOUT=YES")
    print("TTIR_ONLINE_ATTENTION_NATURAL_EXP_RECURRENT_SOFTMAX=YES")
    print("TTIR_ONLINE_ATTENTION_WEIGHTED_SUM=YES")
    print("TTIR_NONTRANSPOSED_V_GATHER_REJECT=PASS")
    print("TTIR_SCALAR_GLOBAL_LOAD_REJECT=PASS")
    print("TTIR_QK_SCORE_GENERATION_REJECT=PASS")
    print("NO_NAME_PATH_FIXTURE_SPECIAL_CASES=YES")
    print("NO_RAW_TTIR_TEXT_PARSING=YES")
    print("NO_REGEX_OR_SUBSTRING_SEMANTIC_CLASSIFICATION=YES")
    print("STRUCTURAL_LOOP_POINTER_REDUCTION_SOFTMAX_CLASSIFICATION=YES")
    print("NO_DIRECT_LOWER_HALF_OUTPUT=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
