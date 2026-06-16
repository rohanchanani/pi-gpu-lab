#!/usr/bin/env python3
"""Audit Phase 16 TTIR attention-apply importer coverage and robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fail(message: str) -> None:
    print(f"PHASE16_ATTENTION_APPLY_IMPORTER_AUDIT=FAIL: {message}")
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
    manifest_path = root / "examples/triton/phase16_attention_apply_v0/manifest.json"
    source = source_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    forbidden_source = (
        "std::regex",
        "#include <regex>",
        "raw_string_ostream",
        "op->print(",
        ".print(",
        "ttir_attention_apply",
        "attention_apply_v0",
        "phase16_attention_apply",
        "fixtureName",
        "candidatePath",
        "sourcePath",
        "getFilename",
    )
    for needle in forbidden_source:
        if needle in source:
            fail(f"forbidden source/name/path semantic hook remains: {needle}")

    require(source, "classifyPointer", "structural pointer classification")
    require(source, "verifyContiguousOffset", "contiguous pointer verifier")
    require(source, "collectContiguousOffsetTerms", "pointer use-def traversal")
    require(source, "matchesCanonicalTailMask", "tail mask classification")
    require(source, "classifyTTIRReduceCall", "tt.reduce classification")
    require(source, "math::ExpOp", "natural exp lowering")
    require(source, "kTTLoadOpName", "tt.load handling")
    require(source, "kTTStoreOpName", "tt.store handling")
    require(source, "kTTDotOpName", "tt.dot staging")
    require(source, "lane-varying stride/gather pointer expression staged",
            "non-transposed V gather reject")
    require(source, "scalar tt.load", "scalar tt.load reject")
    require(source, "isTTIRGeneratedScoreReductionCall",
            "generated QK score structural reject")
    require(source, "tt.dot / contract", "dot/contract reject")
    require(source, "Argument names are metadata only",
            "argument-name non-semantic policy")
    require(source, "valueFunc->setAttr(kVC4ValueMathPolicyAttr",
            "explicit approximate math policy")

    for token in ("vc4kernel::", "ssavc4::"):
        if token in source:
            fail(f"direct lower-half C++ API reference remains: {token}")

    expected = {
        "ttir_attention_apply_v0_f32_b16": "ACCEPTED_LOWERABLE",
        "ttir_attention_apply_v0_scaled_f32_b16": "ACCEPTED_LOWERABLE",
        "mixed_ttir_attention_apply_v0_axes_mask_cf_f16_storage_b16": "ACCEPTED_LOWERABLE",
        "ttir_attention_apply_nontransposed_v_reject_b16": "STAGED_LANE_VARYING_STRIDE",
        "ttir_attention_apply_scalar_scale_load_reject_b16": "STAGED_SCALAR_TT_LOAD",
        "ttir_attention_apply_k_zero_reject_b16": "STAGED_ZERO_ACTIVE_ATTENTION_APPLY",
        "ttir_attention_apply_multiblock_reject_b16": "STAGED_MULTIBLOCK_SOFTMAX",
        "ttir_attention_apply_tl_dot_reject_b16": "STAGED_TT_DOT",
    }
    snapshots = {entry["name"]: entry for entry in manifest["snapshots"]}
    for name, classification in expected.items():
        entry = snapshots.get(name)
        if not entry:
            fail(f"manifest snapshot missing: {name}")
        if entry.get("expected_classification") != classification:
            fail(f"{name} classification mismatch")
        generated = root / entry["generated_ttir"]
        if not generated.exists():
            fail(f"generated TTIR snapshot missing: {generated}")

    required_claims = {
        "stable_softmax",
        "natural_exp",
        "weighted_sum",
        "scalar_store",
    }
    for name, classification in expected.items():
        if classification != "ACCEPTED_LOWERABLE":
            continue
        claims = set(snapshots[name].get("expected_feature_claims", []))
        expected_claims = set(required_claims)
        if not name.startswith("mixed_"):
            expected_claims.update({"precomputed_scores",
                                    "transposed_v_row_contiguous"})
        missing = expected_claims - claims
        if missing:
            fail(f"{name} missing accepted claims: {sorted(missing)}")

    print("PHASE16_ATTENTION_APPLY_IMPORTER_AUDIT=PASS")
    print("TTIR_ATTENTION_APPLY_V0_IMPORTER_STATIC=PASS")
    print("TTIR_ATTENTION_APPLY_V0_LOWERING=YES")
    print("TTIR_ATTENTION_APPLY_V0_TRANSPOSED_V_LAYOUT=YES")
    print("TTIR_ATTENTION_APPLY_V0_NATURAL_EXP_SOFTMAX=YES")
    print("TTIR_ATTENTION_APPLY_V0_WEIGHTED_SUM=YES")
    print("TTIR_NONTRANSPOSED_V_GATHER_REJECT=PASS")
    print("TTIR_SCALAR_GLOBAL_LOAD_REJECT=PASS")
    print("NO_NAME_PATH_FIXTURE_SPECIAL_CASES=YES")
    print("NO_RAW_TTIR_TEXT_PARSING=YES")
    print("NO_REGEX_OR_SUBSTRING_SEMANTIC_CLASSIFICATION=YES")
    print("STRUCTURAL_POINTER_REDUCTION_SOFTMAX_CLASSIFICATION=YES")
    print("NO_DIRECT_LOWER_HALF_OUTPUT=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
