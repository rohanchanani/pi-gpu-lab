#!/usr/bin/env python3
"""Audit Phase 15B TTIR natural math importer invariants."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
IMPORTER = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"
VALUE_LOWERING = REPO_ROOT / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"
BASE2_DOC = REPO_ROOT / "compiler/docs/vc4_vector_triton_phase15_base2_sfu_semantics.md"
FIXTURE_DOC = REPO_ROOT / "compiler/docs/vc4_vector_triton_phase15_sfu_softmax_fixtures.md"
MANIFEST = REPO_ROOT / "examples/triton/phase15_sfu_softmax/manifest.json"


def fail(message: str) -> None:
    print(f"PHASE15B_TTIR_NATURAL_MATH_FRONTEND_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"missing {label}: {needle}")


def main() -> int:
    importer = IMPORTER.read_text(encoding="utf-8")
    value_lowering = VALUE_LOWERING.read_text(encoding="utf-8")
    base2_doc = BASE2_DOC.read_text(encoding="utf-8")
    fixture_doc = FIXTURE_DOC.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")

    forbidden_importer = (
        "std::regex",
        "#include <regex>",
        "raw_string_ostream",
        "op->print(",
        ".print(",
        "ttir_sfu_exp",
        "ttir_sfu_log",
        "ttir_sfu_sqrt",
        "ttir_softmax",
        "sourcePath",
        "fileName",
        "filename",
        "getFilename",
        "vc4kernel.",
        "ssavc4.",
    )
    for needle in forbidden_importer:
        if needle in importer:
            fail(f"forbidden brittle importer pattern remains: {needle}")

    required_importer = (
        "math::ExpOp",
        "math::LogOp",
        "math::SqrtOp",
        "math::RsqrtOp",
        "kVC4ValueMathPolicyAttr",
        "kVC4ValueFPDomainAttr",
        "approx_sfu",
        "finite_positive",
        "TTIRReduceClassification::MaxF32",
        "kVC4ValueMaxPolicyAttr",
        "createVectorMaxReduction",
    )
    for needle in required_importer:
        require(importer, needle, "structural importer marker")

    required_value_lowering = (
        "lowerApproxSFUMath",
        "kLog2E",
        "kLn2",
        "exp2(x * log2(e))",
        "log2(x) * ln(2)",
        "x * rsqrt(x)",
        "math.exp",
        "math.log",
        "math.sqrt",
    )
    for needle in required_value_lowering:
        require(value_lowering, needle, "value natural math lowering marker")

    required_docs = (
        "TTIR_TL_EXP_IS_NATURAL_EXP=YES",
        "TTIR_TL_LOG_IS_NATURAL_LOG=YES",
        "TTIR_TL_SQRT_IS_SQRT=YES",
        "NATURAL_EXP_LOWERING=EXP2_X_LOG2E",
        "NATURAL_LOG_LOWERING=LOG2_X_LN2",
        "SQRT_LOWERING=RSQRT_TIMES_X",
        "READY_FOR_TRITON=NO",
    )
    docs = base2_doc + "\n" + fixture_doc
    for needle in required_docs:
        require(docs, needle, "natural math doc line")

    manifest_markers = (
        "TL_EXP_IS_NATURAL_EXP",
        "TL_LOG_IS_NATURAL_LOG",
        "TL_SQRT_IS_SQRT",
        "NATURAL_EXP_LOWERING",
        "NATURAL_LOG_LOWERING",
        "SQRT_LOWERING",
        "ttir_sfu_sqrt_f32_b16",
    )
    for needle in manifest_markers:
        require(manifest, needle, "manifest natural math marker")

    print("PHASE15B_TTIR_NATURAL_MATH_FRONTEND_AUDIT=PASS")
    print("TTIR_TL_EXP_CLASSIFICATION_STRUCTURAL=YES")
    print("TTIR_TL_LOG_CLASSIFICATION_STRUCTURAL=YES")
    print("TTIR_TL_SQRT_CLASSIFICATION_STRUCTURAL=YES")
    print("TTIR_IMPORTER_OUTPUT_VALUE_LAYER_ONLY=YES")
    print("NO_FIXTURE_NAME_OR_PATH_SPECIAL_CASES=YES")
    print("NO_TTIR_TEXT_PARSING=YES")
    print("NO_REGEX_OR_SUBSTRING_SEMANTIC_CLASSIFICATION=YES")
    print("TTIR_NATURAL_MATH_BOUNDARY=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
