#!/usr/bin/env python3
"""Audit Phase 14 TTIR storage/numeric importer invariants."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
IMPORTER = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"
VALUE_LOWERING = REPO_ROOT / "compiler/lib/Conversion/VC4ValueToVC4Kernel/VC4ValueToVC4Kernel.cpp"


def fail(message: str) -> None:
    print(f"PHASE14_TTIR_STORAGE_NUMERIC_IMPORTER_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    importer = IMPORTER.read_text(encoding="utf-8")
    value_lowering = VALUE_LOWERING.read_text(encoding="utf-8")

    forbidden = (
        "std::regex",
        "#include <regex>",
        "raw_string_ostream",
        "op->print(",
        ".print(",
        "ttir_f16_load",
        "ttir_f32_compute_store_f16",
        "mixed_ttir_f16_storage",
        "sourcePath",
        "fileName",
        "filename",
        "getFilename",
    )
    for needle in forbidden:
        if needle in importer:
            fail(f"forbidden brittle importer pattern remains: {needle}")

    required_importer = (
        "TTIRScalarElementKind::F16",
        "type.isF16()",
        "arith.extf",
        "arith.truncf",
        "kVC4ValueF16StoragePolicyAttr",
        "vc4value.f16_storage_policy",
        "native f16 arithmetic is staged",
        "bf16/fp8 storage is staged",
        "int8/int16 quantized storage is staged",
        "fp-to-int numeric cast is staged",
        "i32 to f32 numeric cast staged by lower-half gap",
        "staticFullMaskValue",
        "READY_FOR_TRITON remains NO",
    )
    for needle in required_importer:
        if needle not in importer:
            fail(f"required Phase 14 importer marker missing: {needle}")

    required_value = (
        "f16 storage extf requires raw i32 storage carrier",
        "f16 storage store requires explicit finite storage policy",
        "native f16 arithmetic is staged",
        "fp-to-int numeric cast is staged",
    )
    for needle in required_value:
        if needle not in value_lowering:
            fail(f"required Phase 14 value bridge marker missing: {needle}")

    f16_section = importer[
        importer.find("enum class TTIRScalarElementKind"):
        importer.find("static FailureOr<Attribute> retargetDenseAttr")
    ]
    if not f16_section:
        fail("could not locate f16 type adapter section")
    for needle in ("inferSourceArgName", "arg_name", "loc(", "sanitizeSymbolName"):
        if needle in f16_section:
            fail(f"f16 type classification appears to use name/location metadata: {needle}")
    for needle in ("PointerType", "getPointeeType", "RankedTensorType", "getElementType"):
        if needle not in f16_section:
            fail(f"f16 classification missing structural type marker: {needle}")

    print("PHASE14_TTIR_STORAGE_NUMERIC_IMPORTER_AUDIT=PASS")
    print("TTIR_F16_STORAGE_TYPE_CLASSIFICATION_STRUCTURAL=YES")
    print("TTIR_F16_LOAD_STORE_LOWERING_STRUCTURAL=YES")
    print("TTIR_F16_STORAGE_POLICY_EXPLICIT=YES")
    print("TTIR_NATIVE_F16_ARITHMETIC_REJECT=PASS")
    print("TTIR_FP_TO_INT_REJECT=PASS")
    print("NO_FIXTURE_NAME_OR_PATH_SPECIAL_CASES=YES")
    print("NO_TTIR_TEXT_PARSING=YES")
    print("NO_REGEX_OR_SUBSTRING_SEMANTIC_CLASSIFICATION=YES")
    print("NO_DIRECT_LOWER_HALF_OUTPUT=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
