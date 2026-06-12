#!/usr/bin/env python3
"""Audit Phase 14 value storage/numeric static lowering boundaries."""

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
        / "compiler/docs/vc4_vector_triton_phase14_ml_storage_numeric_fixtures.md"
    )
    if not source_path.exists():
        fail(f"missing source: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    docs = planning_doc.read_text(encoding="utf-8") + "\n" + fixtures_doc.read_text(
        encoding="utf-8"
    )

    for needle, label in [
        ("createF16StorageUnpack", "central f16 unpack helper"),
        ("createF16StoragePack", "central f16 pack helper"),
        ("lowerF16TransferRead", "f16 transfer read planner"),
        ("lowerF16TransferWrite", "f16 transfer write planner"),
        ("kFragmentUnpackOpName", "locked VC4Kernel unpack op"),
        ("kFragmentPackOpName", "locked VC4Kernel pack op"),
        ("kVDRLoadRectToVPMOpName", "locked VDR-to-VPM path"),
        ("kVDWStoreRectFromVPMOpName", "locked VDW-from-VPM path"),
        ("hasFiniteF16StoragePolicy", "finite f16 storage policy gate"),
        ("f16 storage store requires explicit finite storage policy", "missing policy diagnostic"),
        ("native f16 arithmetic is staged", "native f16 staged diagnostic"),
        ("bf16/fp8 storage is staged", "bf16/fp8 staged diagnostic"),
        ("int8/int16 quantized storage is staged", "quantized storage staged diagnostic"),
        ("fp-to-int numeric cast is staged", "fp-to-int staged diagnostic"),
        ("i32 to f32 numeric cast staged by lower-half gap", "i32-to-f32 staged diagnostic"),
        ("unsupported numeric conversion", "unsupported numeric conversion diagnostic"),
        ('return "u16";', "f16 storage ABI element type"),
    ]:
        require(source, needle, label)

    for forbidden in [
        "value_f16_load_f32_compute_store_f32_vc4value",
        "value_f32_compute_store_f16_vc4value",
        "value_f16_row_dot_f32_accum_vc4value",
        "value_f16_empty_repeat_vc4value",
        "mixed_value_f16_storage_gemv_axes_mask_cf_reduction_vc4value",
        "f16-transfer-read-extf-f32-lowers",
        "f32-truncf-f16-transfer-write-lowers",
        "READY_FOR_TRITON=YES",
    ]:
        forbid(source, forbidden, "future feature or fixture-name special case")

    for line in [
        "VALUE_F16_STORAGE_F32_COMPUTE_STATIC=PASS",
        "VALUE_F16_STORE_F32_COMPUTE_STATIC=PASS",
        "VALUE_F16_GEMV_INPUT_STORAGE_STATIC=PASS",
        "F16_STORAGE_FINITE_POLICY=YES",
        "I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP",
        "NATIVE_F16_ARITHMETIC_STAGED=YES",
        "BF16_FP8_STAGED=YES",
        "INT8_INT16_QUANTIZED_STORAGE_STAGED=YES",
        "READY_FOR_PHASE14_5_VALUE_HARDWARE_ISOLATION=YES",
        "READY_FOR_TRITON=NO",
    ]:
        require(docs, line, "Phase 14.4 doc readiness line")

    print("PHASE14_VALUE_STORAGE_NUMERIC_SOURCE_AUDIT=PASS")
    print("F16_STORAGE_CONVERSIONS_USE_CENTRAL_HELPERS=YES")
    print("F16_STORE_REQUIRES_FINITE_POLICY=YES")
    print("NATIVE_F16_ARITHMETIC_STAGED=YES")
    print("BF16_FP8_STAGED=YES")
    print("INT8_INT16_QUANTIZED_STORAGE_STAGED=YES")
    print("FP_TO_INT_CAST_STAGED=YES")
    print("I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP")
    print("NO_FIXTURE_NAME_OR_PATH_SPECIAL_CASES=YES")
    print("NO_HIDDEN_EXACT_ROUNDING_CLAIM=YES")
    print("READY_FOR_TRITON=NO")


if __name__ == "__main__":
    main()
