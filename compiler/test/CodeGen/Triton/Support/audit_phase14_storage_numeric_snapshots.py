#!/usr/bin/env python3
"""Audit Phase 14 controlled ML storage/numeric TTIR snapshots."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ACCEPTED = {
    "ttir_f16_load_f32_compute_store_f32_b16",
    "ttir_f32_compute_store_f16_b16",
    "ttir_f16_row_dot_f32_accum_b16",
    "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16",
}

STAGED = {
    "ttir_i32_to_f32_cast_b16",
    "ttir_native_f16_arithmetic_reject_b16",
    "ttir_f32_to_i32_cast_reject_b16",
    "ttir_bf16_or_fp8_reject_b16",
    "ttir_int8_quantized_storage_reject_b16",
}

FORBIDDEN_ACCEPTED = (
    "tt.dot",
    "tt.make_tensor_ptr",
    "tt.advance",
    "tt.atomic",
    "ttg.",
    "triton_gpu.",
    "nvgpu.",
    "nvvm.",
    "math.",
    "arith.sitofp",
    "arith.fptosi",
    "tensor<16xi8>",
    "tensor<16xbf16>",
)


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label}: missing {needle!r}")


def forbid(text: str, needle: str, label: str) -> None:
    if needle in text:
        fail(f"{label}: unexpected {needle!r}")


def main() -> int:
    if len(sys.argv) != 2:
        fail("usage: audit_phase14_storage_numeric_snapshots.py REPO_ROOT")
    root = Path(sys.argv[1])
    manifest_path = root / "examples/triton/phase14_ml_storage_numeric/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("READY_FOR_TRITON") != "NO":
        fail("READY_FOR_TRITON must remain NO")
    if manifest.get("triton_version") != "3.7.0":
        fail("manifest must record Triton 3.7.0")

    snapshots = manifest.get("snapshots", [])
    by_name = {entry.get("name"): entry for entry in snapshots}
    missing = (ACCEPTED | STAGED) - set(by_name)
    if missing:
        fail(f"manifest missing snapshots: {sorted(missing)}")

    for name in ACCEPTED:
        entry = by_name[name]
        if entry.get("expected_classification") != "ACCEPTED_LOWERABLE":
            fail(f"{name}: expected ACCEPTED_LOWERABLE")
        text = (root / entry["generated_ttir"]).read_text(encoding="utf-8")
        require(text, "tt.func public", name)
        require(text, "!tt.ptr<f16>", name)
        if name != "ttir_f32_compute_store_f16_b16":
            require(text, "arith.extf", name)
        if name == "ttir_f32_compute_store_f16_b16":
            require(text, "arith.truncf", name)
            require(text, "tensor<16xf16>", name)
        if "row_dot" in name or "gemv" in name:
            require(text, "tt.reduce", name)
            require(text, "arith.mulf", name)
        for needle in FORBIDDEN_ACCEPTED:
            forbid(text, needle, name)
        if re.search(r"arith\\.addf .+ : tensor<16xf16>", text):
            fail(f"{name}: accepted snapshot contains native f16 addf")

    i32 = (root / by_name["ttir_i32_to_f32_cast_b16"]["generated_ttir"]).read_text(
        encoding="utf-8"
    )
    require(i32, "arith.sitofp", "ttir_i32_to_f32_cast_b16")
    if by_name["ttir_i32_to_f32_cast_b16"].get("current_phase14_2_status") != (
        "STAGED_BY_LOWER_HALF_GAP"
    ):
        fail("i32-to-f32 cast must be staged by current lower-half gap")

    native_f16 = (
        root
        / by_name["ttir_native_f16_arithmetic_reject_b16"]["generated_ttir"]
    ).read_text(encoding="utf-8")
    require(native_f16, "arith.addf", "ttir_native_f16_arithmetic_reject_b16")
    require(native_f16, "tensor<16xf16>", "ttir_native_f16_arithmetic_reject_b16")

    fptosi = (
        root / by_name["ttir_f32_to_i32_cast_reject_b16"]["generated_ttir"]
    ).read_text(encoding="utf-8")
    require(fptosi, "arith.fptosi", "ttir_f32_to_i32_cast_reject_b16")

    bf16 = (root / by_name["ttir_bf16_or_fp8_reject_b16"]["generated_ttir"]).read_text(
        encoding="utf-8"
    )
    require(bf16, "!tt.ptr<bf16>", "ttir_bf16_or_fp8_reject_b16")
    require(bf16, "tensor<16xbf16>", "ttir_bf16_or_fp8_reject_b16")

    int8 = (
        root / by_name["ttir_int8_quantized_storage_reject_b16"]["generated_ttir"]
    ).read_text(encoding="utf-8")
    require(int8, "!tt.ptr<i8>", "ttir_int8_quantized_storage_reject_b16")
    require(int8, "tensor<16xi8>", "ttir_int8_quantized_storage_reject_b16")

    print("PHASE14_STORAGE_NUMERIC_MANIFEST_AUDIT=PASS")
    print(f"PHASE14_ACCEPTED_SNAPSHOTS={len(ACCEPTED)}")
    print(f"PHASE14_STAGED_SNAPSHOTS={len(STAGED)}")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
