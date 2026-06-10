#!/usr/bin/env python3
"""Audit Phase 10 TTIR mask/memory importer constraints."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE = REPO_ROOT / "compiler/lib/Conversion/TritonToVC4Value/TritonToVC4Value.cpp"


def fail(message: str) -> None:
    print(f"PHASE10_TTIR_MASK_MEMORY_IMPORTER_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    source = SOURCE.read_text(encoding="utf-8")

    required = (
        "matchCanonicalTailMask",
        "recordCanonicalTailMask",
        "sparse or unknown tt.store memory mask",
        "sparse or unknown tt.load memory mask",
        "tt.load nonzero other value",
        "op->getNumOperands() != 1 && op->getNumOperands() != 3",
        "op->getNumOperands() != 2 && op->getNumOperands() != 3",
        "mask ? 1 : 0",
        "createVectorTransferRead",
        "createVectorTransferWrite",
    )
    for needle in required:
        if needle not in source:
            fail(f"required structural mask/memory marker missing: {needle}")

    forbidden = (
        "std::regex",
        "#include <regex>",
        "raw_string_ostream",
        "op->print(",
        ".print(",
        "fixture",
        "kernel_name",
        "generated_ttir",
        "ttir_mask_tail_load_store_b16",
        "ttir_mask_full_no_mask_b16",
        "ttir_mask_compute_select_tail_b16",
        "mixed_ttir_mask_memory_cf_axes_b16",
        "ttir_mask_sparse_store_reject_b16",
        "ttir_mask_nonzero_other_reject_b16",
    )
    for needle in forbidden:
        if needle in source:
            fail(f"forbidden fixture/text semantic marker remains: {needle}")

    lower_half_markers = ("vc4kernel.", "ssavc4.", "vc4.module")
    for marker in lower_half_markers:
        if marker in source:
            fail(f"direct lower-half output marker appears in importer: {marker}")

    print("PHASE10_TTIR_MASK_MEMORY_IMPORTER_AUDIT=PASS")
    print("TTIR_MASK_CLASSIFICATION_STRUCTURAL=YES")
    print("NO_FIXTURE_NAME_OR_PATH_SPECIAL_CASES=YES")
    print("NO_TTIR_TEXT_PARSING=YES")
    print("NO_REGEX_OR_SUBSTRING_SEMANTIC_CLASSIFICATION=YES")
    print("NO_DIRECT_LOWER_HALF_OUTPUT=YES")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
