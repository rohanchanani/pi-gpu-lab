#!/usr/bin/env python3
"""Check C++ TTIR-to-VC4Value output without depending on SSA names."""

from __future__ import annotations

import argparse
from pathlib import Path


FORBIDDEN = (
    "tt.",
    "ttg.",
    "triton_gpu.",
    "gpu.",
    "nvgpu.",
    "nvvm.",
    "rocdl.",
    "spirv.",
    "vc4kernel.",
    "ssavc4.",
    "vc4.qpu",
)

COMMON_REQUIRED = (
    "func.func",
    "vc4value.kernel",
    "vc4value.program_id",
    "axis = 0",
    "vector.step",
    "vector.create_mask",
    "vector.transfer_read",
    "vector.transfer_write",
    "return",
)

KIND_REQUIRED = {
    "vector-add": (
        "memref<?xf32, #vc4value.global>",
        "arith.addf",
    ),
    "saxpy-select": (
        "vc4value.fp_domain = \"finite\"",
        "memref<?xf32, #vc4value.global>",
        "vector.broadcast",
        "arith.mulf",
        "arith.addf",
        "arith.cmpf olt",
        "arith.select",
    ),
    "i32-add-select": (
        "memref<?xi32, #vc4value.global>",
        "arith.addi",
        "vector.broadcast",
        "arith.subi",
        "arith.cmpi sgt",
        "arith.select",
    ),
}


def fail(message: str) -> None:
    print(f"TTIR_TO_VC4VALUE_OUTPUT_CHECK=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("value_ir")
    parser.add_argument("--kind", choices=sorted(KIND_REQUIRED), required=True)
    args = parser.parse_args()

    text = Path(args.value_ir).read_text(encoding="utf-8")
    for needle in FORBIDDEN:
      if needle in text:
        fail(f"forbidden dialect/lower-half marker found: {needle}")
    for needle in COMMON_REQUIRED + KIND_REQUIRED[args.kind]:
      if needle not in text:
        fail(f"required value IR marker missing: {needle}")

    print("TTIR_TO_VC4VALUE_OUTPUT_CHECK=PASS")
    print(f"TTIR_TO_VC4VALUE_KIND={args.kind}")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
