#!/usr/bin/env python3
"""Generate parser-valid Phase 7c negative TTIR probes from vector_add_b16."""

from __future__ import annotations

import argparse
from pathlib import Path


def mutate(base: str, kind: str) -> str:
    if kind == "make_range_end8":
        text = base.replace(
            "end = 16 : i32, start = 0 : i32",
            "end = 8 : i32, start = 0 : i32",
            1,
        )
        return (
            text.replace("tensor<16xi32>", "tensor<8xi32>")
            .replace("tensor<16xi64>", "tensor<8xi64>")
            .replace("tensor<16xi1>", "tensor<8xi1>")
            .replace("tensor<16xf32>", "tensor<8xf32>")
            .replace("tensor<16x!tt.ptr<f32>>", "tensor<8x!tt.ptr<f32>>")
        )
    if kind == "nonzero_other":
        return base.replace(
            "dense<0.000000e+00> : tensor<16xf32>",
            "dense<1.000000e+00> : tensor<16xf32>",
            1,
        )
    if kind == "load_nomask":
        return base.replace(
            "tt.load %x_24, %mask_23, %x_26 : tensor<16x!tt.ptr<f32>>",
            "tt.load %x_24 : tensor<16x!tt.ptr<f32>>",
            1,
        )
    if kind == "store_nomask":
        return base.replace(
            "tt.store %1, %2, %mask_23 : tensor<16x!tt.ptr<f32>>",
            "tt.store %1, %2 : tensor<16x!tt.ptr<f32>>",
            1,
        )
    if kind == "axis_y":
        return base.replace("tt.get_program_id x : i32", "tt.get_program_id y : i32", 1)
    if kind == "store_scatter":
        return base.replace(
            "tt.addptr %0, %offsets_22 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>",
            "tt.addptr %0, %offsets_10 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>",
            1,
        )
    if kind == "load_scatter":
        return base.replace(
            "tt.addptr %x, %offsets_22 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>",
            "tt.addptr %x, %offsets_10 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>",
            1,
        )
    if kind == "rank2_probe":
        return base.replace(
            "%offsets_11 = tt.splat %offsets_9",
            "%rank2_probe = tt.expand_dims %offsets_10 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>\n"
            "    %offsets_11 = tt.splat %offsets_9",
            1,
        )
    if kind == "f16_type":
        return (
            base.replace("!tt.ptr<f32>", "!tt.ptr<f16>")
            .replace("tensor<16xf32>", "tensor<16xf16>")
            .replace(": f32", ": f16")
        )
    raise SystemExit(f"unknown negative TTIR kind: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = Path(args.base).read_text(encoding="utf-8")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(mutate(base, args.kind), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
