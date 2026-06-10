#!/usr/bin/env python3
"""Audit controlled TTIR snapshots for strict VC4Value output boundaries."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


FORBIDDEN_MARKERS = (
    "builtin.unrealized_conversion_cast",
    "tt.",
    "ttg.",
    "triton_gpu.",
    "nvgpu.",
    "nvvm.",
    "rocdl.",
    "gpu.",
    "llvm.",
    "spirv.",
    "tensor.",
    "linalg.",
    "stablehlo.",
    "mhlo.",
    "tosa.",
    "iree.",
    "vc4kernel.",
    "ssavc4.",
    "vc4.",
    "!tt.ptr",
    "tensor<",
)


def fail(message: str) -> None:
    print(f"TTIR_VALUE_OUTPUT_BOUNDARY_AUDIT=FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--vc4-triton-opt", required=True)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    tool = Path(args.vc4_triton_opt).resolve()
    snapshots = (
        "examples/triton/phase8_5_control_flow/generated/ttir_cf_scalar_if_b16.ttir.mlir",
        "examples/triton/phase8_5_control_flow/generated/ttir_cf_tl_range_loop_b16.ttir.mlir",
        "examples/triton/phase8_5_control_flow/generated/ttir_cf_while_loop_b16.ttir.mlir",
        "examples/triton/phase8_5_control_flow/generated/ttir_cf_persistent_loop_b16.ttir.mlir",
        "examples/triton/phase8_5_control_flow/generated/mixed_ttir_cf_loop_if_tail_b16.ttir.mlir",
        "examples/triton/phase8_5_control_flow/generated/ttir_cf_static_range_unrolled_b16.ttir.mlir",
    )
    checked = 0
    with tempfile.TemporaryDirectory(prefix="vc4_ttir_value_boundary_") as tmpdir:
        tmp = Path(tmpdir)
        for ttir_rel in snapshots:
            ttir = repo / ttir_rel
            if not ttir.is_file():
                fail(f"controlled TTIR snapshot missing: {ttir}")
            out = tmp / f"{Path(ttir_rel).stem}.value.mlir"
            result = subprocess.run(
                [str(tool), str(ttir), "--convert-triton-to-vc4-value", "-o", str(out)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if result.returncode != 0:
                fail(f"lowering failed for {ttir.relative_to(repo)}:\n{result.stdout}")
            text = out.read_text(encoding="utf-8")
            for marker in FORBIDDEN_MARKERS:
                if marker in text:
                    fail(
                        f"forbidden output marker {marker!r} in "
                        f"{ttir_rel}"
                    )
            checked += 1

    print("TTIR_VALUE_OUTPUT_BOUNDARY_AUDIT=PASS")
    print("CONTROLLED_TTIR_VALUE_OUTPUT_BOUNDARY=PASS")
    print(f"CONTROLLED_TTIR_SNAPSHOTS_CHECKED={checked}")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
