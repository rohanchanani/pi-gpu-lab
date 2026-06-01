#!/usr/bin/env python3
"""Check semantic vc4.resource manifest data against generated runtime data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SEMANTIC_FIELDS = {
    "schedule_mode",
    "warps_per_block",
    "user_vpm_rows_per_block",
    "compiler_vpm_staging_rows_per_warp",
    "compiler_vpm_staging_rows_per_block",
    "spill_vpm_rows_per_block",
    "total_vpm_rows_per_block",
    "uses_tmu",
    "uses_vpm",
    "uses_vpm_qpu_read",
    "uses_vpm_qpu_write",
    "uses_vdr",
    "uses_vdw",
    "uses_barrier",
    "semaphore_count_per_block",
    "requires_vpm_base_row_builtin",
    "requires_semaphore_base_builtin",
    "max_resident_blocks",
}


def fail(message: str) -> None:
    print(f"check_resource_runtime_descriptor.py: ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def expect_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def load_kernel(manifest_path: Path, public_name: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kernels = manifest.get("kernels")
    if not isinstance(kernels, list):
        fail("manifest kernels must be a list")
    matches = [
        k for k in kernels
        if isinstance(k, dict) and k.get("public_name") == public_name
    ]
    if len(matches) != 1:
        fail(
            f"expected exactly one kernel public_name={public_name!r}, "
            f"found {len(matches)}"
        )
    return matches[0]


def parse_macro_u32(source: str, name: str) -> int:
    pattern = re.compile(r"(?m)^#define\s+" + re.escape(name) + r"\s+([0-9]+)u\s*$")
    match = pattern.search(source)
    if not match:
        fail(f"generated source missing macro {name}")
    return int(match.group(1))


def require_runtime_descriptor_field(source: str, field: str) -> None:
    if re.search(r"(?<![A-Za-z0-9_])\." + re.escape(field) + r"\b", source) is None:
        fail(f"generated source missing runtime resource descriptor field .{field}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("kernel_launch_c")
    parser.add_argument("--public-name", required=True)
    parser.add_argument(
        "--schedule-mode",
        choices=["independent_vector", "cooperative_block"],
        required=True,
    )
    parser.add_argument("--semantic-total-vpm-rows", type=int, required=True)
    parser.add_argument(
        "--semantic-compiler-vpm-staging-rows-per-warp",
        type=int,
        required=True,
    )
    parser.add_argument("--semantic-user-vpm-rows", type=int, required=True)
    parser.add_argument("--semantic-semaphore-count", type=int, required=True)
    args = parser.parse_args(argv)

    kernel = load_kernel(Path(args.manifest), args.public_name)
    resources = kernel.get("resources")
    if not isinstance(resources, dict):
        fail(f"kernel {args.public_name!r} missing machine-readable resources object")

    missing = sorted(SEMANTIC_FIELDS - set(resources))
    if missing:
        fail(f"resources object missing semantic fields: {missing}")

    expect_equal("resources.schedule_mode", resources["schedule_mode"], args.schedule_mode)
    expect_equal(
        "resources.total_vpm_rows_per_block",
        resources["total_vpm_rows_per_block"],
        args.semantic_total_vpm_rows,
    )
    expect_equal(
        "resources.compiler_vpm_staging_rows_per_warp",
        resources["compiler_vpm_staging_rows_per_warp"],
        args.semantic_compiler_vpm_staging_rows_per_warp,
    )
    expect_equal(
        "resources.user_vpm_rows_per_block",
        resources["user_vpm_rows_per_block"],
        args.semantic_user_vpm_rows,
    )
    expect_equal(
        "resources.semaphore_count_per_block",
        resources["semaphore_count_per_block"],
        args.semantic_semaphore_count,
    )

    source = Path(args.kernel_launch_c).read_text(encoding="utf-8")
    if "VC4_LEGACY_RUNTIME_ADAPTER" in source or "LEGACY_COOPERATIVE" in source:
        fail("generated source still contains temporary runtime adapter names")
    if "struct vc4_kernel_resource" not in source and ".resource = {" not in source:
        fail("generated source lacks a runtime-visible semantic resource descriptor")

    for field in SEMANTIC_FIELDS - {"spill_vpm_rows_per_block", "max_resident_blocks"}:
        require_runtime_descriptor_field(source, field)

    kernel_id = kernel.get("kernel_id")
    if not isinstance(kernel_id, int):
        fail("kernel_id must be an integer")
    semantic_rows = parse_macro_u32(
        source, f"KERNEL_{kernel_id}_TOTAL_VPM_ROWS_PER_BLOCK"
    )
    expect_equal(
        "generated semantic total VPM rows macro",
        semantic_rows,
        args.semantic_total_vpm_rows,
    )

    print(f"check_resource_runtime_descriptor.py: PASS {args.public_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
