#!/usr/bin/env python3
"""Parse TTIR through Triton's MLIR parser and report an inventory summary."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


TOOL_SCHEMA_VERSION = 1


def fail(message: str) -> "None":
    print(f"vc4_parse_ttir.py: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_triton():
    try:
        import inspect
        import triton
        from triton._C.libtriton import ir
        from triton.backends.compiler import GPUTarget
    except Exception as exc:
        fail(
            "Triton is required for this optional parser. Install pinned Phase 6 "
            "Triton with `python -m pip install 'triton==3.7.0'` or the "
            f"documented source fallback. Import failed: {exc}"
        )
    return triton, ir, GPUTarget, Path(inspect.getfile(triton)).resolve()


def parse_target(target_text: str):
    parts = target_text.split(":")
    if len(parts) != 3:
        fail("target must have form '<backend>:<arch>:<warp-size>', e.g. cuda:80:32")
    backend, arch, warp_size = parts
    arch_value: int | str = int(arch) if arch.isdigit() else arch
    try:
        warp_value = int(warp_size)
    except ValueError:
        fail(f"target warp-size must be an integer, got {warp_size}")
    return backend, arch_value, warp_value


def inventory_ops(parsed_module_text: str) -> dict[str, int]:
    # Inventory only. The TTIR file has already been parsed by Triton's MLIR parser.
    ops: list[str] = []
    for line in parsed_module_text.splitlines():
        stripped = line.strip()
        match = re.match(
            r'(?:%[A-Za-z0-9_.$-]+\s*=\s*)?'
            r'"?([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.$]*)"?\b',
            stripped,
        )
        if match:
            ops.append(match.group(1))
    return dict(sorted(Counter(ops).items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input .ttir.mlir file")
    parser.add_argument("--target", default="cuda:80:32")
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        fail(f"input TTIR file does not exist: {input_path}")

    triton, ir, GPUTarget, triton_file = load_triton()
    target = GPUTarget(*parse_target(args.target))
    backend = triton.compiler.make_backend(target)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    module = ir.parse_mlir_module(str(input_path), context)
    module.context = context
    entry = "<unknown>"
    if hasattr(module, "get_entry_func_name"):
        entry = str(module.get_entry_func_name())

    parsed_text = str(module)
    summary = {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "parse_ttir": "PASS",
        "input": str(input_path),
        "target": args.target,
        "triton_version": getattr(triton, "__version__", "<missing>"),
        "triton_file": str(triton_file),
        "entry_func": entry,
        "op_inventory_note": "Op names are collected from printed module text only after Triton's MLIR parser succeeds; this is inventory, not importing/lowering.",
        "op_inventory": inventory_ops(parsed_text),
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("PARSE_TTIR=PASS")
    print("ENTRY_FUNC=" + entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
