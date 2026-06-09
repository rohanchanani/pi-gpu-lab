#!/usr/bin/env python3
"""Emit frontend TTIR from a real Triton JIT kernel.

This is optional Phase 6 tooling. It uses Triton frontend APIs only and never
emits TTGIR, PTX, cubin, hsaco, or VC4 IR.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


TOOL_SCHEMA_VERSION = 1


def fail(message: str) -> "None":
    print(f"vc4_emit_ttir.py: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_triton(required_major_minor: str):
    try:
        import inspect
        import triton
        from triton._C.libtriton import ir
        from triton.backends.compiler import GPUTarget
    except Exception as exc:
        fail(
            "Triton is required for this optional tool. Install pinned Phase 6 "
            "Triton with `python -m pip install 'triton==3.7.0'` or the "
            f"documented source fallback. Import failed: {exc}"
        )

    version = getattr(triton, "__version__", "<missing>")
    if not str(version).startswith(required_major_minor + "."):
        fail(
            f"expected Triton {required_major_minor}.x, found {version}. "
            "Use tools/requirements-triton-phase6.txt or the pinned v3.7.0 source fallback."
        )
    return triton, ir, GPUTarget, Path(inspect.getfile(triton)).resolve()


def constexpr(value: str) -> int | float | None:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return None


def parse_signature(kernel: Any, signature_text: str):
    entries = [entry.strip(" ") for entry in signature_text.split(",")]
    if len(entries) > len(kernel.arg_names):
        fail(
            f"signature has {len(entries)} entries but kernel has only "
            f"{len(kernel.arg_names)} arguments"
        )

    hints = {
        (i,): constexpr(entry.split(":", 1)[1])
        for i, entry in enumerate(entries)
        if ":" in entry
    }
    hints = {key: value for key, value in hints.items() if value is not None}
    for value in hints.values():
        if value not in (1, 16):
            fail(f"only type hints 1 and 16 are supported, got {value}")

    constants = {
        kernel.arg_names[i]: constexpr(entry)
        for i, entry in enumerate(entries)
        if constexpr(entry) is not None
    }
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value

    signature = {
        kernel.arg_names[i]: entry.split(":", 1)[0]
        for i, entry in enumerate(entries)
    }
    for key in constants:
        signature[key] = "constexpr"

    attrs = {key: [["tt.divisibility", 16]] for key, value in hints.items() if value == 16}
    return entries, constants, signature, attrs, hints


def attrs_for_json(attrs: dict[tuple[int], list[list[Any]]]) -> dict[str, list[list[Any]]]:
    return {str(key[0]): value for key, value in attrs.items()}


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


def import_kernel(source_path: Path, kernel_name: str):
    if not source_path.is_file():
        fail(f"source file does not exist: {source_path}")
    sys.path.insert(0, str(source_path.parent))
    spec = importlib.util.spec_from_file_location(source_path.stem, source_path)
    if spec is None or spec.loader is None:
        fail(f"could not import source file: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, kernel_name)
    except AttributeError:
        fail(f"source does not define kernel {kernel_name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Python source containing a @triton.jit kernel")
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--target", default="cuda:80:32")
    parser.add_argument("--num-warps", type=int, default=1)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--out", required=True, help="Output .ttir.mlir path")
    parser.add_argument("--metadata-out", required=True, help="Output metadata JSON path")
    parser.add_argument("--required-triton-major-minor", default="3.7")
    args = parser.parse_args()

    triton, ir, GPUTarget, triton_file = load_triton(args.required_triton_major_minor)
    source_path = Path(args.source).resolve()
    out_path = Path(args.out)
    metadata_path = Path(args.metadata_out)

    kernel = import_kernel(source_path, args.kernel_name)
    entries, constants, signature, attrs, hints = parse_signature(kernel, args.signature)

    if hasattr(kernel, "create_binder"):
        try:
            kernel.create_binder()
        except RuntimeError as exc:
            if "active drivers" not in str(exc):
                raise

    target = GPUTarget(*parse_target(args.target))
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": args.num_warps, "num_stages": args.num_stages})

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    source_cls = getattr(kernel, "ASTSource", triton.compiler.ASTSource)
    src = source_cls(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)
    module = src.make_ir(
        target,
        options,
        backend.get_codegen_implementation(options),
        backend.get_module_map(),
        context,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(str(module), encoding="utf-8")

    source_bytes = source_path.read_bytes()
    metadata = {
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "triton_version": getattr(triton, "__version__", "<missing>"),
        "triton_file": str(triton_file),
        "source": str(source_path),
        "kernel_name": args.kernel_name,
        "signature": signature,
        "signature_entries": entries,
        "constants": constants,
        "attrs": attrs_for_json(attrs),
        "hints": {str(key[0]): value for key, value in hints.items()},
        "target": args.target,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "output_ttir": str(out_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

