#!/usr/bin/env python3
"""Validate a pinned local Triton TTIR generator environment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCE = Path("compiler/test/CodeGen/Triton/Snapshots/ControlFlow/sources/scalar_if_probe.py")
DEFAULT_KERNEL = "scalar_if_probe_kernel"
DEFAULT_SIGNATURE = "*fp32,*fp32,i32,i32,16"
DEFAULT_SPEC = Path("tools/vc4_ttir_generator_toolchain.json")


def fail(message: str) -> "None":
    print(f"vc4_check_ttir_generator.py: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if capture:
            if exc.stdout:
                print(exc.stdout, file=sys.stderr, end="")
            if exc.stderr:
                print(exc.stderr, file=sys.stderr, end="")
        fail(f"command failed: {' '.join(cmd)}")


def require(path: Path, description: str) -> Path:
    if not path.exists():
        fail(f"{description} does not exist: {path}")
    return path


def load_spec(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def audit_text(path: Path, required: list[str], forbidden: list[str]) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    for needle in required:
        count = text.count(needle)
        counts[needle] = count
        if count == 0:
            fail(f"{path}: missing required text {needle!r}")
    for needle in forbidden:
        count = text.count(needle)
        counts[needle] = count
        if count:
            fail(f"{path}: found forbidden text {needle!r}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-root", required=True, type=Path)
    parser.add_argument("--vc4-triton-opt", default=Path("compiler/build-triton-llvm/bin/vc4-triton-opt"), type=Path)
    parser.add_argument("--spec", default=DEFAULT_SPEC, type=Path)
    parser.add_argument("--source", default=DEFAULT_SOURCE, type=Path)
    parser.add_argument("--kernel-name", default=DEFAULT_KERNEL)
    parser.add_argument("--signature", default=DEFAULT_SIGNATURE)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--required", action="append", default=["tt.func", "scf.if", "scf.yield", "tt.load", "tt.store"])
    parser.add_argument(
        "--forbidden",
        action="append",
        default=["ttg.", "triton_gpu.", "nvgpu.", "nvvm.", "arith.sitofp"],
    )
    args = parser.parse_args()

    spec = load_spec(args.spec)
    smoke = spec.get("smoke", {})
    if args.source == DEFAULT_SOURCE and smoke.get("source"):
        args.source = Path(smoke["source"])
    if args.kernel_name == DEFAULT_KERNEL and smoke.get("kernel_name"):
        args.kernel_name = smoke["kernel_name"]
    if args.signature == DEFAULT_SIGNATURE and smoke.get("signature"):
        args.signature = smoke["signature"]
    if args.required == ["tt.func", "scf.if", "scf.yield", "tt.load", "tt.store"]:
        args.required = smoke.get("required_ttir", args.required)
    if args.forbidden == ["ttg.", "triton_gpu.", "nvgpu.", "nvvm.", "arith.sitofp"]:
        args.forbidden = smoke.get("forbidden_ttir", args.forbidden)

    generator_root = require(args.generator_root, "generator root")
    wrapper = require(generator_root / "run_vc4_emit_ttir.sh", "generator wrapper")
    metadata = require(generator_root / "generator_env.json", "generator metadata")
    vc4_triton_opt = require(args.vc4_triton_opt, "vc4-triton-opt")
    source = require(args.source, "Triton source")
    env = json.loads(metadata.read_text(encoding="utf-8"))

    out_dir = args.out_dir or (generator_root / "check")
    ttir = out_dir / "ttir" / f"{args.kernel_name}.ttir.mlir"
    generated_metadata = out_dir / "metadata" / f"{args.kernel_name}.json"
    roundtrip = out_dir / "roundtrip" / f"{args.kernel_name}.roundtrip.mlir"
    ttir.parent.mkdir(parents=True, exist_ok=True)
    generated_metadata.parent.mkdir(parents=True, exist_ok=True)
    roundtrip.parent.mkdir(parents=True, exist_ok=True)

    run(
        [
            str(wrapper),
            str(source),
            "--kernel-name",
            args.kernel_name,
            "--signature",
            args.signature,
            "--target",
            "cuda:80:32",
            "--num-warps",
            "1",
            "--num-stages",
            "3",
            "--required-triton-major-minor",
            "3.7",
            "--out",
            str(ttir),
            "--metadata-out",
            str(generated_metadata),
        ]
    )
    run([str(vc4_triton_opt), str(ttir), "-o", str(roundtrip)])

    ttir_counts = audit_text(ttir, args.required, args.forbidden)
    roundtrip_counts = audit_text(roundtrip, ["tt.func"], args.forbidden)
    generated = json.loads(generated_metadata.read_text(encoding="utf-8"))
    if generated.get("triton_version") != env.get("triton_version"):
        fail(
            "generated metadata Triton version mismatch: "
            f"{generated.get('triton_version')} != {env.get('triton_version')}"
        )

    report = {
        "VC4_TTIR_GENERATOR_CHECK": "PASS",
        "generator_root": str(generator_root),
        "triton_version": env.get("triton_version"),
        "triton_source_head": env.get("triton_source_head"),
        "ttir": str(ttir),
        "roundtrip": str(roundtrip),
        "ttir_counts": ttir_counts,
        "roundtrip_counts": roundtrip_counts,
        "spec": str(args.spec),
        "ready_for_triton": "NO",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("VC4_TTIR_GENERATOR_CHECK=PASS")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
