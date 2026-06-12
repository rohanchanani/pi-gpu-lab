#!/usr/bin/env python3
"""Audit Phase 7 TTIR static pipeline layer boundaries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PRODUCER_OPS = (
    "tt.",
    "ttg.",
    "gpu.",
    "nvgpu.",
    "nvvm.",
    "rocdl.",
    "llvm.",
    "func.func",
    "vector.",
    "memref.",
    "vc4value.",
)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssertionError(f"{path}: could not read: {exc}") from exc


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: missing required text {needle!r}")


def require_any(text: str, needles: tuple[str, ...], label: str) -> None:
    if not any(needle in text for needle in needles):
        choices = ", ".join(repr(needle) for needle in needles)
        raise AssertionError(f"{label}: missing one of required texts {choices}")


def forbid(text: str, needles: tuple[str, ...], label: str) -> None:
    for needle in needles:
        if needle in text:
            raise AssertionError(f"{label}: forbidden boundary text {needle!r}")


def forbid_op_token(text: str, token: str, label: str) -> None:
    if re.search(rf"(^|\s){re.escape(token)}(\s|$)", text):
        raise AssertionError(f"{label}: forbidden boundary op {token!r}")


def audit_value(path: Path) -> None:
    text = read_text(path)
    for needle in ("func.func", "vc4value.kernel", "vc4value.program_id",
                   "vector.create_mask",
                   "vector.transfer_read", "memref<"):
        require(text, needle, str(path))
    require_any(text, ("vector.transfer_write", "memref.store"), str(path))
    forbid(
        text,
        (
            "tt.",
            "ttg.",
            "gpu.",
            "nvgpu.",
            "nvvm.",
            "rocdl.",
            "llvm.",
            "vc4kernel.",
            "ssavc4.",
            "vc4.module",
            "vc4.qpu",
        ),
        str(path),
    )


def audit_vc4kernel(path: Path) -> None:
    text = read_text(path)
    for needle in ("vc4kernel.kernel", "vc4kernel.program_id",
                   "vc4kernel.tmu_load_fragment", "vc4kernel.vdw_store_fragment"):
        require(text, needle, str(path))
    forbid(
        text,
        (
            "tt.",
            "ttg.",
            "gpu.",
            "nvgpu.",
            "nvvm.",
            "rocdl.",
            "llvm.",
            "func.func",
            "vector.",
            "memref.",
            "vc4value.",
            "ssavc4.",
            "vc4.module",
            "vc4.qpu",
        ),
        str(path),
    )


def audit_ssavc4(path: Path) -> None:
    text = read_text(path)
    for needle in ("ssavc4.func", "ssavc4.tmu.request", "ssavc4.vdw.store"):
        require(text, needle, str(path))
    forbid(text, PRODUCER_OPS + ("vc4kernel.", "vc4.qpu"), str(path))
    forbid_op_token(text, "vc4.module", str(path))


def audit_vc4(path: Path) -> None:
    text = read_text(path)
    for needle in ("vc4.module", "vc4.func", "vc4.qpu."):
        require(text, needle, str(path))
    forbid(text, PRODUCER_OPS + ("vc4kernel.", "ssavc4."), str(path))


def audit_bundle(path: Path) -> None:
    manifest_path = path / "manifest.json"
    for required in (path / "kernel_launch.c", path / "kernel_launch.h", manifest_path):
        if not required.is_file():
            raise AssertionError(f"{path}: missing generated artifact {required.name}")
    kernels_dir = path / "kernels"
    qasm_files = sorted(kernels_dir.glob("*.qasm")) if kernels_dir.is_dir() else []
    if not qasm_files:
        raise AssertionError(f"{path}: missing generated QASM artifact")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{manifest_path}: invalid JSON: {exc}") from exc
    manifest_text = json.dumps(manifest, sort_keys=True)
    for needle in ('"uses_tmu": true', '"uses_vdw": true'):
        if needle not in manifest_text:
            raise AssertionError(f"{manifest_path}: missing {needle}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value", type=Path, required=True)
    parser.add_argument("--vc4kernel", type=Path, required=True)
    parser.add_argument("--ssavc4", type=Path, required=True)
    parser.add_argument("--vc4", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit_value(args.value)
        audit_vc4kernel(args.vc4kernel)
        audit_ssavc4(args.ssavc4)
        audit_vc4(args.vc4)
        if args.bundle is not None:
            audit_bundle(args.bundle)
    except AssertionError as exc:
        print(f"TTIR_STATIC_PIPELINE_AUDIT=FAIL: {exc}", file=sys.stderr)
        return 1
    print("TTIR_STATIC_PIPELINE_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
