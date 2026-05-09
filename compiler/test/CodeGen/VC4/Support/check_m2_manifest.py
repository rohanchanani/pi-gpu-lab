#!/usr/bin/env python3
"""Validate the VC4 M2 manifest-v2 shape used by lit tests.

This deliberately checks only stable contract fields. The richer typed verifier
performs the full source-of-truth validation during slice verification.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REQ_TOP = {"schema_version", "kind", "program_name", "target", "kernels"}
REQ_KERNEL = {"kernel_id", "symbol_name", "public_name", "qasm_path", "code_symbol", "scheduled_sink_ops", "uniform_words_per_request", "tail_policy", "schedule_mode", "args", "builtins", "resources"}
REQ_TARGET = {"name", "warp_size", "max_active_qpus", "shared_vpm_bytes", "semaphores"}


def fail(msg: str) -> None:
    print(f"check_m2_manifest.py: ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--schema-version", type=int, default=2)
    ap.add_argument("--kernel-count", type=int)
    ap.add_argument("--require-target", action="store_true")
    ap.add_argument("--require-kernel-fields", action="store_true")
    ap.add_argument("--public-name", action="append", default=[])
    ap.add_argument("--arg-direction", action="append", default=[], help="NAME:DIRECTION expected in some kernel args")
    ns = ap.parse_args(argv)

    path = Path(ns.manifest)
    data = json.loads(path.read_text())
    missing_top = sorted(REQ_TOP - set(data))
    if missing_top:
        fail(f"missing top-level keys: {missing_top}")
    if data.get("schema_version") != ns.schema_version:
        fail(f"schema_version expected {ns.schema_version}, got {data.get('schema_version')!r}")
    if data.get("kind") != "vc4-codegen-artifact-bundle":
        fail("kind must be vc4-codegen-artifact-bundle")
    kernels = data.get("kernels")
    if not isinstance(kernels, list):
        fail("kernels must be a list")
    if ns.kernel_count is not None and len(kernels) != ns.kernel_count:
        fail(f"expected {ns.kernel_count} kernels, got {len(kernels)}")
    if ns.require_target:
        target = data.get("target")
        if not isinstance(target, dict):
            fail("target must be an object")
        missing = sorted(REQ_TARGET - set(target))
        if missing:
            fail(f"target missing keys: {missing}")
    ids=set(); publics=set(); code_symbols=set(); qasm_paths=set()
    for k in kernels:
        if not isinstance(k, dict):
            fail("each kernel entry must be an object")
        if ns.require_kernel_fields:
            missing = sorted(REQ_KERNEL - set(k))
            if missing:
                fail(f"kernel {k.get('public_name')} missing keys: {missing}")
        kid=k.get("kernel_id")
        public=k.get("public_name")
        code=k.get("code_symbol")
        qasm=k.get("qasm_path")
        if kid in ids: fail(f"duplicate kernel_id {kid!r}")
        if public in publics: fail(f"duplicate public_name {public!r}")
        if code in code_symbols: fail(f"duplicate code_symbol {code!r}")
        if qasm in qasm_paths: fail(f"duplicate qasm_path {qasm!r}")
        ids.add(kid); publics.add(public); code_symbols.add(code); qasm_paths.add(qasm)
        if not isinstance(qasm, str) or not qasm or qasm.startswith('/') or '..' in Path(qasm).parts:
            fail(f"invalid bundle-relative qasm_path {qasm!r}")
    for public in ns.public_name:
        if public not in publics:
            fail(f"missing public_name {public!r}")
    expected_dirs=[]
    for item in ns.arg_direction:
        if ':' not in item:
            fail(f"--arg-direction must be NAME:DIRECTION, got {item!r}")
        expected_dirs.append(tuple(item.split(':',1)))
    for name, direction in expected_dirs:
        found=False
        for k in kernels:
            for a in k.get('args', []):
                if isinstance(a, dict) and a.get('name') == name and a.get('direction') == direction:
                    found=True
        if not found:
            fail(f"missing arg direction {name}:{direction}")
    print(f"check_m2_manifest.py: PASS {path} kernels={len(kernels)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
