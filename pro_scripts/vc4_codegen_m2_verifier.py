#!/usr/bin/env python3
"""Compatibility wrapper for VC4 Codegen M2 typed verification.

The canonical implementation remains pro_scripts/vc4_codegen_m1_verifier.py so
M1 and M2 share one cumulative verifier.  This wrapper only supplies M2 default
--spec/--worklist values when the caller omits them.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import vc4_codegen_m1_verifier as verifier  # type: ignore


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--spec" not in args:
        args.extend(["--spec", "pro_scripts/vc4_codegen_m2_verifications.json"])
    if "--worklist" not in args:
        args.extend(["--worklist", "pro_scripts/vc4_codegen_m2_worklist.json"])
    return verifier.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
