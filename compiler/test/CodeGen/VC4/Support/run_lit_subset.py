#!/usr/bin/env python3
"""Run a filtered llvm-lit subset for VC4 codegen tests.

This helper is intentionally small and standalone so milestone verification can
run already-established Emit tests without also requiring future M2 tests to
pass before their implementation slices are complete.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable, List


def _find_lit() -> str:
    env_lit = os.environ.get("LLVM_LIT") or os.environ.get("LIT")
    if env_lit:
        return env_lit
    for name in ("llvm-lit", "lit"):
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise SystemExit("run_lit_subset.py: ERROR: could not find llvm-lit or lit in PATH")


def _regex_for_include(names: Iterable[str]) -> str:
    parts = [re.escape(name) for name in names if name]
    if not parts:
        return ".*"
    return r"(?:" + "|".join(parts) + r")"


def _regex_for_exclude(prefixes: Iterable[str]) -> str:
    parts = [re.escape(prefix) for prefix in prefixes if prefix]
    if not parts:
        return ".*"
    return r"^(?!.*(?:" + "|".join(parts) + r")).*$"


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Repository root")
    parser.add_argument("--root", default="compiler/build/test/CodeGen/VC4/Emit",
                        help="Lit root, relative to --repo unless absolute")
    parser.add_argument("--include", action="append", default=[],
                        help="Test basename/substr to include; may repeat")
    parser.add_argument("--exclude-prefix", action="append", default=[],
                        help="Test basename prefix/substr to exclude; may repeat")
    parser.add_argument("--show-cmd", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    root = Path(args.root)
    if not root.is_absolute():
        root = repo / root
    if not root.exists():
        raise SystemExit(f"run_lit_subset.py: ERROR: lit root does not exist: {root}")

    if args.include and args.exclude_prefix:
        raise SystemExit("run_lit_subset.py: ERROR: use either --include or --exclude-prefix, not both")

    if args.include:
        regex = _regex_for_include(args.include)
    else:
        regex = _regex_for_exclude(args.exclude_prefix)

    lit = _find_lit()
    cmd = [lit, "-v", "--filter", regex, str(root)]
    if args.show_cmd:
        print("run_lit_subset.py:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
