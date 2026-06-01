#!/usr/bin/env python3
"""Reject old lower-half resource field identifiers in compiler/runtime source."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


FORBIDDEN = [
    "shared_" "vpm_bytes",
    "user_shared_" "vpm_rows_" "per_block",
    "vpm_" "rows_" "per_block",
    "vpm_" "bytes_" "per_block",
    "warps_" "per_block_max",
    "uses_" "shared_vpm",
    "require_" "full_block_residency",
    "semaphores_" "per_block",
]

ALLOWED_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".td",
    ".mlir",
    ".py",
    ".qasm",
}


def should_scan(path: Path) -> bool:
    if path.suffix not in ALLOWED_SUFFIXES:
        return False
    parts = set(path.parts)
    if "__pycache__" in parts:
        return False
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args(argv)

    pattern = re.compile(
        r"(?<![A-Za-z0-9_])("
        + "|".join(re.escape(name) for name in FORBIDDEN)
        + r")(?![A-Za-z0-9_])"
    )
    failures: list[str] = []
    for root_arg in args.roots:
        root = Path(root_arg)
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in paths:
            if not should_scan(path):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), 1):
                match = pattern.search(line)
                if match:
                    failures.append(f"{path}:{lineno}: old resource field {match.group(1)}")
    if failures:
        print("check_no_old_resource_fields.py: ERROR", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("check_no_old_resource_fields.py: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
