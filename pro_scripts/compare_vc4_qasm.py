#!/usr/bin/env python3
"""Compare VC4 QASM files after removing non-operational noise."""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path


LABEL_RE = re.compile(r"^:vc4_qpu_slot_\d+\s*$")
BRANCH_LABEL_COMMENT_RE = re.compile(r"\s*;+\s*branch target:?\s*:vc4_qpu_slot_\d+\s*$", re.I)
SLOT_COMMENT_RE = re.compile(r"\s*;+\s*vc4_qpu_slot_\d+\s*$", re.I)


def normalize_line(line: str) -> str | None:
    line = line.rstrip()
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#") or stripped.startswith("//") or stripped.startswith(";"):
        return None
    if LABEL_RE.match(stripped):
        return None

    line = BRANCH_LABEL_COMMENT_RE.sub("", line)
    line = SLOT_COMMENT_RE.sub("", line)

    # Drop comments that are purely labels while preserving ordinary operands
    # and immediates. Handwritten QASM often annotates branch offsets this way.
    line = re.sub(r"\s+//\s*:vc4_qpu_slot_\d+\s*$", "", line)
    line = re.sub(r"\s+#\s*:vc4_qpu_slot_\d+\s*$", "", line)

    return re.sub(r"\s+", " ", line.strip())


def normalize(path: Path) -> list[str]:
    return [line for raw in path.read_text().splitlines() if (line := normalize_line(raw)) is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("-U", "--context", type=int, default=3)
    args = parser.parse_args()

    ref = normalize(args.reference)
    cand = normalize(args.candidate)
    if ref == cand:
        print(f"normalized QASM matches: {args.reference} == {args.candidate}")
        return 0

    diff = difflib.unified_diff(
        ref,
        cand,
        fromfile=str(args.reference),
        tofile=str(args.candidate),
        lineterm="",
        n=args.context,
    )
    print("\n".join(diff))
    return 1


if __name__ == "__main__":
    sys.exit(main())
