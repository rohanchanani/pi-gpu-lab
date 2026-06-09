#!/usr/bin/env python3
"""Create parser-valid staged TTIR probes from real Phase 6 snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path


AXIS_REPLACEMENTS = {
    "axis-y": "tt.get_program_id y : i32",
    "axis-z": "tt.get_program_id z : i32",
}

RANGE_REPLACEMENTS = {
    "make-range-1-17": "tt.make_range {end = 17 : i32, start = 1 : i32}",
}

NAME_REPLACEMENTS = {
    "tail-bound-weird-name": (
        ("%n_elements: i32 loc(\"n_elements\"(#loc))", "%items: i32 loc(\"items\"(#loc))"),
        ("%n_elements", "%items"),
    ),
    "compute-i32-name-n": (
        ("%bias: i32 loc(\"bias\"(#loc))", "%n: i32 loc(\"n\"(#loc))"),
        ("%bias", "%n"),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument(
        "--kind",
        choices=sorted(
            set(AXIS_REPLACEMENTS) | set(RANGE_REPLACEMENTS) | set(NAME_REPLACEMENTS)
        ),
        required=True,
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = Path(args.base).read_text(encoding="utf-8")
    if args.kind in AXIS_REPLACEMENTS:
      if "tt.get_program_id x : i32" not in base:
        raise SystemExit("base TTIR does not contain the expected axis-x program id")
      text = base.replace("tt.get_program_id x : i32", AXIS_REPLACEMENTS[args.kind], 1)
    elif args.kind in RANGE_REPLACEMENTS:
      marker = "tt.make_range {end = 16 : i32, start = 0 : i32}"
      if marker not in base:
        raise SystemExit("base TTIR does not contain the expected make_range 0..16")
      text = base.replace(marker, RANGE_REPLACEMENTS[args.kind], 1)
    else:
      text = base
      for old, new in NAME_REPLACEMENTS[args.kind]:
        if old not in text:
          raise SystemExit(f"base TTIR does not contain expected name marker: {old}")
        text = text.replace(old, new)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
