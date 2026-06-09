#!/usr/bin/env python3
"""Create parser-valid staged TTIR probes from real Phase 6 snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path


AXIS_REPLACEMENTS = {
    "axis-y": "tt.get_program_id y : i32",
    "axis-z": "tt.get_program_id z : i32",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--kind", choices=sorted(AXIS_REPLACEMENTS), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = Path(args.base).read_text(encoding="utf-8")
    if "tt.get_program_id x : i32" not in base:
      raise SystemExit("base TTIR does not contain the expected axis-x program id")
    text = base.replace("tt.get_program_id x : i32", AXIS_REPLACEMENTS[args.kind], 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
