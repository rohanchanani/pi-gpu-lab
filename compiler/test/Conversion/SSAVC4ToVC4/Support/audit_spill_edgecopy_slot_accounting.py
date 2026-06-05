#!/usr/bin/env python3
"""Compatibility audit for spill/edge-copy branch layout accounting."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: audit_spill_edgecopy_slot_accounting.py SOURCE", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    hard_audit = Path(__file__).with_name("audit_branch_layout_accounting.py")
    result = subprocess.run(
        [sys.executable, str(hard_audit), str(source)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    print("spill/edge-copy slot accounting audit PASS")
    print(result.stdout, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
