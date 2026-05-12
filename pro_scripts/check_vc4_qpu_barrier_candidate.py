#!/usr/bin/env python3
"""Reject dishonest qpu_barrier_syncthreads generated candidates."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(message)
    sys.exit(1)


def semantic_qasm_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(".") or line.startswith(":"):
            continue
        lines.append(re.sub(r"\s+", " ", line))
    return lines


def main(argv: list[str]) -> int:
    repo = Path(argv[1]) if len(argv) > 1 else Path.cwd()
    qasm = repo / ".vc4_auto/codegen_m2/candidates/qpu_barrier_syncthreads/kernels/qpu_barrier_syncthreads.qasm"
    harness = repo / "compiler/test/CodeGen/VC4/Hardware/Run/qpu_barrier_syncthreads/candidate/qpu_barrier_syncthreads_candidate_harness.c"

    if not qasm.exists():
        fail(f"missing generated qpu_barrier_syncthreads qasm: {qasm}")
    lines = semantic_qasm_lines(qasm)
    if len(lines) == 3 and "thrend" in lines[0] and lines[1] == "nop" and lines[2] == "nop":
        fail("qpu_barrier_syncthreads generated QASM is the dummy thrend/nop/nop program")

    if not harness.exists():
        fail(f"missing qpu_barrier_syncthreads candidate harness: {harness}")
    text = harness.read_text(encoding="utf-8", errors="replace")
    if re.search(r"if\s*\(\s*rc\s*<\s*0\s*\)[\s\S]{0,500}status=PASS", text):
        fail("qpu_barrier_syncthreads harness can print PASS after rc < 0")
    if "launch_failures" not in text or 'rc < 0 ? "FAIL" : "PASS"' not in text:
        fail("qpu_barrier_syncthreads harness does not make rc < 0 an explicit FAIL")

    print("qpu_barrier_syncthreads generated candidate honesty checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
