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


def require(pattern: str, text: str, message: str, flags: int = 0) -> None:
    if not re.search(pattern, text, flags):
        fail(message)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def initializer_expr(text: str, name: str) -> str:
    match = re.search(rf"\b{name}\b\s*=", text)
    if not match:
        return ""

    expr_start = match.end()
    depth = 0
    in_string = False
    in_char = False
    escape = False
    for index in range(expr_start, len(text)):
        ch = text[index]
        if escape:
            escape = False
            continue
        if in_string or in_char:
            if ch == "\\":
                escape = True
            elif in_string and ch == '"':
                in_string = False
            elif in_char and ch == "'":
                in_char = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "'":
            in_char = True
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            continue
        if ch == ";" and depth == 0:
            return text[expr_start:index]
    return ""


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def has_zero_gate(expr: str, name: str) -> bool:
    patterns = [
        rf"\b{name}\b\s*==\s*0u?\b",
        rf"0u?\b\s*==\s*\b{name}\b",
        rf"!\s*\b{name}\b",
    ]
    return any(re.search(pattern, expr) for pattern in patterns)


def check_harness(text: str) -> None:
    code = strip_comments(text)
    status_expr = initializer_expr(code, "status_pass")
    status_flat = compact(status_expr)

    if re.search(
        r'VC4_TEST_RESULT\s+name=qpu_barrier_syncthreads\s+status=PASS\s+runs=5\b',
        text,
    ):
        fail("qpu_barrier_syncthreads harness has a hard-coded PASS result for five runs")

    if not status_expr:
        fail("qpu_barrier_syncthreads harness does not compute a final status_pass expression")
    if not re.search(r'\bstatus_pass\b\s*\?\s*"PASS"\s*:\s*"FAIL"', code):
        fail("qpu_barrier_syncthreads harness final result is not gated by status_pass")

    for mode in [
        "same_slice",
        "cross_slice_0_1",
        "cross_slice_0_2",
        "full_block",
        "multi_block",
    ]:
        if mode not in text:
            fail(f"qpu_barrier_syncthreads harness does not validate mode {mode}")
    if not re.search(r"\b(?:configs\s*\[\s*5\s*\]|run_count\s*=\s*5u?\b|runs=%d)", code):
        fail("qpu_barrier_syncthreads harness does not account for five barrier runs")

    require(
        r"\b(?:vc4_m2_copy_dtoh|vc4MemcpyDtoH)\s*\(",
        code,
        "qpu_barrier_syncthreads harness does not copy device state back from the GPU",
    )
    require(
        r"\bqpu_barrier_syncthreads_launch\s*\(",
        code,
        "qpu_barrier_syncthreads harness does not launch qpu_barrier_syncthreads_launch",
    )

    if "launch_failures" not in code:
        fail("qpu_barrier_syncthreads harness has no launch_failures accounting")
    if not has_zero_gate(status_flat, "launch_failures"):
        fail("qpu_barrier_syncthreads final status does not require launch_failures == 0")
    require(
        r"\blaunch_run\s*\([^;]*\)\s*<\s*0[\s\S]{0,120}\blaunch_failures\s*(?:\+\+|[+]=\s*1)",
        code,
        "qpu_barrier_syncthreads harness does not increment launch_failures after launch_run failure",
    )

    if not re.search(r"\btimeouts?\b", code):
        fail("qpu_barrier_syncthreads harness has no timeout accounting")
    if not has_zero_gate(status_flat, "timeouts"):
        fail("qpu_barrier_syncthreads final status does not require timeouts == 0")
    require(
        r"if\s*\(\s*rc\s*<\s*0\s*\)[\s\S]{0,240}\btimeouts?\b[\s\S]{0,240}return\s+-?1",
        code,
        "qpu_barrier_syncthreads harness does not record timeout state when launch rc < 0",
    )

    for field in ["qpu_mismatches", "data_mismatches"]:
        if field not in code:
            fail(f"qpu_barrier_syncthreads harness does not compute {field}")
        if not has_zero_gate(status_flat, field):
            fail(f"qpu_barrier_syncthreads final status does not require {field} == 0")
    for label, pattern in {
        "peer masks": r"\b(?:peer_mask|all_seen_mask|always_seen_mask)[A-Za-z0-9_]*\b",
        "observed masks": r"\bobserved_[A-Za-z0-9_]*mask\b",
        "expected masks": r"\bexpected_[A-Za-z0-9_]*mask\b",
        "mismatch counts": r"\bmismatch_count\b",
    }.items():
        require(
            pattern,
            code,
            f"qpu_barrier_syncthreads harness does not analyze device-derived {label}",
        )

    create_match = re.search(r"\bvc4_program_create\s*\(", code)
    if not create_match:
        fail("qpu_barrier_syncthreads harness does not create the VC4 program")
    create_tail = code[create_match.start() : create_match.start() + 900]
    if not (
        re.search(r"if\s*\(\s*rc\s*<\s*0\s*\)[\s\S]{0,700}status=FAIL", create_tail)
        or re.search(r"if\s*\(\s*rc\s*<\s*0\s*\)[\s\S]{0,700}\bstatus_pass\s*=\s*0", create_tail)
    ):
        fail("qpu_barrier_syncthreads program-create failure does not force status=FAIL")


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
    check_harness(text)

    print("qpu_barrier_syncthreads generated candidate honesty checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
