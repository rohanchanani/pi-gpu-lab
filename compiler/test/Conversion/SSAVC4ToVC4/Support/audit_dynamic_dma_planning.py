#!/usr/bin/env python3
"""Static audit for planned dynamic DMA lowering invariants."""

import argparse
import re
import sys
from pathlib import Path


COMMENT_MARKERS = (
    "not branch-target-critical",
    "derived from planned emission",
    "flattened layout only",
)

BRANCH_CRITICAL_HELPERS = (
    r"getDynamicVDW\w*SlotCount",
    r"getStaticVDW\w*SlotCount",
    r"getVDWRowByRow\w*SlotCount",
    r"getDynamicVDR\w*SlotCount",
    r"getDynamicVDRRowByRow\w*SlotCount",
)

MANUAL_DMA_HELPER_RE = re.compile(
    r"\bstatic\s+unsigned\s+"
    r"(get(?:RowOffsetScratch|DMARowAddress|VDRLoadRectDynamic|VDWStoreRectDynamic|"
    r"DynamicVDR\w*|DynamicVDW\w*|StaticVDW\w*|VDWRowByRow\w*)SlotCount)\s*\("
)

FUNCTION_RE = re.compile(r"\b(?:static\s+)?(?:unsigned|LogicalResult|PlannedRegion|FailureOr<PlannedRegion>)\s+(\w+)\s*\(")


def strip_line_comment(line: str) -> str:
    return line.split("//", 1)[0]


def find_enclosing_functions(lines):
    current = None
    brace_depth = 0
    functions = {}
    pending = None
    pending_line = 0
    for idx, line in enumerate(lines, start=1):
        if current is None:
            match = FUNCTION_RE.search(line)
            if match:
                pending = match.group(1)
                pending_line = idx
        if pending and "{" in line:
            current = pending
            functions[idx] = current
            brace_depth = 0
            pending = None
        if current is not None:
            brace_depth += line.count("{") - line.count("}")
            functions[idx] = current
            if brace_depth <= 0:
                current = None
                brace_depth = 0
        elif pending and idx - pending_line > 4:
            pending = None
    return functions


def has_marker_before(lines, line_index):
    start = max(0, line_index - 8)
    window = "\n".join(lines[start:line_index])
    return any(marker in window for marker in COMMENT_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    source = args.source
    text = source.read_text()
    lines = text.splitlines()
    failures = []

    for idx, line in enumerate(lines, start=1):
      code = strip_line_comment(line)
      if "0xffff" in code or "65535" in code:
          failures.append(
              f"{source}:{idx}: forbidden VDW stride mask/limit token in code"
          )

    functions_by_line = find_enclosing_functions(lines)
    helper_res = [re.compile(pattern) for pattern in BRANCH_CRITICAL_HELPERS]
    allowed_flattening_functions = {
        "getVDRLoadRectDynamicSlotCount",
        "getVDWStoreRectDynamicSlotCount",
    }
    for idx, line in enumerate(lines, start=1):
        code = strip_line_comment(line)
        for helper_re in helper_res:
            match = helper_re.search(code)
            if not match:
                continue
            helper = match.group(0)
            if re.search(rf"\bstatic\s+unsigned\s+{helper}\s*\(", code):
                continue
            fn = functions_by_line.get(idx, "")
            if fn in allowed_flattening_functions:
                continue
            failures.append(
                f"{source}:{idx}: {helper} used outside flattened-layout-only accounting"
            )

    for idx, line in enumerate(lines):
        match = MANUAL_DMA_HELPER_RE.search(line)
        if match and "{" not in line:
            continue
        if match and not has_marker_before(lines, idx):
            failures.append(
                f"{source}:{idx + 1}: {match.group(1)} lacks a non-critical comment"
            )

    for idx, line in enumerate(lines, start=1):
        code = strip_line_comment(line)
        if "planStaticRegion" not in code:
            continue
        window = "\n".join(lines[max(0, idx - 8):idx + 8])
        if re.search(r"Dynamic|VDR|VDW|activeRows|activeCols|stride|pitch", window):
            failures.append(
                f"{source}:{idx}: planStaticRegion used near dynamic DMA branch body"
            )

    forbidden_source_tokens = (
        "VC4KernelToVC4",
        "VC4Tile",
        "fixture-name",
        "candidate-name",
        "generated-output",
        "status-string",
    )
    for idx, line in enumerate(lines, start=1):
        code = strip_line_comment(line)
        for token in forbidden_source_tokens:
            if token in code:
                failures.append(f"{source}:{idx}: forbidden source token {token}")

    if failures:
        print("dynamic DMA planning audit FAILED", file=sys.stderr)
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    remaining_helpers = []
    for idx, line in enumerate(lines, start=1):
        match = MANUAL_DMA_HELPER_RE.search(line)
        if match and "{" in line:
            remaining_helpers.append(f"{match.group(1)} at line {idx}")
    print("dynamic DMA planning audit PASS")
    if remaining_helpers:
        print("remaining non-critical manual helpers:")
        for helper in remaining_helpers:
            print(f"  - {helper}")
    else:
        print("remaining non-critical manual helpers: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
