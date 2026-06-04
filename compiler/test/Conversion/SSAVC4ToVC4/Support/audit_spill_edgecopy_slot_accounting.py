#!/usr/bin/env python3
"""Static audit for spill/edge-copy/layout slot accounting boundaries."""

import argparse
import re
import sys
from pathlib import Path


BANNED_SPILL_EDGECOUNT_HELPERS = (
    "getRegfileResultSpacerSlotCount",
    "getSpillSlotBaseSlotCount",
    "getRawVDWStoreSlotCount",
    "getSpillActionSlotCount",
    "getEdgeCopySlotCount",
)

SELF_COUNTING_SEQUENCES = (
    "ResultSpacerSequence",
    "SpillSlotBaseSequence",
    "RawVDWSpillStoreSequence",
    "SpillActionSequence",
    "EdgeCopySequence",
)

DYNAMIC_DMA_MIRROR_PATTERNS = (
    r"\bgetDynamicVDW\w*SlotCount\s*\(",
    r"\bgetStaticVDW\w*SlotCount\s*\(",
    r"\bgetVDWRowByRow\w*SlotCount\s*\(",
    r"\bgetDynamicVDR\w*SlotCount\s*\(",
    r"\bgetDynamicVDRRowByRow\w*SlotCount\s*\(",
)

BRANCH_PAYLOAD_FUNCTION_RE = re.compile(
    r"\b(?:static\s+)?(?:FailureOr<PlannedRegion>|PlannedRegion)\s+"
    r"(planDynamicVDR\w*|planDynamicVDW\w*|planVDRRowByRow\w*|"
    r"planVDW\w*Fallback|planDMARectZeroFill\w*)\s*\([^;{}]*\)\s*\{",
    re.S,
)


def strip_line_comment(line: str) -> str:
    return line.split("//", 1)[0]


def function_body(text: str, match: re.Match) -> str:
    brace_pos = text.rfind("{", match.start(), match.end())
    depth = 0
    for pos in range(brace_pos, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_pos : pos + 1]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    source = args.source
    text = source.read_text()
    code_without_comments = "\n".join(
        strip_line_comment(line) for line in text.splitlines()
    )
    failures = []

    for helper in BANNED_SPILL_EDGECOUNT_HELPERS:
        if re.search(rf"\b{helper}\s*\(", code_without_comments):
            failures.append(f"{source}: legacy manual mirror remains: {helper}")

    for name in SELF_COUNTING_SEQUENCES:
        struct_match = re.search(rf"\bstruct\s+{name}\s*\{{", text)
        if not struct_match:
            failures.append(f"{source}: missing self-counting sequence {name}")
            continue
        body = function_body(text, struct_match)
        if "slotCount()" not in body:
            failures.append(f"{source}: {name} does not expose slotCount()")
        if name != "ResultSpacerSequence" and f"{name}::emit" not in text:
            failures.append(f"{source}: {name} does not own an emit() definition")

    for match in BRANCH_PAYLOAD_FUNCTION_RE.finditer(text):
        body = function_body(text, match)
        for pattern in DYNAMIC_DMA_MIRROR_PATTERNS:
            if re.search(pattern, body):
                failures.append(
                    f"{source}: {match.group(1)} uses dynamic-DMA branch-critical "
                    "manual slot mirror"
                )

    required_markers = (
        "Central flattened layout accounting",
        "VC4 branches carry exactly three hardware delay slots",
        "thread_end emits thrend plus two scheduler hazard slots",
    )
    for marker in required_markers:
        if marker not in text:
            failures.append(f"{source}: missing audit marker: {marker}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print("spill/edge-copy slot accounting audit PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
