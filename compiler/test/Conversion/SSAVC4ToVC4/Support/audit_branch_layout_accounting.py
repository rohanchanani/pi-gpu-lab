#!/usr/bin/env python3
"""Hard audit for branch/layout accounting in SSAVC4ToVC4."""

import argparse
import re
import sys
from pathlib import Path


FIXED_CONSTANTS = (
    "kSingleScheduledSlot",
    "kBranchDelaySlots",
    "kBranchWindowPaddingSlots",
    "kPlannedBranchWindowSlots",
    "kActiveGuardSlots",
    "kThreadEndTrailingNops",
)


def strip_line_comment(line: str) -> str:
    return line.split("//", 1)[0]


def function_body(text: str, name: str) -> str:
    escaped_name = re.escape(name)
    if "::" in name:
        pattern = rf"{escaped_name}\s*\([^;{{}}]*\)\s*(?:const\s*)?\{{"
    else:
        pattern = rf"\b{escaped_name}\s*\([^;{{}}]*\)\s*(?:const\s*)?\{{"
    match = re.search(pattern, text)
    if not match:
        return ""
    brace = text.find("{", match.start())
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : pos + 1]
    return ""


def class_body(text: str, name: str) -> str:
    match = re.search(rf"\bclass\s+{re.escape(name)}\b[^{{]*\{{", text)
    if not match:
        return ""
    brace = text.find("{", match.start())
    depth = 0
    for pos in range(brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : pos + 1]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    source = args.source
    text = source.read_text()
    lines = text.splitlines()
    code = "\n".join(strip_line_comment(line) for line in lines)
    failures = []

    if re.search(r"\bget\w*SlotCount\s*\(", code):
        failures.append(f"{source}: forbidden get*SlotCount helper remains")
    if "FlattenedSlotCount" in code:
        failures.append(f"{source}: forbidden FlattenedSlotCount identifier remains")
    if "flattened layout only" in text:
        failures.append(f"{source}: forbidden flattened-layout accounting comment remains")
    if "legacy-untagged" in text:
        failures.append(f"{source}: forbidden legacy-untagged planned-region append remains")
    if re.search(r"region\.append\(\s*(?:slots|rowSlots|/\*slotCount=)", code):
        failures.append(f"{source}: forbidden manual region.append slot payload remains")
    if re.search(r"\.appendRegion\(\s*(?!\")", code):
        failures.append(f"{source}: untagged appendRegion call remains")
    if ".slotCount(" in code or " slotCount(" in code:
        failures.append(f"{source}: forbidden generic slotCount() sequence remains")

    raw_reload_body = function_body(text, "RawVDRSpillReloadSequence::emit")
    if not raw_reload_body:
        failures.append(f"{source}: could not find RawVDRSpillReloadSequence::emit")
    elif "ldtmu0" in raw_reload_body:
        failures.append(f"{source}: spill reload still uses ldtmu0")

    required_fragments = (
        "countTemplateSlotsByEmission",
        "countTemplateBranchPreludeSlotsByEmission",
        "emitScheduledTemplateBody",
        "countScheduledVC4Slots",
        "createThreadEndSequence",
        "kBranchDelaySlots",
    )
    for fragment in required_fragments:
        if fragment not in text:
            failures.append(f"{source}: missing plan-derived layout fragment {fragment}")

    for constant in FIXED_CONSTANTS:
        if not re.search(rf"\bconstexpr\s+unsigned\s+{constant}\b", code):
            failures.append(f"{source}: missing centralized fixed constant {constant}")

    branch_layout_body = class_body(text, "BranchLayoutPlanner")
    if not branch_layout_body:
        failures.append(f"{source}: could not find BranchLayoutPlanner")
    else:
        if "countTemplateSlotsByEmission" not in branch_layout_body:
            failures.append(
                f"{source}: branch layout does not derive template size by emission"
            )
        if "countTemplateBranchPreludeSlotsByEmission" not in branch_layout_body:
            failures.append(
                f"{source}: branch source slot prelude is not emission-derived"
            )

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print("branch layout accounting audit PASS")
    print("allowed fixed constants: " + ", ".join(FIXED_CONSTANTS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
