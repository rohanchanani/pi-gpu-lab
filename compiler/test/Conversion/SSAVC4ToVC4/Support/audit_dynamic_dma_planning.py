#!/usr/bin/env python3
"""Static audit for planned dynamic DMA lowering invariants."""

import argparse
import re
import sys
from pathlib import Path


BRANCH_CRITICAL_HELPERS = (
    r"getDynamicVDW\w*SlotCount",
    r"getStaticVDW\w*SlotCount",
    r"getVDWRowByRow\w*SlotCount",
    r"getDynamicVDR\w*SlotCount",
    r"getDynamicVDRRowByRow\w*SlotCount",
)

MANUAL_DMA_HELPER_RE = re.compile(
    r"\bstatic\s+unsigned\s+"
    r"(get(?:RowOffsetScratch|DMARowAddress|VDRLoadRectDynamic(?:Flattened)?|VDWStoreRectDynamic(?:Flattened)?|"
    r"DynamicVDR\w*|DynamicVDW\w*|StaticVDW\w*|VDWRowByRow\w*)SlotCount)\s*\("
)

FUNCTION_RE = re.compile(
    r"\b(?:static\s+)?"
    r"(?:unsigned|LogicalResult|PlannedRegion|FailureOr<PlannedRegion>)"
    r"\s+(\w+)\s*\([^;{}]*\)\s*\{",
    re.S,
)
BRANCH_PAYLOAD_FUNCTION_RE = re.compile(
    r"^(?:planDynamicVDR\w*|planDynamicVDW\w*|planVDRRowByRow\w*|planVDW\w*Fallback|planDMARectZeroFill\w*)$"
)
FLATTENED_HELPER_RE = re.compile(
    r"\bget(?:VDRLoadRectDynamic|VDWStoreRectDynamic)FlattenedSlotCount\s*\("
)
FORBIDDEN_LEGACY_COMMENTS = (
    "flattened layout only",
    "not branch-target-critical",
)
OLD_DYNAMIC_DMA_DIAGNOSTIC_FRAGMENTS = (
    (
        "dynamic rectangular VDW lowering",
        "currently requires",
        "constant source row",
    ),
    (
        "dynamic rectangular VDR runtime pitch",
        "currently requires",
    ),
    (
        "dynamic rectangular VDR runtime active_rows",
        "currently requires",
        "full static active_cols",
    ),
    (
        "dynamic rectangular VDW runtime active_rows",
        "currently requires",
        "full static active_cols",
    ),
    (
        "dynamic rectangular VDR partial zero-fill",
        "currently supports only one-row rectangles",
    ),
)
VPM_DMA_ROW_16_CAP_PATTERNS = (
    re.compile(r"\b(?:vpmBaseRow|vpmSourceRow)\w*\s*(?:<|<=|>|>=)\s*15\b"),
    re.compile(r"\b(?:vpmBaseRow|vpmSourceRow)\w*\s*&\s*(?:0xf|15)\b", re.I),
    re.compile(r"\b(?:vpmBaseRow|vpmSourceRow)\w*\s*%\s*16\b"),
    re.compile(r"VPM (?:base|source) row.*\[(?:0,\s*)?15\]", re.I),
)
VDR_ROW_BY_ROW_FALLBACK_RE = re.compile(r"\bplanVDRRowByRow\w*Fallback\s*\(")
VDW_FALLBACK_RE = re.compile(r"\bplanVDW\w*Fallback\s*\(")


def strip_line_comment(line: str) -> str:
    return line.split("//", 1)[0]


def find_enclosing_functions(text, lines):
    functions = {}
    line_starts = [0]
    for match in re.finditer(r"\n", text):
        line_starts.append(match.end())

    def line_for_pos(pos):
        lo, hi = 0, len(line_starts)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid
        return lo + 1

    for match in FUNCTION_RE.finditer(text):
        name = match.group(1)
        brace_pos = text.rfind("{", match.start(), match.end())
        if brace_pos < 0:
            continue
        depth = 0
        end_pos = brace_pos
        for pos in range(brace_pos, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    end_pos = pos
                    break
        for line_no in range(line_for_pos(match.start()), line_for_pos(end_pos) + 1):
            functions[line_no] = name
    return functions


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
        for marker in FORBIDDEN_LEGACY_COMMENTS:
            if marker in line:
                failures.append(
                    f"{source}:{idx}: forbidden legacy dynamic-DMA accounting comment {marker!r}"
                )
        if "0xffff" in code or "65535" in code:
            failures.append(
                f"{source}:{idx}: forbidden VDW stride mask/limit token in code"
            )
        for pattern in VPM_DMA_ROW_16_CAP_PATTERNS:
            if pattern.search(code):
                failures.append(
                    f"{source}:{idx}: forbidden 0..15 cap on dynamic VPM DMA row operand"
                )
        if re.search(
            r"\bVDW\b.*constant source row|constant source row.*\bVDW\b",
            code,
        ):
            failures.append(
                f"{source}:{idx}: forbidden VDW constant-source-row limitation"
            )
        if re.search(
            r"VDR.*runtime pitch.*active_rows|runtime pitch.*VDR.*active_rows",
            code,
        ):
            failures.append(
                f"{source}:{idx}: forbidden VDR runtime-pitch active_rows limitation"
            )
        if re.search(
            r"VDR.*partial zero-fill.*one-row|partial zero-fill.*VDR.*one-row",
            code,
        ):
            failures.append(
                f"{source}:{idx}: forbidden one-row VDR partial zero-fill limitation"
            )

    functions_by_line = find_enclosing_functions(text, lines)
    helper_res = [re.compile(pattern) for pattern in BRANCH_CRITICAL_HELPERS]
    for idx, line in enumerate(lines, start=1):
        code = strip_line_comment(line)
        for helper_re in helper_res:
            match = helper_re.search(code)
            if not match:
                continue
            helper = match.group(0)
            if re.search(rf"\bstatic\s+unsigned\s+{helper}\s*\(", code):
                failures.append(
                    f"{source}:{idx}: forbidden branch-critical slot-count mirror {helper}"
                )
                continue
            fn = functions_by_line.get(idx, "")
            failures.append(
                f"{source}:{idx}: forbidden branch-critical slot-count mirror use {helper} in {fn or '<unknown>'}"
            )

    for idx, line in enumerate(lines, start=1):
        code = strip_line_comment(line)
        fn = functions_by_line.get(idx, "")
        if FLATTENED_HELPER_RE.search(code) and fn != "getFlattenedSlotCount":
            if not re.search(r"\bstatic\s+unsigned\s+\w+\s*\(", code):
                failures.append(
                    f"{source}:{idx}: flattened layout helper used outside top-level layout sizing"
                )
        elif FLATTENED_HELPER_RE.search(code):
            failures.append(
                f"{source}:{idx}: forbidden flattened dynamic-DMA slot-count helper"
            )

    for idx, line in enumerate(lines, start=1):
        code = strip_line_comment(line)
        fn = functions_by_line.get(idx, "")
        if not BRANCH_PAYLOAD_FUNCTION_RE.match(fn):
            continue
        if re.search(r"region\.append\s*\(\s*(?:slots|rowSlots|fastPathSlots)", code):
            failures.append(
                f"{source}:{idx}: {fn} uses a manual counted PlannedRegion payload"
            )
        if re.search(r"get\w*SlotCount\s*\(", code):
            failures.append(
                f"{source}:{idx}: {fn} uses a branch-critical slot-count mirror"
            )
        if fn.startswith("planDMARectZeroFill") and re.search(
            r"\b(?:rowSlots|zeroFillSlots|getRowOffsetScratchSlotCount)\b", code
        ):
            failures.append(
                f"{source}:{idx}: {fn} uses legacy zero-fill slot-count accounting"
            )
        if "planStaticRegion" in code:
            window = "\n".join(lines[max(0, idx - 8):idx + 8])
            if re.search(r"Dynamic|VDR|VDW|activeRows|activeCols|stride|pitch", window):
                failures.append(
                    f"{source}:{idx}: {fn} uses planStaticRegion near dynamic DMA payload"
                )

    for idx, line in enumerate(lines, start=1):
        code = strip_line_comment(line)
        fn = functions_by_line.get(idx, "")
        if re.search(r"\bstatic\s+FailureOr<PlannedRegion>\s+planVDRRowByRow", code):
            continue
        if VDR_ROW_BY_ROW_FALLBACK_RE.search(code):
            window = "\n".join(lines[max(0, idx - 25):idx + 25])
            if not re.search(
                r"encodeVDRMemoryPitchBytes|kMaxVDRMPITCHBBytes|planDynamicVDRPitchFallbackDispatch",
                window,
            ):
                failures.append(
                    f"{source}:{idx}: VDR row-by-row fallback lacks pitch overflow/unencodable guard"
                )
        if re.search(r"\bstatic\s+PlannedRegion\s+planVDW\w*Fallback", code):
            continue
        if VDW_FALLBACK_RE.search(code):
            window = "\n".join(lines[max(0, idx - 25):idx + 25])
            if not re.search(
                r"vpmSourceLayoutEncodable|isVDWRectangularVPMSourceLayoutEncodable|strideGap|isEncodableVDWStrideGap|planDynamicVDWStrideGapFallbackDispatch",
                window,
            ):
                failures.append(
                    f"{source}:{idx}: VDW fallback lacks stride-gap or VPM-source-layout guard"
                )

    for idx, line in enumerate(lines):
        match = MANUAL_DMA_HELPER_RE.search(line)
        if match and "{" not in line:
            continue
        if match:
            failures.append(
                f"{source}:{idx + 1}: forbidden manual dynamic-DMA slot-count helper {match.group(1)}"
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

    for match in re.finditer(
        r"\[\[maybe_unused\]\]\s+static\s+void\s+emitDynamic(?:VDR|VDW)\w*",
        text,
    ):
        line_no = text.count("\n", 0, match.start()) + 1
        failures.append(
            f"{source}:{line_no}: unused legacy dynamic DMA raw emitter remains"
        )

    for idx, line in enumerate(lines, start=1):
        for diagnostic in OLD_DYNAMIC_DMA_DIAGNOSTIC_FRAGMENTS:
            if all(fragment in line for fragment in diagnostic):
                failures.append(
                    f"{source}:{idx}: old artificial dynamic DMA diagnostic is present"
                )
        code = strip_line_comment(line)
        fn = functions_by_line.get(idx, "")
        if (
            fn.startswith("plan")
            and "VDW" in fn
            and "nrows" in code
            and "& 0x0f" in code
        ):
            failures.append(
                f"{source}:{idx}: VDW dynamic count path appears to use VDR 0=>16 encoding"
            )

    if "appendSplat32LDISlot(region, loc, 16, 32)" not in text:
        failures.append(
            f"{source}: VDW active-lane clamp must preserve count 16 instead of encoding it as zero"
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
