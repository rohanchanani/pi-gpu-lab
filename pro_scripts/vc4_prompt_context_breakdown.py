#!/usr/bin/env python3
"""Print a size breakdown for a rendered VC4 prompt.

Usage:
  python3 pro_scripts/vc4_prompt_context_breakdown.py PATH/attempt-01.md
  python3 pro_scripts/vc4_prompt_context_breakdown.py PATH/attempt-01.md --top 80
  python3 pro_scripts/vc4_prompt_context_breakdown.py PATH/attempt-01.md --json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def extract_context_metadata(text: str) -> dict[str, Any] | None:
    patterns = [
        r"^## Context metadata\s*\n\s*````json\s*\n(.*?)\n````",
        r"^## Context pack metadata\s*\n\s*````json\s*\n(.*?)\n````",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.S | re.M)
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    return None


def heading_sections(text: str) -> list[dict[str, Any]]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.M))
    sections: list[dict[str, Any]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append(
            {
                "title": match.group(1),
                "chars": end - start,
                "source": "heading-fallback",
                "truncated_to_fit": False,
                "omitted_budget": False,
            }
        )
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", type=Path, help="Rendered prompt markdown file")
    parser.add_argument("--top", type=int, default=40, help="Number of largest sections to print")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--all", action="store_true", help="Show all sections")
    args = parser.parse_args()

    text = args.prompt.read_text(encoding="utf-8", errors="replace")
    meta = extract_context_metadata(text) or {}
    sections = list(meta.get("sections") or [])
    source = "context-metadata"
    if not sections:
        sections = heading_sections(text)
        source = "heading-fallback"

    sections = sorted(sections, key=lambda s: int(s.get("chars") or 0), reverse=True)
    total_context = meta.get("chars")
    if total_context is None:
        total_context = sum(int(s.get("chars") or 0) for s in sections)

    output = {
        "prompt": str(args.prompt),
        "prompt_chars": len(text),
        "context_chars": total_context,
        "profile": meta.get("profile") or meta.get("context_profile"),
        "mode": meta.get("mode"),
        "metadata_source": source,
        "section_count": len(sections),
        "sections": sections if args.all else sections[: max(args.top, 0)],
    }

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=False))
        return 0

    print(f"prompt: {output['prompt']}")
    print(f"prompt_chars: {output['prompt_chars']}")
    print(f"context_chars: {output['context_chars']}")
    if output.get("profile"):
        print(f"profile: {output['profile']}")
    if output.get("mode"):
        print(f"mode: {output['mode']}")
    print(f"metadata_source: {output['metadata_source']}")
    print(f"sections: {output['section_count']}")
    print()
    print(f"{'chars':>10}  {'title':<72}  source")
    print(f"{'-' * 10}  {'-' * 72}  {'-' * 20}")
    for section in output["sections"]:
        chars = int(section.get("chars") or 0)
        title = str(section.get("title") or "")[:72]
        src = str(section.get("source") or "")[:120]
        flags = []
        if section.get("truncated_to_fit"):
            flags.append("truncated")
        if section.get("omitted_budget"):
            flags.append("omitted")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        print(f"{chars:10d}  {title:<72}  {src}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
