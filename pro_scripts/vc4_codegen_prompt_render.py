#!/usr/bin/env python3
"""Render GPT Pro prompts for VC4 codegen Milestone 1 slices.

This script combines a prompt template with a deterministic context pack.  It is
called by vc4_codegen_m1_autorun.py immediately before invoking the unchanged
pro_scripts/gpt_web_driver.js transport.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from vc4_codegen_context_pack import render_context_pack
    from vc4_codegen_state import DriverError, MilestoneConfig, find_repo_root, read_json_file, relpath, write_json_file
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_context_pack import render_context_pack  # type: ignore
    from vc4_codegen_state import DriverError, MilestoneConfig, find_repo_root, read_json_file, relpath, write_json_file  # type: ignore


TEMPLATE_DIR = Path("pro_scripts/prompts/vc4_codegen_m1")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fenced(text: str, language: str = "") -> str:
    return f"````{language}\n{text.rstrip()}\n````"


def markdown_list(items: Any, *, code: bool = True) -> str:
    if not isinstance(items, list) or not items:
        return "- <none>"
    out = []
    for item in items:
        text = str(item)
        out.append(f"- `{text}`" if code else f"- {text}")
    return "\n".join(out)


def load_template(repo: Path, mode: str) -> tuple[Path, str]:
    mapping = {
        "initial": "gpt_slice_prompt.md.j2",
        "failure": "gpt_failure_prompt.md.j2",
        "diagnosis": "gpt_diagnosis_prompt.md.j2",
    }
    name = mapping.get(mode)
    if not name:
        raise DriverError(f"unknown prompt mode: {mode}")
    path = repo / TEMPLATE_DIR / name
    if not path.exists():
        raise DriverError(f"missing prompt template: {relpath(repo, path)}")
    return path, path.read_text(encoding="utf-8")


def load_optional_file(repo: Path, rel: str) -> str:
    path = repo / rel
    if not path.exists():
        return f"<missing {rel}>"
    return path.read_text(encoding="utf-8", errors="replace")



def hardened_output_contract() -> str:
    return """

## Autorunner hard requirements learned from slice 1 failures

These requirements override any ambiguous wording elsewhere in this prompt.

1. Emit exactly two user-facing GPTWEB files for implementation/fix attempts: `response.json` and `changes.patch`.
2. Do not emit direct compiler/source/test files such as `compiler/.../*.cpp`, `compiler/.../*.mlir`, or `CMakeLists.txt` as GPTWEB files. All repo edits must be represented only inside `changes.patch`.
3. The `changes.patch` GPTWEB file must use `encoding=git-patch-lines` in the BEGIN header:

````text
BEGIN_GPTWEB_FILE token=<token> encoding=git-patch-lines path=changes.patch
R|diff --git a/path b/path
R|--- a/path
R|+++ b/path
R|@@ -1 +1 @@
D|old line without the leading minus
A|new line without the leading plus
END_GPTWEB_FILE token=<token>
````

4. For git-patch-lines records, use exactly these prefixes: `R|` for raw diff metadata, `C|` for context lines, `D|` for deleted lines, and `A|` for added lines. Do not put raw unified-diff lines directly in the GPT response.
5. Entity-shield Markdown/HTML-sensitive payload characters in git-patch-lines records, especially `&`, `<`, and `>`, as `&amp;`, `&lt;`, and `&gt;`. This prevents C++ templates, includes, JSON strings, and string literals from being mangled by the web UI.
6. `response.json.changed_paths` must exactly match the repo-relative paths changed by `changes.patch`.
7. Do not rely on bare tool names in new lit tests unless the patch also wires the tool into lit PATH/substitutions. Prefer `%vc4-codegen`-style substitutions when they exist; otherwise make the CMake/lit PATH wiring explicit.
8. Treat command-not-found, CMake target missing, and lit config syntax issues as mechanical plumbing. Keep fixes narrow and local.
""".rstrip()


def response_schema(mode: str) -> str:
    if mode == "diagnosis":
        schema = {
            "summary": "one-sentence diagnosis summary",
            "classification": "generator|launcher_abi|qasm_syntax|runtime|hardware|test|missing_feature|infra|unknown",
            "recommended_next_action": "patch|add_smaller_test|route_to_codex|abort_for_human|rerun_gate",
            "reasoning_summary": ["concise, user-visible reasons; no private chain-of-thought"],
            "patch_permitted": False,
        }
    else:
        schema = {
            "summary": "one-sentence patch summary",
            "diagnosis": ["concise, user-visible diagnosis bullets"],
            "changed_paths": ["repo/relative/path/from/changes.patch"],
            "tests_to_run": ["deterministic gates or commands expected to pass"],
            "risk_notes": ["known limitations or assumptions, if any"],
        }
    return fenced(json.dumps(schema, indent=2, sort_keys=True), "json")


def failure_packet_json(repo: Path, failure_packet: Path | None) -> str:
    if not failure_packet:
        return "<no failure packet supplied>"
    path = failure_packet if failure_packet.is_absolute() else repo / failure_packet
    if not path.exists():
        return f"<failure packet not found: {path}>"
    try:
        data = read_json_file(path)
    except Exception as exc:
        return f"<failure packet parse error: {exc}>"
    return fenced(json.dumps(data, indent=2, sort_keys=True), "json")


def simple_render(template: str, values: Mapping[str, str]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", value)
    unresolved = []
    import re
    for match in re.finditer(r"\{\{([A-Z0-9_]+)\}\}", out):
        unresolved.append(match.group(1))
    if unresolved:
        raise DriverError(f"template has unresolved placeholders: {sorted(set(unresolved))}")
    return out.rstrip() + "\n"


def render_prompt(
    *,
    repo: Path,
    config: MilestoneConfig,
    slice_id: str,
    attempt: int,
    mode: str,
    failure_packet: Path | None,
    max_chars: int | None = None,
    allow_large_context: bool = False,
    context_out: Path | None = None,
    metadata_out: Path | None = None,
) -> str:
    slice_entry = config.get_slice(slice_id)
    template_path, template = load_template(repo, mode)
    context_pack, context_meta = render_context_pack(
        repo=repo,
        config=config,
        slice_id=slice_id,
        mode=mode,
        failure_packet=failure_packet,
        max_chars_override=max_chars,
        allow_large_context=allow_large_context,
        metadata_out=metadata_out,
    )
    if context_out:
        context_out.parent.mkdir(parents=True, exist_ok=True)
        context_out.write_text(context_pack, encoding="utf-8")
    output_contract = load_optional_file(repo, "pro_scripts/prompts/vc4_codegen_m1/output_contract.md")
    output_contract = output_contract.rstrip() + "\n\n" + hardened_output_contract()
    slice_contract = load_optional_file(repo, "pro_scripts/prompts/vc4_codegen_m1/slice_contract.md")
    constitution = load_optional_file(repo, "pro_scripts/prompts/vc4_codegen_m1/constitution.md")
    codex_contract = load_optional_file(repo, "pro_scripts/prompts/vc4_codegen_m1/codex_contract.md")

    values = {
        "GENERATED_AT_UTC": utc_now(),
        "SLICE_ID": slice_id,
        "SLICE_TITLE": str(slice_entry.get("title", "")),
        "SLICE_INTENT": str(slice_entry.get("intent", "")),
        "ALLOWED_PATHS": markdown_list(config.allowed_paths_for_slice(slice_entry)),
        "FORBIDDEN_PATHS": markdown_list(config.forbidden_paths_for_slice(slice_entry)),
        "GATES": markdown_list(slice_entry.get("gates", []), code=False),
        "NON_GOALS": markdown_list(slice_entry.get("non_goals", []), code=False),
        "ATTEMPT": str(attempt),
        "MODE": mode,
        "TEMPLATE_PATH": relpath(repo, template_path),
        "CONSTITUTION": constitution.rstrip(),
        "OUTPUT_CONTRACT": output_contract.rstrip(),
        "SLICE_CONTRACT": slice_contract.rstrip(),
        "CODEX_CONTRACT": codex_contract.rstrip(),
        "RESPONSE_JSON_SCHEMA": response_schema(mode),
        "FAILURE_PACKET_JSON": failure_packet_json(repo, failure_packet),
        "CONTEXT_PACK": context_pack.rstrip(),
        "CONTEXT_METADATA_JSON": fenced(json.dumps(context_meta, indent=2, sort_keys=True), "json"),
    }
    return simple_render(template, values)


def cmd_render(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    failure_packet = Path(args.failure_packet) if args.failure_packet else None
    out = Path(args.out)
    context_out = Path(args.context_out) if args.context_out else None
    metadata_out = Path(args.metadata_out) if args.metadata_out else None
    text = render_prompt(
        repo=repo,
        config=config,
        slice_id=args.slice,
        attempt=args.attempt,
        mode=args.mode,
        failure_packet=failure_packet,
        max_chars=args.max_chars or None,
        allow_large_context=args.allow_large_context,
        context_out=context_out,
        metadata_out=metadata_out,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out), "chars": len(text), "context_out": str(context_out) if context_out else "", "metadata_out": str(metadata_out) if metadata_out else ""}, indent=2, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--worklist", default="pro_scripts/vc4_codegen_m1_worklist.json")
    parser.add_argument("--context-profiles", default="pro_scripts/vc4_codegen_m1_context_profiles.json")
    parser.add_argument("--slice", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--mode", choices=["initial", "failure", "diagnosis"], default="initial")
    parser.add_argument("--failure-packet", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--context-out", default="")
    parser.add_argument("--metadata-out", default="")
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--allow-large-context", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return cmd_render(args)
    except DriverError as exc:
        print(f"[vc4-prompt] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
