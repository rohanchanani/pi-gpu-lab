#!/usr/bin/env python3
"""Render GPT Pro prompts for VC4 codegen slices.

This script combines a prompt template with a deterministic context pack. It is
called by vc4_codegen_m1_autorun.py immediately before invoking the unchanged
pro_scripts/gpt_web_driver.js transport.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from vc4_codegen_context_pack import render_context_pack
    from vc4_codegen_state import (
        DriverError,
        MilestoneConfig,
        find_repo_root,
        read_json_file,
        relpath,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_context_pack import render_context_pack  # type: ignore
    from vc4_codegen_state import (  # type: ignore
        DriverError,
        MilestoneConfig,
        find_repo_root,
        read_json_file,
        relpath,
    )


DEFAULT_TEMPLATE_DIR = Path("pro_scripts/prompts/vc4_codegen_m1")
DOWNLOAD_TRANSPORT = "vc4_codegen_download_bundle_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def artifact_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_artifact_component(value: str) -> str:
    out: list[str] = []
    for ch in str(value):
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "-")
    return "".join(out).strip("-._") or "slice"


def artifact_milestone_slug(config: Any, slice_id: str) -> str:
    """Return vc4_codegen_m1 / vc4_codegen_m2 / ... from config or slice id."""
    raw = str(getattr(config, "milestone", "") or "")
    match = re.search(r"(?:^|[^A-Za-z0-9])(m[0-9]+)(?:[^A-Za-z0-9]|$)", raw)
    if not match:
        match = re.match(r"(m[0-9]+)-", str(slice_id))
    milestone = match.group(1) if match else "m"
    return f"vc4_codegen_{milestone}"


def build_download_contract(config: Any, slice_id: str, attempt: int) -> dict[str, Any]:
    prefix = (
        f"{artifact_milestone_slug(config, slice_id)}__{safe_artifact_component(slice_id)}__"
        f"attempt-{attempt:02d}__{artifact_stamp()}"
    )
    return {
        "schema_version": 1,
        "transport": DOWNLOAD_TRANSPORT,
        "artifact_prefix": prefix,
        "bundle_zip": f"{prefix}.zip",
        "apply_script": f"{prefix}.sh",
    }


def fenced(text: str, language: str = "") -> str:
    return f"````{language}\n{text.rstrip()}\n````"


def render_download_contract_markdown(contract: Mapping[str, Any]) -> str:
    """Render a contract in the exact JSON shape parsed by gpt_web_driver.js."""
    final_response = {
        "status": "ok",
        "bundle_zip": str(contract["bundle_zip"]),
        "apply_script": str(contract["apply_script"]),
        "source": "",
    }
    return "\n".join(
        [
            "## Required downloadable artifact transport",
            "",
            f"Transport: `{DOWNLOAD_TRANSPORT}`",
            "",
            "If any earlier template text mentions `response.json`, `changes.patch`, or GPTWEB file blocks for this implementation/failure attempt, ignore that older transport language. Use the downloadable bundle transport below.",
            "",
            f"Artifact prefix: `{contract['artifact_prefix']}`",
            f"Bundle zip filename: `{contract['bundle_zip']}`",
            f"Apply script filename: `{contract['apply_script']}`",
            "",
            "The local browser driver parses the prompt for JSON keys named exactly `bundle_zip` and `apply_script`; keep those keys and filenames byte-for-byte unchanged.",
            "",
            "Machine-readable contract:",
            "",
            fenced(json.dumps(dict(contract), indent=2, sort_keys=True), "json"),
            "",
            "Final visible ChatGPT response must start with this JSON object and then expose two downloadable links/attachments whose visible labels are exactly the two filenames:",
            "",
            fenced(json.dumps(final_response, separators=(",", ":")), "json"),
            "",
            str(contract["bundle_zip"]),
            str(contract["apply_script"]),
        ]
    ) + "\n"


def append_download_contract_to_prompt(rendered: str, contract: Mapping[str, Any], mode: str) -> str:
    if mode == "diagnosis":
        return rendered
    bundle = str(contract["bundle_zip"])
    apply = str(contract["apply_script"])
    if DOWNLOAD_TRANSPORT in rendered and bundle in rendered and apply in rendered:
        return rendered
    return (
        rendered.rstrip()
        + "\n\n## Downloadable artifact contract\n\n"
        + render_download_contract_markdown(contract).rstrip()
        + "\n"
    )


def markdown_list(items: Any, *, code: bool = True) -> str:
    if not isinstance(items, list) or not items:
        return "- <none>"
    out = []
    for item in items:
        text = str(item)
        out.append(f"- `{text}`" if code else f"- {text}")
    return "\n".join(out)


def template_dir_for_config(config: MilestoneConfig) -> Path:
    defaults = config.defaults if isinstance(config.defaults, dict) else {}
    raw = defaults.get("prompt_template_dir")
    if isinstance(raw, str) and raw:
        return Path(raw)
    milestone = str(config.milestone)
    if milestone.endswith("m2") or "m2" in milestone:
        return Path("pro_scripts/prompts/vc4_codegen_m2")
    return DEFAULT_TEMPLATE_DIR


def load_template(repo: Path, mode: str, template_dir: Path) -> tuple[Path, str]:
    mapping = {
        "initial": "gpt_slice_prompt.md.j2",
        "failure": "gpt_failure_prompt.md.j2",
        "diagnosis": "gpt_diagnosis_prompt.md.j2",
    }
    name = mapping.get(mode)
    if not name:
        raise DriverError(f"unknown prompt mode: {mode}")
    path = repo / template_dir / name
    if not path.exists():
        raise DriverError(f"missing prompt template: {relpath(repo, path)}")
    return path, path.read_text(encoding="utf-8")


def load_optional_file(repo: Path, rel: str) -> str:
    path = repo / rel
    if not path.exists():
        return f"<missing {rel}>"
    return path.read_text(encoding="utf-8", errors="replace")


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
            "status": "ok",
            "bundle_zip": "exact zip filename from DOWNLOAD_CONTRACT_JSON",
            "apply_script": "exact shell filename from DOWNLOAD_CONTRACT_JSON",
            "source": "",
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
    template_dir = template_dir_for_config(config)
    template_path, template = load_template(repo, mode, template_dir)
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

    output_contract = load_optional_file(repo, str(template_dir / "output_contract.md"))
    slice_contract = load_optional_file(repo, str(template_dir / "slice_contract.md"))
    constitution = load_optional_file(repo, str(template_dir / "constitution.md"))
    codex_contract = load_optional_file(repo, str(template_dir / "codex_contract.md"))
    download_contract = build_download_contract(config, slice_id, attempt)

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
        "ARTIFACT_PREFIX": str(download_contract["artifact_prefix"]),
        "BUNDLE_ZIP_FILENAME": str(download_contract["bundle_zip"]),
        "APPLY_SCRIPT_FILENAME": str(download_contract["apply_script"]),
        "DOWNLOAD_CONTRACT_JSON": fenced(json.dumps(download_contract, indent=2, sort_keys=True), "json"),
        "DOWNLOAD_CONTRACT_MARKDOWN": render_download_contract_markdown(download_contract).rstrip(),
        "RESPONSE_JSON_SCHEMA": response_schema(mode),
        "FAILURE_PACKET_JSON": failure_packet_json(repo, failure_packet),
        "CONTEXT_PACK": context_pack.rstrip(),
        "CONTEXT_METADATA_JSON": fenced(json.dumps(context_meta, indent=2, sort_keys=True), "json"),
    }
    rendered = simple_render(template, values)
    return append_download_contract_to_prompt(rendered, download_contract, mode)


def cmd_render(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(
        repo,
        worklist_path=args.worklist,
        context_profiles_path=args.context_profiles,
    )
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
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(out),
                "chars": len(text),
                "context_out": str(context_out) if context_out else "",
                "metadata_out": str(metadata_out) if metadata_out else "",
            },
            indent=2,
            sort_keys=True,
        )
    )
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
