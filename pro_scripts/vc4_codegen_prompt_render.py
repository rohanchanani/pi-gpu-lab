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
        "slice_id": slice_id,
        "attempt": int(attempt),
        "artifact_prefix": prefix,
        "bundle_zip": f"{prefix}.zip",
        "apply_script": f"{prefix}.sh",
    }


def fenced(text: str, language: str = "") -> str:
    return f"````{language}\n{text.rstrip()}\n````"


def render_download_contract_markdown(contract: Mapping[str, Any]) -> str:
    """Render the exact downloadable transport contract parsed by the web driver."""
    attempt = int(contract["attempt"])
    slice_id = str(contract["slice_id"])
    final_response = {
        "status": "ok",
        "bundle_zip": str(contract["bundle_zip"]),
        "apply_script": str(contract["apply_script"]),
        "source": "",
    }
    manifest_shape = {
        "schema_version": 1,
        "transport": DOWNLOAD_TRANSPORT,
        "slice_id": slice_id,
        "attempt": attempt,
        "diagnosis": ["one concise user-visible reason for the change"],
        "risk_notes": [],
        "tests_to_run": [],
        "changed_paths": [
            {
                "path": "compiler/lib/Target/VC4/VC4ArtifactEmitter.cpp",
                "action": "write",
                "mode": "0644",
                "sha256": "64 lowercase hex characters for repo/<path>",
            }
        ],
    }
    applier_example = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "REPO_ROOT=\"${1:-${VC4_REPO:-$PWD}}\"\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        f"python3 \"$REPO_ROOT/pro_scripts/vc4_codegen_download_bundle_apply.py\" validate-apply --repo \"$REPO_ROOT\" --slice \"{slice_id}\" --bundle \"$SCRIPT_DIR/{contract['bundle_zip']}\" --apply-script \"$SCRIPT_DIR/{contract['apply_script']}\" --expect-attempt {attempt}\n"
    )
    return "\n".join(
        [
            "## Required downloadable artifact transport",
            "",
            f"Transport: `{DOWNLOAD_TRANSPORT}`",
            "",
            "Use this section as the source of truth. Ignore any previous-attempt artifact filenames, artifact_prefix values, bundle_zip values, apply_script values, or downloadable links shown in failure packets or chat history.",
            "",
            "Hard artifact filename rules:",
            f"- The zip filename MUST be exactly `{contract['bundle_zip']}`.",
            f"- The shell filename MUST be exactly `{contract['apply_script']}`.",
            f"- The zip and manifest MUST say `slice_id = {slice_id}` and `attempt = {attempt}`.",
            "- Do not use `attempt-01` unless this prompt's `Attempt:` line is exactly `1` and the exact filenames below contain `attempt-01`.",
            "- Do not invent a new timestamp or reuse an old timestamp; use the exact filenames below byte-for-byte.",
            "",
            f"Artifact prefix: `{contract['artifact_prefix']}`",
            f"Bundle zip filename: `{contract['bundle_zip']}`",
            f"Apply script filename: `{contract['apply_script']}`",
            "",
            "Machine-readable DOWNLOAD_CONTRACT_JSON:",
            "",
            fenced(json.dumps(dict(contract), indent=2, sort_keys=True), "json"),
            "",
            "The bundle zip layout must be exactly:",
            "",
            "````text\nmanifest.json\nrepo/<repo-relative changed files>\n````",
            "",
            "`manifest.json` must use this shape. `diagnosis`, `risk_notes`, and `tests_to_run` are arrays, not strings:",
            "",
            fenced(json.dumps(manifest_shape, indent=2, sort_keys=True), "json"),
            "",
            "The shell script must be a tiny trusted-applier launcher only. Do not put Python patching logic, heredocs, file writes, unzip/copy logic, or `rm -rf` in it. It should be equivalent to:",
            "",
            fenced(applier_example, "bash"),
            "",
            "Final visible ChatGPT response must start with this exact JSON object and then expose two downloadable links/attachments whose visible labels are exactly the two filenames:",
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


def sanitize_failure_packet_for_prompt(data: Any) -> Any:
    """Return a prompt-safe failure packet without stale artifact filenames.

    Failure packets from rejected downloadable bundles can contain the prior
    attempt's artifact_transport.json, apply shell, and bundle manifest.  GPT
    has copied those stale filenames into later answers.  Keep the useful
    validation facts, but remove exact prior artifact names and large generated
    payloads.
    """
    if isinstance(data, list):
        return [sanitize_failure_packet_for_prompt(x) for x in data]
    if not isinstance(data, dict):
        return data

    out: dict[str, Any] = {}
    for key, value in data.items():
        if key == "apply_bundle_sh":
            out[key] = "<redacted: previous generated apply shell; do not copy; use the current prompt's apply_script filename and trusted-applier launcher>"
            continue
        if key == "artifact_transport_json":
            summary: dict[str, Any] = {
                "redacted": True,
                "reason": "previous attempt artifact metadata contains stale filenames; do not copy",
            }
            if isinstance(value, dict):
                for keep in ("transport", "schema_version", "collected_at"):
                    if keep in value:
                        summary[keep] = value[keep]
                for name_key in ("bundle_zip", "apply_script", "artifact_prefix"):
                    if name_key in value:
                        summary[f"{name_key}_redacted"] = True
            out[key] = summary
            continue
        if key == "bundle_manifest_json":
            summary = {
                "redacted": True,
                "reason": "previous bundle manifest may contain stale attempt IDs and filenames",
            }
            if isinstance(value, dict):
                for keep in ("schema_version", "transport", "slice_id", "attempt"):
                    if keep in value:
                        summary[keep] = value[keep]
                changed = value.get("changed_paths")
                if isinstance(changed, list):
                    summary["changed_paths_count"] = len(changed)
                    summary["changed_paths_preview"] = [
                        (x.get("path") if isinstance(x, dict) else x)
                        for x in changed[:20]
                    ]
                if "diagnosis" in value:
                    summary["diagnosis_type"] = type(value.get("diagnosis")).__name__
                    if not isinstance(value.get("diagnosis"), list):
                        summary["diagnosis_error"] = "diagnosis must be an array of strings"
            out[key] = summary
            continue
        if key == "bundle_zip":
            if isinstance(value, dict):
                out[key] = {"path_redacted": True, "bytes": value.get("bytes")}
            else:
                out[key] = "<redacted: previous bundle zip metadata>"
            continue
        if key == "bundle_zip_members":
            out[key] = "<redacted: previous bundle member list>"
            continue
        out[key] = sanitize_failure_packet_for_prompt(value)

    if data.get("stage") == "patch-guard" or "extra" in data:
        out["artifact_filename_rule"] = (
            "Ignore every artifact filename, artifact_prefix, bundle_zip, and apply_script from this failure packet. "
            "For the next answer, use only the current prompt's DOWNLOAD_CONTRACT_JSON values."
        )
    return out


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
    sanitized = sanitize_failure_packet_for_prompt(data)
    return fenced(json.dumps(sanitized, indent=2, sort_keys=True), "json")


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
        # Context truncation is disabled for this workflow.  Ignore profile
        # budgets and any accidental --max-chars so GPT receives complete
        # selected files/logs instead of reconstructing missing source.
        max_chars_override=None,
        allow_large_context=True,
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
