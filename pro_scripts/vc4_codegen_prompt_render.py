#!/usr/bin/env python3
"""Render GPT Pro prompts for VC4 codegen slices.

This script combines a prompt template with a deterministic context pack. It is
called by vc4_milestone_autorun.py immediately before invoking the unchanged
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
    from vc4_codegen_contracts import build_repo_capabilities, render_capabilities_markdown
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
    try:
        from vc4_codegen_contracts import build_repo_capabilities, render_capabilities_markdown  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        def build_repo_capabilities(repo: Path) -> dict[str, Any]:  # type: ignore
            return {
                "schema_version": 1,
                "fallback": True,
                "repo": str(repo),
                "compiler_dir_exists": (repo / "compiler").exists(),
                "build_dir_exists": (repo / "compiler/build").exists(),
            }
        def render_capabilities_markdown(capabilities: Mapping[str, Any]) -> str:  # type: ignore
            return fenced(json.dumps(capabilities, indent=2, sort_keys=True), "json")
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
        f"exec python3 \"$REPO_ROOT/pro_scripts/vc4_codegen_download_bundle_apply.py\" validate-apply --repo \"$REPO_ROOT\" --slice \"{slice_id}\" --bundle \"$SCRIPT_DIR/{contract['bundle_zip']}\" --apply-script \"$SCRIPT_DIR/{contract['apply_script']}\" --expect-attempt {attempt}\n"
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
            "The shell script must be a tiny trusted-applier launcher only. It must contain exactly the trusted-applier launcher below except for normal final newline handling. Do not put Python patching logic, heredocs, file writes, unzip/copy logic, chmod/chown logic, backup logic, or `rm -rf` in it. The patch gate rejects oversized or non-launcher apply scripts before any implementation verification runs.",
            "",
            "Exact apply-script content:",
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
    contract_block = "## Downloadable artifact contract\n\n" + render_download_contract_markdown(contract).rstrip() + "\n"
    if DOWNLOAD_TRANSPORT in rendered and bundle in rendered and apply in rendered:
        # The M5 templates now place this section near the top.  For older
        # templates that still render it only after a large context pack, prefix
        # a duplicate copy so exact filenames and trusted launcher rules are not
        # buried beneath tens of thousands of tokens.
        head = rendered[:4000]
        if DOWNLOAD_TRANSPORT in head and bundle in head and apply in head:
            return rendered
        return contract_block + "\n" + rendered.rstrip() + "\n"
    return contract_block + "\n" + rendered.rstrip() + "\n"


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
        if key in {"artifactResult", "artifact_result", "artifact_transport", "downloaded_files", "downloadedFiles", "attempts"}:
            out[key] = "<redacted: previous downloadable artifact result; do not copy; use current DOWNLOAD_CONTRACT_JSON>"
            continue
        if key in {"artifact_prefix", "apply_script", "apply_script_filename", "bundle_zip_filename"}:
            out[key] = "<redacted: stale artifact filename field; use current DOWNLOAD_CONTRACT_JSON>"
            continue
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
    extra = sanitized.get("extra", {}) if isinstance(sanitized, dict) else {}
    typed = extra.get("typed_verifier", {}) if isinstance(extra, dict) else {}
    summary = {
        "failure_packet_path": relpath(repo, path),
        "slice_id": sanitized.get("slice_id") if isinstance(sanitized, dict) else None,
        "stage": sanitized.get("stage") if isinstance(sanitized, dict) else None,
        "message": sanitized.get("message") if isinstance(sanitized, dict) else None,
        "exit_code": sanitized.get("exit_code") if isinstance(sanitized, dict) else None,
        "timed_out": sanitized.get("timed_out") if isinstance(sanitized, dict) else None,
        "top_log_path": sanitized.get("log_path") if isinstance(sanitized, dict) else None,
        "log_tail": sanitized.get("log_tail") if isinstance(sanitized, dict) else "",
        "gate": extra.get("gate") if isinstance(extra, dict) else None,
        "typed_verifier_first_failure": typed.get("first_failure") if isinstance(typed, dict) else None,
        "typed_verifier_failure_count": typed.get("failure_count") if isinstance(typed, dict) else None,
        "typed_verifier_result_count": typed.get("result_count") if isinstance(typed, dict) else None,
        "referenced_log_count": typed.get("referenced_log_count") if isinstance(typed, dict) else None,
        "generated_artifact_count": typed.get("generated_artifact_count") if isinstance(typed, dict) else None,
        "note": "The full sanitized failure packet and any selected full failure logs are included in the context pack below; this header is intentionally compact to avoid duplicate 1MB+ prompt sections.",
    }
    return fenced(json.dumps(summary, indent=2, sort_keys=True), "json")


def failure_packet_summary(repo: Path, failure_packet: Path | None) -> Mapping[str, Any]:
    data = load_failure_packet_data(repo, failure_packet)
    if not isinstance(data, Mapping):
        return {}
    return data


def render_failure_repair_discipline_markdown(
    repo: Path,
    slice_id: str,
    failure_packet: Path | None,
) -> str:
    packet = failure_packet_summary(repo, failure_packet)
    gate = str(packet.get("gate") or "")
    stage = str(packet.get("stage") or "")
    log_tail = str(packet.get("log_tail") or "")
    first_failure = ""
    extra = packet.get("extra")
    if isinstance(extra, Mapping):
        typed = extra.get("typed_verifier")
        if isinstance(typed, Mapping):
            first_failure = str(typed.get("first_failure") or "")

    lines = [
        "A failure attempt is a repair attempt, not a fresh implementation pass.",
        "",
        "- Fix the current failure packet.",
        "- Preserve checks that already passed in the previous attempt; preserve already-passing behavior unless the current failure log explicitly names it.",
        "- Do not rewrite, restyle, or regenerate unrelated files merely because they appear in the context pack.",
        "- If the failure is `source_product_missing`, regenerate the active slice candidate and include every exact source product, but do not rewrite unrelated earlier-slice surfaces.",
        "- If the failure is build or `check-vc4`, inspect the failed test list/log and target only files needed by those failures.",
        "- If a previous Codex mechanical repair fixed compile/API/link/path issues, the next GPT bundle must carry it forward because failed candidate changes are cleaned.",
        "- If the candidate changes files outside active source products or failed-test/log scope, explain why in `manifest.json risk_notes`.",
    ]

    if slice_id.startswith("m4-"):
        lines.extend(
            [
                "- Preserve custom parser/printer definitions when ODS declares custom parse/print.",
                "- Do not set or keep ODS custom parser/printer declarations unless matching C++ definitions are present and linked.",
                "- Do not touch earlier-passed formal-args, program-id, global-store, SAXPY, vector-store, warp-reduce, m4-03 minimal ABI, or m4-09 cooperative matrix surfaces unless the current failure log explicitly names them.",
            ]
        )

    if any(token in (gate + "\n" + stage + "\n" + log_tail + "\n" + first_failure) for token in ("source_product_missing", "check-vc4", "build")):
        lines.append("- Keep the patch boundary aligned with the named gate and its direct logs.")

    return "\n".join(lines).rstrip()


def render_prior_codex_mechanical_repair_markdown(
    repo: Path,
    failure_packet: Path | None,
) -> str:
    packet = failure_packet_summary(repo, failure_packet)
    if not packet:
        return "No prior Codex mechanical repair was detected in the failure packet."

    packet_path = str(
        packet.get("failure_packet_path")
        or (relpath(repo, failure_packet if failure_packet and failure_packet.is_absolute() else repo / failure_packet) if failure_packet else "")
    )
    stage = str(packet.get("stage") or "")
    log_tail = str(packet.get("log_tail") or "")
    message = str(packet.get("message") or "")
    text = "\n".join([stage, message, log_tail])
    lower = text.lower()
    has_codex = stage == "codex" or "codex" in lower or "candidate diff" in lower
    mentions_kernel_parse = "kernelop::parse" in lower
    mentions_kernel_print = "kernelop::print" in lower
    if not (has_codex or mentions_kernel_parse or mentions_kernel_print):
        return "No prior Codex mechanical repair was detected in the failure packet."

    lines = [
        f"Failure packet: `{packet_path or '<current failure packet>'}`",
        "",
        "A previous Codex mechanical repair appears in this failure history. Preserve the mechanical compile/API/link/path fix in the next GPT bundle; do not rely on failed-candidate cleanup to keep it.",
    ]
    if mentions_kernel_parse or mentions_kernel_print:
        lines.append(
            "Carry-forward constraint: if ODS declares custom `KernelOp` parse/print hooks, keep matching linked C++ definitions for `KernelOp::parse` and `KernelOp::print`; do not leave undefined symbols and do not remove only one side of the declaration/definition pair."
        )
    lines.append("Do not auto-replay old diffs; reimplement the necessary mechanical repair deliberately in the active candidate.")
    return "\n".join(lines).rstrip()


def candidate_verification_spec_paths(repo: Path, config: MilestoneConfig) -> list[Path]:
    """Return plausible verification-spec paths for the active milestone.

    Prompt rendering only receives the worklist/context-profile paths, so infer
    the sibling verification spec from the worklist path.  Keep this helper
    defensive so older milestones without source-product prompts still render.
    """
    candidates: list[Path] = []
    for raw in (
        config.defaults.get("verifications") if isinstance(config.defaults, dict) else None,
        config.worklist.get("verifications") if isinstance(config.worklist, dict) else None,
        config.worklist.get("verification_spec") if isinstance(config.worklist, dict) else None,
    ):
        if isinstance(raw, str) and raw:
            path = Path(raw)
            candidates.append(path if path.is_absolute() else repo / path)

    worklist_path = config.worklist_path
    name = worklist_path.name
    if name.endswith("_worklist.json"):
        candidates.append(worklist_path.with_name(name[: -len("_worklist.json")] + "_verifications.json"))
    if "worklist" in name:
        candidates.append(worklist_path.with_name(name.replace("worklist", "verifications")))

    milestone = safe_artifact_component(config.milestone)
    candidates.append(repo / "pro_scripts" / f"{milestone}_verifications.json")

    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def load_verification_spec_for_prompt(repo: Path, config: MilestoneConfig) -> tuple[Path | None, Mapping[str, Any] | None]:
    for path in candidate_verification_spec_paths(repo, config):
        if not path.exists():
            continue
        try:
            data = read_json_file(path)
        except Exception:
            continue
        if isinstance(data, Mapping) and isinstance(data.get("slices"), Mapping):
            return path, data
    return None, None


def collect_source_product_entries(spec: Mapping[str, Any], slice_id: str) -> tuple[list[str], list[str]]:
    slices = spec.get("slices")
    if not isinstance(slices, Mapping):
        return [], []
    slice_spec = slices.get(slice_id)
    if not isinstance(slice_spec, Mapping):
        return [], []
    verifications = slice_spec.get("verifications")
    if not isinstance(verifications, list):
        return [], []

    files: list[str] = []
    globs: list[str] = []
    for verification in verifications:
        if not isinstance(verification, Mapping):
            continue
        if verification.get("mechanism") != "source_products":
            continue
        raw_files = verification.get("files", [])
        if isinstance(raw_files, list):
            files.extend(str(x) for x in raw_files)
        for key in ("globs", "required_globs"):
            raw_globs = verification.get(key, [])
            if isinstance(raw_globs, list):
                globs.extend(str(x) for x in raw_globs)

    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    return dedupe(files), dedupe(globs)


def collect_failure_missing_source_products(value: Any) -> tuple[list[str], list[str]]:
    """Extract missing source-product paths from failure packets/log tails.

    This intentionally collects only repo-looking compiler/pro_scripts paths so
    scheduled-artifact missing intermediates under .vc4_auto do not pollute the
    final source deliverables checklist.
    """
    files: list[str] = []
    globs: list[str] = []

    def keep_path(raw: Any) -> str | None:
        text = str(raw)
        for prefix in ("compiler/", "pro_scripts/"):
            idx = text.find(prefix)
            if idx >= 0:
                return text[idx:]
        return None

    def visit(obj: Any) -> None:
        if isinstance(obj, Mapping):
            for key, child in obj.items():
                if key in {"missing_files", "missing_source_files"} and isinstance(child, list):
                    for item in child:
                        kept = keep_path(item)
                        if kept:
                            files.append(kept)
                    continue
                if key in {"missing_globs", "missing_source_globs"} and isinstance(child, list):
                    for item in child:
                        kept = keep_path(item)
                        if kept:
                            globs.append(kept)
                    continue
                visit(child)
            return
        if isinstance(obj, list):
            for child in obj:
                visit(child)
            return
        if isinstance(obj, str):
            for match in re.finditer(r"missing_files=\[(.*?)\]", obj, flags=re.S):
                for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
                    kept = keep_path(item)
                    if kept:
                        files.append(kept)
            for match in re.finditer(r"missing_globs=\[(.*?)\]", obj, flags=re.S):
                for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
                    kept = keep_path(item)
                    if kept:
                        globs.append(kept)

    visit(value)

    def dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    return dedupe(files), dedupe(globs)


def render_source_product_deliverables(
    repo: Path,
    config: MilestoneConfig,
    slice_id: str,
    failure_packet: Path | None,
) -> str:
    spec_path, spec = load_verification_spec_for_prompt(repo, config)
    if spec is None:
        return ""
    files, globs = collect_source_product_entries(spec, slice_id)
    if not files and not globs:
        return ""

    failure_data = load_failure_packet_data(repo, failure_packet)
    missing_files, missing_globs = collect_failure_missing_source_products(failure_data)

    lines: list[str] = [
        "## Exact source-product deliverables for this slice",
        "",
        "These exact repo-relative source products are part of the active typed verifier contract. If any required path is missing after your patch, the source-products pre-gate will fail before semantic verification starts.",
        "",
        "Do not rename, approximate, or substitute these paths. If this is a failure attempt and the previous candidate was cleaned, regenerate the full slice candidate; do not only add the paths that were missing in the prior attempt.",
        "",
    ]
    if spec_path is not None:
        lines.append(f"Verification spec source: `{relpath(repo, spec_path)}`")
        lines.append("")
    if missing_files or missing_globs:
        lines.append("Previously reported missing source products in the current failure packet:")
        if missing_files:
            lines.append("\nMissing files:")
            lines.extend(f"- `{item}`" for item in missing_files)
        if missing_globs:
            lines.append("\nMissing globs:")
            lines.extend(f"- `{item}`" for item in missing_globs)
        lines.append("")
    if files:
        lines.append("Required files:")
        lines.extend(f"- `{item}`" for item in files)
        lines.append("")
    if globs:
        lines.append("Required globs:")
        lines.extend(f"- `{item}`" for item in globs)
        lines.append("")
    lines.append("Before producing the downloadable bundle, verify that these exact source products either already exist in the repo or are created by your patch. The bundle must still include only real intended source changes; do not create empty placeholder files just to satisfy this checklist.")
    return "\n".join(lines).rstrip() + "\n"


def append_source_product_deliverables_to_prompt(
    rendered: str,
    repo: Path,
    config: MilestoneConfig,
    slice_id: str,
    failure_packet: Path | None,
    mode: str,
) -> str:
    if mode == "diagnosis" or "## Exact source-product deliverables for this slice" in rendered:
        return rendered
    section = render_source_product_deliverables(repo, config, slice_id, failure_packet)
    if not section.strip():
        return rendered
    return rendered.rstrip() + "\n\n" + section.rstrip() + "\n"


TEXT_CONTEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".json",
    ".log",
    ".md",
    ".mlir",
    ".py",
    ".qasm",
    ".qinc",
    ".sh",
    ".txt",
}

TEXT_CONTEXT_BASENAMES = {"Makefile", ".gitignore", "README"}

SLICE_FIXTURE_MATRIX_NAMES = {
    "m2-07-independent-vector-scheduler": ["independent_vector"],
    "m2-08-cooperative-block-scheduler": ["cooperative_shared_reduction"],
    "m2-09-memory-subsystem-matrix": [
        "smoke_runtime",
        "independent_vector",
        "memory_sfu",
        "cooperative_shared_reduction",
    ],
    "m2-10-final-acceptance": ["m2_required"],
}


def is_prompt_text_file(path: Path) -> bool:
    return path.name in TEXT_CONTEXT_BASENAMES or path.suffix in TEXT_CONTEXT_SUFFIXES


def language_for_path(path: Path) -> str:
    suffix = path.suffix
    if suffix in {".c", ".h"}:
        return "c"
    if suffix in {".cc", ".cpp", ".hpp"}:
        return "cpp"
    if suffix == ".json":
        return "json"
    if suffix == ".md":
        return "markdown"
    if suffix == ".mlir":
        return "mlir"
    if suffix == ".py":
        return "python"
    if suffix in {".sh", ".qasm", ".qinc"}:
        return "text"
    return "text"


def render_full_text_file(repo: Path, rel: str, *, heading_prefix: str = "File") -> str:
    path = repo / rel
    if not path.exists() or not path.is_file():
        return f"## {heading_prefix}: {rel}\n_Source: {rel}_\n\n<missing>\n"
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"## {heading_prefix}: {rel}\n_Source: {rel}_\n\n<read error: {exc}>\n"
    return "\n".join(
        [
            f"## {heading_prefix}: {rel}",
            f"_Source: {rel}_",
            "",
            "FULL FILE INCLUDED by workflow hardening. No context/profile max_chars budget was applied.",
            "",
            fenced(body, language_for_path(path)),
            "",
        ]
    )


def load_m2_verification_spec(repo: Path) -> Mapping[str, Any]:
    spec_path = repo / "pro_scripts/vc4_codegen_m2_verifications.json"
    if not spec_path.exists():
        return {}
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def m2_fixture_matrices(repo: Path) -> Mapping[str, list[str]]:
    spec = load_m2_verification_spec(repo)
    raw = spec.get("fixture_matrices")
    if not isinstance(raw, dict):
        raw = spec.get("defaults", {}).get("fixture_matrices") if isinstance(spec.get("defaults"), dict) else {}
    matrices: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for name, fixtures in raw.items():
            if isinstance(name, str) and isinstance(fixtures, list):
                matrices[name] = [str(f) for f in fixtures if isinstance(f, str)]
    return matrices


def fixtures_for_slice(repo: Path, slice_id: str) -> list[str]:
    names = SLICE_FIXTURE_MATRIX_NAMES.get(slice_id, [])
    matrices = m2_fixture_matrices(repo)
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        for fixture in matrices.get(name, []):
            if fixture not in seen:
                seen.add(fixture)
                out.append(fixture)
    return out


def fixture_core_text_paths(repo: Path, fixture: str) -> list[str]:
    root_rel = f"compiler/test/CodeGen/VC4/Hardware/Run/{fixture}"
    root = repo / root_rel
    if not root.exists() or not root.is_dir():
        return []
    paths: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = relpath(repo, path)
        parts = set(path.parts)

        # Never dump build products, generated byproducts, or fixture reference
        # implementation trees into the prompt.  Slice 7 showed that broad
        # fixture_tree expansion can inflate prompts to ~1MB and trigger
        # ChatGPT's fragile pasted-text attachment path.  Keep the source
        # contract complete for each selected core file, but avoid bulk
        # reference/share context unless a human explicitly asks for it.
        if (
            "objs" in parts
            or "Output" in parts
            or ".vc4_auto" in parts
            or "reference" in parts
            or "share" in parts
        ):
            continue
        if path.name.endswith("_harness_host"):
            continue
        if path.suffix in {".bin", ".elf", ".o", ".d"}:
            continue

        # Root fixture contracts plus candidate harness/run inputs are enough
        # to debug matrix failures without guessing input.mlir contents.
        is_root_core = path.parent == root and path.name in {
            "README.md",
            "input.mlir",
            "expected.json",
            "run.sh",
        }
        is_candidate_core = "candidate" in parts and is_prompt_text_file(path)
        if (is_root_core or is_candidate_core) and is_prompt_text_file(path):
            paths.append(rel)
    return paths


def render_fixture_source_context(repo: Path, slice_id: str) -> str:
    fixtures = fixtures_for_slice(repo, slice_id)
    if not fixtures:
        return ""
    lines: list[str] = [
        "# Workflow hardening: full core fixture matrix source context",
        "",
        "This section is added by the workflow renderer only when the selected context profile did not already include fixture input files. It includes each matrix fixture's full input.mlir, expected.json, run.sh, and candidate harness/run files. It intentionally does not expand reference/share trees; do not guess fixture contents, and ask for a specific reference file only if truly needed.",
        "",
        "Fixture matrix members:",
    ]
    lines.extend(f"- `{fixture}`" for fixture in fixtures)
    lines.append("")
    for fixture in fixtures:
        rels = fixture_core_text_paths(repo, fixture)
        lines.append(f"## Fixture source files: {fixture}")
        lines.append("")
        if not rels:
            lines.append("<no text fixture files found>")
            lines.append("")
            continue
        for rel in rels:
            lines.append(render_full_text_file(repo, rel, heading_prefix="Fixture file").rstrip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _json_mapping_looks_failed(value: Mapping[str, Any]) -> bool:
    if value.get("ok") is False or value.get("timed_out") is True:
        return True
    if str(value.get("status", "")).upper() in {"FAIL", "FAILED", "ERROR"}:
        return True
    if "exit_code" in value:
        try:
            return int(value.get("exit_code")) != 0
        except Exception:
            return bool(value.get("exit_code"))
    return False


def collect_failed_log_paths_from_json(value: Any) -> list[str]:
    """Collect log_path values attached to failed phases/results only."""
    found: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        if raw not in seen:
            seen.add(raw)
            found.append(raw)

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            raw = node.get("log_path")
            if isinstance(raw, str) and raw and _json_mapping_looks_failed(node):
                add(raw)
            for sub in node.values():
                visit(sub)
        elif isinstance(node, list):
            for sub in node:
                visit(sub)

    visit(value)
    return found


def packet_embeds_full_log_text(data: Any, repo: Path, path: Path) -> bool:
    """Return true if the failure packet already contains full text for path."""
    wanted = {str(path), relpath(repo, path)}
    try:
        wanted.add(str(path.resolve()))
    except Exception:
        pass

    def visit(node: Any) -> bool:
        if isinstance(node, Mapping):
            p = node.get("path")
            if isinstance(p, str) and p in wanted and node.get("included_full_text") and isinstance(node.get("text"), str):
                return True
            return any(visit(v) for v in node.values())
        if isinstance(node, list):
            return any(visit(x) for x in node)
        return False

    return visit(data)


def load_failure_packet_data(repo: Path, failure_packet: Path | None) -> Any:
    if not failure_packet:
        return None
    path = failure_packet if failure_packet.is_absolute() else repo / failure_packet
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_repo_log_path(repo: Path, raw: str) -> Path | None:
    path = Path(raw)
    if not path.is_absolute():
        path = repo / path
    try:
        path.resolve().relative_to(repo.resolve())
    except Exception:
        return None
    return path


def render_failure_log_context(repo: Path, slice_id: str, failure_packet: Path | None) -> str:
    paths: list[Path] = []
    seen: set[str] = set()

    data = load_failure_packet_data(repo, failure_packet)
    for raw in collect_failed_log_paths_from_json(data):
        path = normalize_repo_log_path(repo, raw)
        if path and path.exists() and path.is_file():
            if packet_embeds_full_log_text(data, repo, path):
                continue
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)

    # Do not glob every historical verifier log for this slice.  Slice 7
    # accumulated many matrix logs across failed attempts; appending all of
    # them made a ~1MB prompt and pushed ChatGPT into the pasted-text
    # attachment transport path.  The current failure packet already names the
    # logs that matter for this attempt, so include those exact files only.

    if not paths:
        return ""

    lines = [
        "# Workflow hardening: full failure logs for this slice",
        "",
        "Failure packets often contain only tails. These full text logs are added so the next attempt can debug the actual generate/assemble/build/hardware/expected-json failures without guessing.",
        "",
    ]
    for path in paths:
        rel = relpath(repo, path)
        lines.append(render_full_text_file(repo, rel, heading_prefix="Failure log").rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def augment_context_pack_for_m2_workflow(
    repo: Path,
    slice_id: str,
    mode: str,
    failure_packet: Path | None,
    context_pack: str,
) -> str:
    extras: list[str] = []
    if slice_id in SLICE_FIXTURE_MATRIX_NAMES:
        fixtures = fixtures_for_slice(repo, slice_id)
        first_fixture = fixtures[0] if fixtures else ""
        first_marker = (
            f"## File: compiler/test/CodeGen/VC4/Hardware/Run/{first_fixture}/input.mlir"
            if first_fixture
            else ""
        )
        if not first_marker or first_marker not in context_pack:
            extras.append(render_fixture_source_context(repo, slice_id))
    if mode == "failure":
        extras.append(render_failure_log_context(repo, slice_id, failure_packet))
    extras = [x.rstrip() for x in extras if x and x.strip()]
    if not extras:
        return context_pack
    return context_pack.rstrip() + "\n\n" + "\n\n".join(extras) + "\n"


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
    context_pack = augment_context_pack_for_m2_workflow(
        repo=repo,
        slice_id=slice_id,
        mode=mode,
        failure_packet=failure_packet,
        context_pack=context_pack,
    )
    if context_out:
        context_out.parent.mkdir(parents=True, exist_ok=True)
        context_out.write_text(context_pack, encoding="utf-8")

    output_contract = load_optional_file(repo, str(template_dir / "output_contract.md"))
    slice_contract = load_optional_file(repo, str(template_dir / "slice_contract.md"))
    constitution = load_optional_file(repo, str(template_dir / "constitution.md"))
    codex_contract = load_optional_file(repo, str(template_dir / "codex_contract.md"))
    download_contract = build_download_contract(config, slice_id, attempt)
    repo_capabilities = build_repo_capabilities(repo)
    repo_capabilities_md = render_capabilities_markdown(repo_capabilities)
    repo_capabilities_json = fenced(json.dumps(repo_capabilities, indent=2, sort_keys=True), "json")

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
        "FAILURE_REPAIR_DISCIPLINE": render_failure_repair_discipline_markdown(repo, slice_id, failure_packet),
        "PRIOR_CODEX_MECHANICAL_REPAIR": render_prior_codex_mechanical_repair_markdown(repo, failure_packet),
        "REPO_CAPABILITY_SNAPSHOT": repo_capabilities_json,
        "REPO_CAPABILITIES": repo_capabilities_md,
        "REPO_CAPABILITIES_MARKDOWN": repo_capabilities_md,
        "REPO_CAPABILITIES_JSON": repo_capabilities_json,
        "NO_TRUNCATION_CONTEXT_POLICY": "Context truncation is disabled: all selected files/logs/artifacts are included in full. Do not reconstruct omitted code; if something is missing from context, explicitly ask for it.",
        "DOWNLOAD_CONTRACT": render_download_contract_markdown(download_contract).rstrip(),
        "DOWNLOAD_ARTIFACT_CONTRACT": render_download_contract_markdown(download_contract).rstrip(),
        "DOWNLOAD_CONTRACT_JSON_RAW": json.dumps(download_contract, indent=2, sort_keys=True),
        "CONTEXT_PACK": context_pack.rstrip(),
        "CONTEXT_METADATA_JSON": fenced(json.dumps(context_meta, indent=2, sort_keys=True), "json"),
    }
    rendered = simple_render(template, values)
    rendered = append_source_product_deliverables_to_prompt(
        rendered, repo, config, slice_id, failure_packet, mode
    )
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
