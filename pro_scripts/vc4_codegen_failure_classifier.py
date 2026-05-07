#!/usr/bin/env python3
"""Deterministic failure routing for VC4 codegen Milestone 1.

Classify by normalized failure signature first, not by broad gate name.  Codex
is available only for narrow mechanical classes and only within per-category
budgets.  Semantic, hardware, qasm, verifier, and ordinary FileCheck failures
route to GPT Pro.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from vc4_codegen_state import DriverError, MilestoneConfig, find_repo_root, read_json_file, write_json_file
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_state import DriverError, MilestoneConfig, find_repo_root, read_json_file, write_json_file  # type: ignore


class Route:
    GPT = "gpt_pro"
    CODEX = "codex"
    SCRIPT = "script"
    ABORT = "abort"


CODEX_POLICIES_ALLOWING_MECHANICAL = {"compile_only_once", "compile_only", "mechanical_once", "mechanical_by_category"}
CODEX_POLICIES_NEVER = {"never", "never_for_hardware", "none", ""}

MECHANICAL_COMPILE_RE = re.compile(
    r"(CMake Error|ninja: build stopped|fatal error: .*file not found|fatal error: .*No such file|"
    r"undefined reference to|ld: .*undefined|error: use of undeclared|error: no member named|"
    r"error: unknown type name|error: expected|cannot find -l|no rule to make target|unknown target)",
    re.IGNORECASE,
)
LIT_CONFIG_SYNTAX_RE = re.compile(r"(fatal: unable to parse config file|IndentationError|SyntaxError|NameError: name 'config'|lit\.local\.cfg)", re.IGNORECASE)
LIT_TOOL_RESOLUTION_RE = re.compile(r"(unable to find [`']?%?[-A-Za-z0-9_+.]+[`']? in PATH|command not found|fg: no job control|No such file or directory)", re.IGNORECASE)
FILECHECK_RE = re.compile(r"(FileCheck|CHECK-|expected string not found|not found in input|possible intended match)", re.IGNORECASE)
PATCH_ALREADY_APPLIED_RE = re.compile(r"(patch already appears applied|--reverse.*would apply|already exists in working directory|patch does not apply)", re.IGNORECASE)
TRANSPORT_RE = re.compile(r"(Could not find ChatGPT prompt box|web-driver|browser|timeout|playwright|CDP|conversation URL)", re.IGNORECASE)
ARTIFACT_RE = re.compile(r"(downloadable artifact|artifact_transport|bundle\.zip|apply_bundle|vc4_codegen_download_bundle_v1|bundle manifest|sha256 mismatch|missing downloadable)", re.IGNORECASE)

DEFAULT_MECHANICAL_BUDGETS = {
    "mechanical_compile": 1,
    "mechanical_cmake_or_target": 1,
    "mechanical_test_invocation": 1,
    "mechanical_lit_config": 1,
    "mechanical_candidate_build": 1,
}


def _decode_first_json_object(text: str) -> Mapping[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, ch in enumerate(text or ""):
        if ch != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def _typed_verifier_first_failure(log_text: str) -> Mapping[str, Any] | None:
    report = _decode_first_json_object(log_text)
    if not isinstance(report, Mapping):
        return None
    failures = report.get("failures")
    if isinstance(failures, list) and failures and isinstance(failures[0], Mapping):
        return failures[0]
    # Failure packets may embed the summarized verifier report under extra.
    extra = report.get("extra")
    if isinstance(extra, Mapping):
        typed = extra.get("typed_verifier")
        if isinstance(typed, Mapping) and isinstance(typed.get("first_failure"), Mapping):
            return typed["first_failure"]
    return None


def _typed_verifier_category(gate: str, log_text: str) -> tuple[str, str] | None:
    if not gate.lower().startswith("typed-verifier:"):
        return None
    failure = _typed_verifier_first_failure(log_text)
    if not failure:
        return "typed_verifier", "Typed verifier failed but no structured failure packet was found"
    mechanism = str(failure.get("mechanism", ""))
    vid = str(failure.get("verification_id", ""))
    message = str(failure.get("message", ""))
    reason = f"Typed verifier failed at {vid or '<unknown>'} ({mechanism or '<unknown>'}): {message}"

    # Keep semantic/product checks on GPT Pro, but allow narrowly mechanical
    # verifier mechanisms to use existing Codex budgets.  This preserves the
    # original routing principle: FileCheck/content, qasm semantics, hardware,
    # and source-product holes go to GPT; compiler/toolchain plumbing can go to
    # Codex once.
    if mechanism == "build":
        return "mechanical_compile", reason
    if mechanism == "c_syntax":
        return "mechanical_compile", reason
    if mechanism == "tool_available":
        return "mechanical_test_invocation", reason
    if mechanism == "command" and LIT_TOOL_RESOLUTION_RE.search(json.dumps(failure)):
        return "mechanical_test_invocation", reason
    if mechanism == "lit":
        blob = json.dumps(failure)
        if LIT_CONFIG_SYNTAX_RE.search(blob):
            return "mechanical_lit_config", reason
        if LIT_TOOL_RESOLUTION_RE.search(blob):
            return "mechanical_test_invocation", reason
        if FILECHECK_RE.search(blob):
            return "lit_filecheck", reason
        return "lit_failure", reason
    if mechanism == "vc4asm_assemble":
        return "qasm_assembler", reason
    if mechanism == "candidate_phase":
        return "candidate_build_semantic", reason
    if mechanism == "hardware_run" or mechanism == "expected_json_result":
        return "hardware_or_result", reason
    if mechanism == "reference_immutable":
        return "patch_policy_or_apply", reason
    return "typed_verifier", reason


def _category_attempts_used(category: str, *, codex_attempts_used: int, codex_attempts_by_category: Mapping[str, Any] | None) -> int:
    if codex_attempts_by_category is None:
        return int(codex_attempts_used)
    try:
        return int(codex_attempts_by_category.get(category, 0) or 0)
    except Exception:
        return int(codex_attempts_used)


def _category_budget(category: str, slice_entry: Mapping[str, Any], *, max_codex: int) -> int:
    raw = slice_entry.get("mechanical_budgets", {})
    if isinstance(raw, Mapping) and category in raw:
        try:
            return int(raw.get(category, 0) or 0)
        except Exception:
            return 0
    if category in DEFAULT_MECHANICAL_BUDGETS:
        return min(max_codex, DEFAULT_MECHANICAL_BUDGETS[category]) if max_codex > 0 else 0
    return 0


def _is_mechanical_category(category: str) -> bool:
    return category.startswith("mechanical_")


def _normalized_category(*, stage: str, gate: str, log_text: str, hardware_required: bool) -> tuple[str, str]:
    s = stage.lower()
    g = gate.lower()
    log = log_text or ""

    typed_category = _typed_verifier_category(gate, log)
    if typed_category is not None:
        return typed_category

    if s in {"chat", "gpt-web-driver", "browser"} or TRANSPORT_RE.search(log) and "chat" in s:
        return "chat_transport", "Chat/browser transport failed; retry same prompt or inspect web-driver logs"
    if ARTIFACT_RE.search(log):
        return "artifact_transport", "Downloadable bundle transport or bundle manifest validation failed"
    if s in {"gpt-output-validation", "chat-output-validation", "patch-output-validation"}:
        return "malformed_gpt_output", "GPT output did not satisfy the required staged output contract"
    if s in {"patch-guard", "path-guard", "forbidden-path", "git-apply"}:
        if PATCH_ALREADY_APPLIED_RE.search(log):
            return "repo_state_or_idempotence", "Patch could not apply cleanly and may already be applied or based on stale repo state"
        return "patch_policy_or_apply", "Patch failed deterministic path/apply validation"
    if s.startswith("preflight") or g.startswith("preflight:"):
        if LIT_CONFIG_SYNTAX_RE.search(log):
            return "mechanical_lit_config", "Changed lit config has a mechanical syntax/config issue"
        if "lit_tool_resolution" in log or LIT_TOOL_RESOLUTION_RE.search(log):
            return "mechanical_test_invocation", "Changed test invocation/tool-resolution invariant failed"
        return "patch_preflight_invariant", "Patch failed deterministic repository invariant checks"

    if g.startswith("hardware:") or g.startswith("result:") or hardware_required:
        return "hardware_or_result", "Hardware/result failures require semantic diagnosis"
    if "vc4asm" in g:
        return "qasm_assembler", "vc4asm rejection is qasm syntax/semantics, not a mechanical edit"
    if g.startswith("candidate:build:"):
        if MECHANICAL_COMPILE_RE.search(log) or LIT_TOOL_RESOLUTION_RE.search(log):
            return "mechanical_candidate_build", "Candidate build log looks like C/CMake/include/linkage/tool issue"
        return "candidate_build_semantic", "Candidate build failure is not a safe mechanical class"
    if g.startswith("cc:"):
        return "mechanical_compile", "C syntax-only gate is a narrow mechanical compile check"
    if g.startswith("build:"):
        if g == "build:check-vc4":
            if LIT_CONFIG_SYNTAX_RE.search(log):
                return "mechanical_lit_config", "lit config parse/syntax failure is mechanical test plumbing"
            if LIT_TOOL_RESOLUTION_RE.search(log):
                return "mechanical_test_invocation", "lit command/substitution/tool resolution failure is mechanical test plumbing"
            if FILECHECK_RE.search(log):
                return "lit_filecheck", "lit/FileCheck content mismatch is semantic"
            return "verifier_or_regression", "VC4 regression failure is semantic by default"
        if MECHANICAL_COMPILE_RE.search(log) or g in {"build:vc4-codegen", "build:vc4-opt"}:
            return "mechanical_compile", "Compile-like gate under mechanical Codex policy"
        return "mechanical_build", "Build-like gate failed"
    if g.startswith("lit:"):
        if LIT_CONFIG_SYNTAX_RE.search(log):
            return "mechanical_lit_config", "lit config parse/syntax failure is mechanical test plumbing"
        if LIT_TOOL_RESOLUTION_RE.search(log):
            return "mechanical_test_invocation", "lit command/substitution/tool resolution failure is mechanical test plumbing"
        if FILECHECK_RE.search(log):
            return "lit_filecheck", "lit/FileCheck content mismatch is semantic"
        return "lit_failure", "lit failure is routed to GPT Pro by default"
    if "verify" in g:
        return "verifier_or_regression", "VC4 verifier failures are semantic by default"

    return "semantic_or_ambiguous", "Ambiguous failures route to GPT Pro"


def classify_failure(
    *,
    slice_entry: Mapping[str, Any],
    stage: str,
    gate: str = "",
    log_text: str = "",
    codex_attempts_used: int = 0,
    codex_attempts_by_category: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    policy = str(slice_entry.get("codex_policy", "never"))
    max_codex = int(slice_entry.get("max_codex_attempts", 0) or 0)
    hardware_required = bool(slice_entry.get("hardware_required", False))
    category, reason = _normalized_category(stage=stage, gate=gate, log_text=log_text, hardware_required=hardware_required)
    used_for_category = _category_attempts_used(category, codex_attempts_used=codex_attempts_used, codex_attempts_by_category=codex_attempts_by_category)
    budget_for_category = _category_budget(category, slice_entry, max_codex=max_codex)

    route = Route.GPT
    if category == "chat_transport":
        route = Route.SCRIPT
    elif _is_mechanical_category(category) and policy in CODEX_POLICIES_ALLOWING_MECHANICAL and budget_for_category > used_for_category:
        route = Route.CODEX
    elif policy in CODEX_POLICIES_NEVER or not _is_mechanical_category(category):
        route = Route.GPT

    return {
        "route": route,
        "category": category,
        "reason": reason,
        "codex_policy": policy,
        "codex_attempts_used": int(codex_attempts_used),
        "codex_attempts_by_category": dict(codex_attempts_by_category or {}),
        "category_attempts_used": used_for_category,
        "category_budget": budget_for_category,
        "max_codex_attempts": max_codex,
        "stage": stage,
        "gate": gate,
    }


def classify_from_packet(packet: Mapping[str, Any], slice_entry: Mapping[str, Any], *, codex_attempts_used: int = 0, codex_attempts_by_category: Mapping[str, Any] | None = None) -> dict[str, Any]:
    log_text = str(packet.get("log_tail", ""))
    if not log_text and isinstance(packet.get("extra"), Mapping):
        log_text = json.dumps(packet.get("extra"), sort_keys=True)
    return classify_failure(
        slice_entry=slice_entry,
        stage=str(packet.get("stage", "")),
        gate=str(packet.get("gate", packet.get("stage", ""))),
        log_text=log_text,
        codex_attempts_used=codex_attempts_used,
        codex_attempts_by_category=codex_attempts_by_category,
    )


def cmd_classify(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    slice_entry = config.get_slice(args.slice)
    log_text = Path(args.log).read_text(encoding="utf-8", errors="replace") if args.log and Path(args.log).exists() else ""
    by_cat = json.loads(args.codex_attempts_by_category) if args.codex_attempts_by_category else None
    route = classify_failure(slice_entry=slice_entry, stage=args.stage, gate=args.gate, log_text=log_text, codex_attempts_used=args.codex_attempts_used, codex_attempts_by_category=by_cat)
    print(json.dumps(route, indent=2, sort_keys=True))
    if args.out:
        write_json_file(Path(args.out), route)
    return 0


def cmd_classify_packet(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    packet = read_json_file(Path(args.packet))
    if not isinstance(packet, dict):
        raise DriverError("failure packet must be a JSON object")
    slice_id = args.slice or str(packet.get("slice_id", ""))
    if not slice_id:
        raise DriverError("slice id missing; pass --slice")
    by_cat = json.loads(args.codex_attempts_by_category) if args.codex_attempts_by_category else None
    route = classify_from_packet(packet, config.get_slice(slice_id), codex_attempts_used=args.codex_attempts_used, codex_attempts_by_category=by_cat)
    print(json.dumps(route, indent=2, sort_keys=True))
    if args.out:
        write_json_file(Path(args.out), route)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--worklist", default="pro_scripts/vc4_codegen_m1_worklist.json")
    parser.add_argument("--context-profiles", default="pro_scripts/vc4_codegen_m1_context_profiles.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("classify", help="classify a stage/gate/log")
    p.add_argument("--slice", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--gate", default="")
    p.add_argument("--log", default="")
    p.add_argument("--codex-attempts-used", type=int, default=0)
    p.add_argument("--codex-attempts-by-category", default="")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_classify)

    pp = sub.add_parser("classify-packet", help="classify a failure packet JSON file")
    pp.add_argument("--packet", required=True)
    pp.add_argument("--slice", default="")
    pp.add_argument("--codex-attempts-used", type=int, default=0)
    pp.add_argument("--codex-attempts-by-category", default="")
    pp.add_argument("--out", default="")
    pp.set_defaults(func=cmd_classify_packet)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DriverError as exc:
        print(f"[vc4-classifier] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
