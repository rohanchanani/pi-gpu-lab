#!/usr/bin/env python3
"""Deterministic failure routing for VC4 codegen Milestone 1.

The classifier never asks a model to decide ownership.  It routes only narrow,
recognizable infrastructure/mechanical failures to Codex.  Slice 1 exposed two
important plumbing classes that must not consume another GPT Pro implementation
attempt: lit PATH/tool lookup failures and lit config syntax failures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from vc4_codegen_state import DriverError, MilestoneConfig, find_repo_root, read_json_file, tail_file, write_json_file
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_state import DriverError, MilestoneConfig, find_repo_root, read_json_file, tail_file, write_json_file  # type: ignore


MECHANICAL_COMPILE_RE = re.compile(
    r"(CMake Error|ninja: build stopped|fatal error: .*file not found|fatal error: .*No such file|"
    r"undefined reference to|ld: .*undefined|error: use of undeclared|error: no member named|"
    r"error: unknown type name|error: expected|cannot find -l|no viable conversion|no matching function)",
    re.IGNORECASE,
)
CMAKE_WIRING_RE = re.compile(
    r"(unknown target|target .* not built|cannot find source file|No rule to make target|add_subdirectory|CMake Error)",
    re.IGNORECASE,
)
TOOL_MISSING_RE = re.compile(
    r"(command not found|No such file or directory|unable to find [`']?[^`'\s]+[`']? in PATH|not found in PATH)",
    re.IGNORECASE,
)
LIT_TOOL_PATH_RE = re.compile(
    r"(unable to find [`']?vc4-codegen[`']? in PATH|vc4-codegen: command not found|line \d+: vc4-codegen: command not found|No such file or directory.*vc4-codegen)",
    re.IGNORECASE,
)
LIT_CONFIG_SYNTAX_RE = re.compile(
    r"(unable to parse config file|IndentationError|SyntaxError|TabError|NameError: name 'config' is not defined)",
    re.IGNORECASE,
)
FILECHECK_RE = re.compile(r"(FileCheck|CHECK-|not found in input|possible intended match)", re.IGNORECASE)
COMMAND_NOT_FOUND_FILECHECK_RE = re.compile(
    r"(CHECK: expected string not found[\s\S]{0,1200}(unable to find|command not found|No such file or directory))",
    re.IGNORECASE,
)


class Route:
    GPT = "gpt_pro"
    CODEX = "codex"
    SCRIPT = "script"
    ABORT = "abort"


CODEX_POLICIES_ALLOWING_MECHANICAL = {
    "compile_only_once",
    "compile_only",
    "mechanical_once",
    "mechanical",
    "mechanical_per_category",
    "compile_and_plumbing",
}
CODEX_POLICIES_NEVER = {"never", "never_for_hardware", "none", ""}

# Defaults intentionally override the old global max_codex_attempts=1 behavior
# for independent mechanical classes.  Each listed class gets its own budget.
DEFAULT_CATEGORY_BUDGETS = {
    "mechanical_compile": 1,
    "mechanical_build": 1,
    "mechanical_candidate_build": 1,
    "cmake_wiring": 1,
    "tool_missing": 1,
    "lit_tool_path": 1,
    "lit_config_syntax": 1,
}
SAFE_CODEX_CATEGORIES = set(DEFAULT_CATEGORY_BUDGETS)


def _category_attempts_used(category: str, codex_attempts_used: int, codex_attempts_by_category: Mapping[str, Any] | None) -> int:
    if codex_attempts_by_category is None:
        return codex_attempts_used
    raw = codex_attempts_by_category.get(category, 0)
    try:
        return int(raw)
    except Exception:
        return 0


def _category_budget(slice_entry: Mapping[str, Any], category: str) -> int:
    raw_budgets = slice_entry.get("mechanical_codex_budgets", {})
    if isinstance(raw_budgets, Mapping) and category in raw_budgets:
        try:
            return max(0, int(raw_budgets[category]))
        except Exception:
            return 0
    legacy_max = int(slice_entry.get("max_codex_attempts", 0) or 0)
    default = DEFAULT_CATEGORY_BUDGETS.get(category, 0)
    # Legacy worklists often set max_codex_attempts=1.  Treat that as one per
    # safe category, not one globally, so compile fixes do not consume the lit
    # PATH/syntax repair budget.
    if legacy_max > 0 and default > 0:
        return default
    return default if str(slice_entry.get("codex_policy", "")) in CODEX_POLICIES_ALLOWING_MECHANICAL else 0


def _can_use_codex(
    *,
    slice_entry: Mapping[str, Any],
    policy: str,
    hardware_required: bool,
    category: str,
    codex_attempts_used: int,
    codex_attempts_by_category: Mapping[str, Any] | None,
) -> tuple[bool, int, int, str]:
    if hardware_required:
        return False, 0, 0, "hardware-required slice disables mechanical Codex routing"
    if policy in CODEX_POLICIES_NEVER:
        return False, 0, 0, "codex_policy disables Codex"
    if policy not in CODEX_POLICIES_ALLOWING_MECHANICAL:
        return False, 0, 0, f"codex_policy={policy!r} is not a mechanical policy"
    if category not in SAFE_CODEX_CATEGORIES:
        return False, 0, 0, f"category {category!r} is not safe for Codex"
    used = _category_attempts_used(category, codex_attempts_used, codex_attempts_by_category)
    budget = _category_budget(slice_entry, category)
    if budget <= used:
        return False, used, budget, f"Codex budget exhausted for {category}: used {used}, budget {budget}"
    return True, used, budget, f"Codex budget available for {category}: used {used}, budget {budget}"


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
    legacy_max_codex = int(slice_entry.get("max_codex_attempts", 0) or 0)
    hardware_required = bool(slice_entry.get("hardware_required", False))

    def result(route: str, category: str, reason: str) -> dict[str, Any]:
        used = _category_attempts_used(category, codex_attempts_used, codex_attempts_by_category)
        budget = _category_budget(slice_entry, category)
        return {
            "route": route,
            "category": category,
            "reason": reason,
            "codex_policy": policy,
            "codex_attempts_used": codex_attempts_used,
            "codex_attempts_by_category": dict(codex_attempts_by_category or {}),
            "category_attempts_used": used,
            "category_budget": budget,
            "max_codex_attempts": legacy_max_codex,
            "stage": stage,
            "gate": gate,
        }

    def codex_result(category: str, reason: str) -> dict[str, Any]:
        ok, used, budget, budget_reason = _can_use_codex(
            slice_entry=slice_entry,
            policy=policy,
            hardware_required=hardware_required,
            category=category,
            codex_attempts_used=codex_attempts_used,
            codex_attempts_by_category=codex_attempts_by_category,
        )
        if ok:
            return result(Route.CODEX, category, reason + "; " + budget_reason)
        return result(Route.GPT, category, reason + "; " + budget_reason)

    s = stage.lower()
    g = gate.lower()
    text = log_text or ""

    # Transport/output validation failures are not repo semantics.  Browser
    # transport goes to script/retry; malformed model output goes back to GPT
    # with an output-contract failure packet.
    if s in {"chat", "gpt-web-driver", "browser"}:
        return result(Route.SCRIPT, "chat_transport", "Chat/browser transport failed; retry same prompt or inspect web-driver logs")
    if s in {"gpt-output-validation", "chat-output-validation", "patch-output-validation"}:
        return result(Route.GPT, "malformed_gpt_output", "GPT output did not satisfy response.json + encoded changes.patch contract")
    if s in {"patch-guard", "path-guard", "forbidden-path", "git-apply"}:
        return result(Route.GPT, "patch_policy_or_apply", "Patch failed deterministic path/apply/transport validation")

    # Hardware/qasm/result failures are semantic by default.
    if g.startswith("hardware:") or g.startswith("result:") or hardware_required:
        return result(Route.GPT, "hardware_or_result", "Hardware/result failures require semantic diagnosis")
    if "vc4asm" in g:
        return result(Route.GPT, "qasm_assembler", "vc4asm rejection is qasm syntax/semantics, not a mechanical edit")

    # lit/check-vc4 failures have both semantic and infrastructure subtypes.
    if g.startswith("lit:") or "check-vc4" in g:
        if LIT_CONFIG_SYNTAX_RE.search(text):
            return codex_result("lit_config_syntax", "lit config failed to parse; this is deterministic test plumbing")
        if LIT_TOOL_PATH_RE.search(text) or COMMAND_NOT_FOUND_FILECHECK_RE.search(text):
            return codex_result("lit_tool_path", "lit could not find vc4-codegen; this is PATH/substitution plumbing")
        if FILECHECK_RE.search(text):
            return result(Route.GPT, "lit_filecheck", "lit/FileCheck mismatch without command-not-found evidence is semantic")
        return result(Route.GPT, "lit_failure", "lit failure is semantic/ambiguous by default")

    # Tool missing and CMake wiring failures are mechanical when the policy allows.
    if CMAKE_WIRING_RE.search(text):
        return codex_result("cmake_wiring", "log matches CMake/target wiring failure")
    if TOOL_MISSING_RE.search(text) and (g.startswith("tool:") or g.startswith("build:") or g.startswith("candidate:")):
        return codex_result("tool_missing", "required local tool/target was missing from PATH or expected build location")

    if g == "build:vc4-codegen" or g.startswith("cc:"):
        return codex_result("mechanical_compile", "compile-like gate under mechanical Codex policy")
    if g.startswith("candidate:build:") and MECHANICAL_COMPILE_RE.search(text):
        return codex_result("mechanical_candidate_build", "candidate build log looks like C/CMake/include/linkage issue")
    if g.startswith("build:") and MECHANICAL_COMPILE_RE.search(text):
        return codex_result("mechanical_build", "build log matches conservative compile/build regex")

    return result(Route.GPT, "semantic_or_ambiguous", "Ambiguous failures route to GPT Pro")


def classify_from_packet(packet: Mapping[str, Any], slice_entry: Mapping[str, Any], *, codex_attempts_used: int = 0, codex_attempts_by_category: Mapping[str, Any] | None = None) -> dict[str, Any]:
    log_text = str(packet.get("log_tail", ""))
    return classify_failure(
        slice_entry=slice_entry,
        stage=str(packet.get("stage", "")),
        gate=str(packet.get("gate", packet.get("stage", ""))),
        log_text=log_text,
        codex_attempts_used=codex_attempts_used,
        codex_attempts_by_category=codex_attempts_by_category,
    )


def _json_mapping_arg(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DriverError(f"invalid JSON mapping: {exc}") from exc
    if not isinstance(value, dict):
        raise DriverError("expected a JSON object")
    return value


def cmd_classify(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    slice_entry = config.get_slice(args.slice)
    log_text = ""
    if args.log:
        log_text = Path(args.log).read_text(encoding="utf-8", errors="replace") if Path(args.log).exists() else ""
    route = classify_failure(
        slice_entry=slice_entry,
        stage=args.stage,
        gate=args.gate,
        log_text=log_text,
        codex_attempts_used=args.codex_attempts_used,
        codex_attempts_by_category=_json_mapping_arg(args.codex_attempts_by_category_json),
    )
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
    slice_entry = config.get_slice(slice_id)
    route = classify_from_packet(
        packet,
        slice_entry,
        codex_attempts_used=args.codex_attempts_used,
        codex_attempts_by_category=_json_mapping_arg(args.codex_attempts_by_category_json),
    )
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
    p.add_argument("--codex-attempts-by-category-json", default="")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_classify)

    pp = sub.add_parser("classify-packet", help="classify a failure packet JSON file")
    pp.add_argument("--packet", required=True)
    pp.add_argument("--slice", default="")
    pp.add_argument("--codex-attempts-used", type=int, default=0)
    pp.add_argument("--codex-attempts-by-category-json", default="")
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
