#!/usr/bin/env python3
"""Deterministic failure routing for VC4 codegen Milestone 1.

The classifier never asks a model to decide ownership.  It uses the failed
stage/gate, the active slice codex_policy, and a few conservative log regexes.
Ambiguous failures route to GPT Pro.  Codex is selected only for mechanical
compile/build classes and only when the slice policy permits it.
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
    r"(CMake Error|ninja: build stopped|No such file or directory|fatal error: .*file not found|"
    r"fatal error: .*No such file|undefined reference to|ld: .*undefined|error: use of undeclared|"
    r"error: no member named|error: unknown type name|error: expected|cannot find -l)",
    re.IGNORECASE,
)

FILECHECK_RE = re.compile(r"(FileCheck|CHECK-|not found in input|possible intended match)", re.IGNORECASE)


class Route:
    GPT = "gpt_pro"
    CODEX = "codex"
    SCRIPT = "script"
    ABORT = "abort"


CODEX_POLICIES_ALLOWING_COMPILE = {"compile_only_once", "compile_only", "mechanical_once"}
CODEX_POLICIES_NEVER = {"never", "never_for_hardware", "none", ""}


def classify_failure(
    *,
    slice_entry: Mapping[str, Any],
    stage: str,
    gate: str = "",
    log_text: str = "",
    codex_attempts_used: int = 0,
) -> dict[str, Any]:
    policy = str(slice_entry.get("codex_policy", "never"))
    max_codex = int(slice_entry.get("max_codex_attempts", 0) or 0)
    hardware_required = bool(slice_entry.get("hardware_required", False))

    def result(route: str, category: str, reason: str) -> dict[str, Any]:
        return {
            "route": route,
            "category": category,
            "reason": reason,
            "codex_policy": policy,
            "codex_attempts_used": codex_attempts_used,
            "max_codex_attempts": max_codex,
            "stage": stage,
            "gate": gate,
        }

    s = stage.lower()
    g = gate.lower()

    # Transport/output validation failures are not repo semantics.  Reuse GPT Pro
    # for malformed model output and script for pure browser/transport failures.
    if s in {"chat", "gpt-web-driver", "browser"}:
        return result(Route.SCRIPT, "chat_transport", "Chat/browser transport failed; retry same prompt or inspect web-driver logs")
    if s in {"gpt-output-validation", "chat-output-validation", "patch-output-validation"}:
        return result(Route.GPT, "malformed_gpt_output", "GPT output did not satisfy response.json + changes.patch contract")
    if s in {"patch-guard", "path-guard", "forbidden-path", "git-apply"}:
        return result(Route.GPT, "patch_policy_or_apply", "Patch failed deterministic path/apply validation")

    # Hardware, qasm, lit, and semantic verifier failures should not go to Codex.
    if g.startswith("hardware:") or g.startswith("result:") or hardware_required:
        return result(Route.GPT, "hardware_or_result", "Hardware/result failures require semantic diagnosis")
    if "vc4asm" in g:
        return result(Route.GPT, "qasm_assembler", "vc4asm rejection is qasm syntax/semantics, not a mechanical edit")
    if g.startswith("lit:"):
        if FILECHECK_RE.search(log_text):
            return result(Route.GPT, "lit_filecheck", "lit/FileCheck failure should be diagnosed semantically unless explicitly known to be a typo")
        return result(Route.GPT, "lit_failure", "lit failure is routed to GPT Pro by default")
    if "verify" in g or "check-vc4" in g:
        return result(Route.GPT, "verifier_or_regression", "VC4 verifier/regression failures are semantic by default")

    # Codex is tightly constrained to mechanical compile/build failures.
    codex_available = policy in CODEX_POLICIES_ALLOWING_COMPILE and max_codex > codex_attempts_used
    if codex_available:
        if g == "build:vc4-codegen" or g.startswith("cc:"):
            return result(Route.CODEX, "mechanical_compile", "Compile-like gate under compile-only Codex policy")
        if g.startswith("candidate:build:") and MECHANICAL_COMPILE_RE.search(log_text):
            return result(Route.CODEX, "mechanical_candidate_build", "Candidate build log looks like C/CMake/include/linkage issue")
        if g.startswith("build:") and MECHANICAL_COMPILE_RE.search(log_text):
            return result(Route.CODEX, "mechanical_build", "Build log matches conservative compile/build regex")

    if policy in CODEX_POLICIES_NEVER or not codex_available:
        return result(Route.GPT, "semantic_or_ambiguous", "Codex is disabled/exhausted or failure is not a safe mechanical class")

    return result(Route.GPT, "semantic_or_ambiguous", "Ambiguous failures route to GPT Pro")


def classify_from_packet(packet: Mapping[str, Any], slice_entry: Mapping[str, Any], *, codex_attempts_used: int = 0) -> dict[str, Any]:
    log_text = str(packet.get("log_tail", ""))
    return classify_failure(
        slice_entry=slice_entry,
        stage=str(packet.get("stage", "")),
        gate=str(packet.get("gate", packet.get("stage", ""))),
        log_text=log_text,
        codex_attempts_used=codex_attempts_used,
    )


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
    route = classify_from_packet(packet, slice_entry, codex_attempts_used=args.codex_attempts_used)
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
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_classify)

    pp = sub.add_parser("classify-packet", help="classify a failure packet JSON file")
    pp.add_argument("--packet", required=True)
    pp.add_argument("--slice", default="")
    pp.add_argument("--codex-attempts-used", type=int, default=0)
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
