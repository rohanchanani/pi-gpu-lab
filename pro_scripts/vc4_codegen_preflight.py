#!/usr/bin/env python3
"""Cheap deterministic preflight checks for VC4 codegen Milestone 1."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

try:
    from vc4_codegen_contracts import build_repo_capability_snapshot, validate_patch_invariants, write_report
    from vc4_codegen_state import DriverError, MilestoneConfig, find_repo_root, git_changed_paths
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_contracts import build_repo_capability_snapshot, validate_patch_invariants, write_report  # type: ignore
    from vc4_codegen_state import DriverError, MilestoneConfig, find_repo_root, git_changed_paths  # type: ignore



def run_repo_capabilities_report(repo: Path, *, out: Path | None = None) -> dict[str, Any]:
    report = build_repo_capability_snapshot(repo, None)
    if out:
        write_report(out, report)
    return report


def _load(args: argparse.Namespace) -> tuple[Path, MilestoneConfig, dict[str, Any] | None]:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    slice_entry = config.get_slice(args.slice) if getattr(args, "slice", "") else None
    return repo, config, slice_entry


def cmd_repo_capabilities(args: argparse.Namespace) -> int:
    repo, _config, slice_entry = _load(args)
    report = build_repo_capability_snapshot(repo, slice_entry)
    if args.report:
        write_report(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_patch_invariants(args: argparse.Namespace) -> int:
    repo, _config, slice_entry = _load(args)
    if slice_entry is None:
        raise DriverError("patch-invariants requires --slice")
    changed = args.changed_path if args.changed_path else git_changed_paths(repo, include_untracked=True)
    report = validate_patch_invariants(repo, slice_entry, changed_paths=changed)
    if args.report:
        write_report(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


def cmd_prompt_templates(args: argparse.Namespace) -> int:
    repo, _config, _slice_entry = _load(args)
    try:
        from vc4_codegen_prompt_render import validate_prompt_templates
    except ModuleNotFoundError:  # pragma: no cover
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vc4_codegen_prompt_render import validate_prompt_templates  # type: ignore
    report = validate_prompt_templates(repo)
    if args.report:
        write_report(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--worklist", default="pro_scripts/vc4_codegen_m1_worklist.json")
    parser.add_argument("--context-profiles", default="pro_scripts/vc4_codegen_m1_context_profiles.json")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("repo-capabilities")
    p.add_argument("--slice", default="")
    p.add_argument("--report", default="")
    p.set_defaults(func=cmd_repo_capabilities)
    q = sub.add_parser("patch-invariants")
    q.add_argument("--slice", required=True)
    q.add_argument("--changed-path", action="append", default=[])
    q.add_argument("--report", default="")
    q.set_defaults(func=cmd_patch_invariants)
    r = sub.add_parser("prompt-templates")
    r.add_argument("--slice", default="")
    r.add_argument("--report", default="")
    r.set_defaults(func=cmd_prompt_templates)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DriverError as exc:
        print(f"[vc4-preflight] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
