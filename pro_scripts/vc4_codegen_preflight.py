#!/usr/bin/env python3
"""Cheap deterministic preflight checks for VC4 codegen automation.

This script is intentionally milestone-generic.  It can be invoked directly for
M1, M2, or future vc4-codegen worklists.  If --worklist/--context-profiles are
not supplied, or if the supplied/default worklist does not contain the requested
slice id, the script resolves the slice by scanning pro_scripts/vc4_codegen_*_worklist.json
and selecting the unique worklist that declares that slice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from vc4_codegen_contracts import (
        build_repo_capability_snapshot,
        validate_patch_invariants,
        write_report,
    )
    from vc4_codegen_state import (
        DriverError,
        MilestoneConfig,
        find_repo_root,
        git_changed_paths,
        relpath,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_contracts import (  # type: ignore
        build_repo_capability_snapshot,
        validate_patch_invariants,
        write_report,
    )
    from vc4_codegen_state import (  # type: ignore
        DriverError,
        MilestoneConfig,
        find_repo_root,
        git_changed_paths,
        relpath,
    )


DEFAULT_WORKLIST = "pro_scripts/vc4_codegen_m1_worklist.json"
DEFAULT_CONTEXT_PROFILES = "pro_scripts/vc4_codegen_m1_context_profiles.json"
ENV_WORKLIST = "VC4_CODEGEN_WORKLIST"
ENV_CONTEXT_PROFILES = "VC4_CODEGEN_CONTEXT_PROFILES"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DriverError(f"failed to read JSON {path}: {exc}") from exc


def _as_repo_path(repo: Path, raw: str | Path) -> Path:
    p = Path(str(raw))
    return p if p.is_absolute() else repo / p


def _slice_ids_in_worklist(path: Path) -> set[str]:
    data = _read_json(path)
    slices = data.get("slices", []) if isinstance(data, dict) else []
    if not isinstance(slices, list):
        return set()
    return {str(s.get("id")) for s in slices if isinstance(s, dict) and s.get("id")}


def _candidate_worklists(repo: Path) -> list[Path]:
    candidates: list[Path] = []

    def add(path: Path) -> None:
        path = path.resolve()
        if path.exists() and path not in candidates:
            candidates.append(path)

    env = os.environ.get(ENV_WORKLIST, "").strip()
    if env:
        add(_as_repo_path(repo, env))
    add(repo / DEFAULT_WORKLIST)
    pro_scripts = repo / "pro_scripts"
    if pro_scripts.exists():
        for path in sorted(pro_scripts.glob("vc4_codegen_*_worklist.json")):
            add(path)
    return candidates


def _paired_context_profiles(repo: Path, worklist_path: Path) -> list[Path]:
    candidates: list[Path] = []

    def add(path: Path) -> None:
        path = path.resolve()
        if path.exists() and path not in candidates:
            candidates.append(path)

    env = os.environ.get(ENV_CONTEXT_PROFILES, "").strip()
    if env:
        add(_as_repo_path(repo, env))

    name = worklist_path.name
    if name.endswith("_worklist.json"):
        add(worklist_path.with_name(name[: -len("_worklist.json")] + "_context_profiles.json"))
    if name == "vc4_codegen_m1_worklist.json":
        add(repo / DEFAULT_CONTEXT_PROFILES)

    # Last-resort fallback: same milestone from JSON, if present.
    try:
        data = _read_json(worklist_path)
        milestone = str(data.get("milestone", "")) if isinstance(data, dict) else ""
        short = milestone.split("-")[-1] if milestone else ""
        if short:
            add(repo / "pro_scripts" / f"vc4_codegen_{short}_context_profiles.json")
    except DriverError:
        pass

    return candidates


def _try_load_config(repo: Path, worklist_path: Path, context_profiles_path: Path) -> MilestoneConfig | None:
    try:
        return MilestoneConfig.load(
            repo,
            worklist_path=worklist_path,
            context_profiles_path=context_profiles_path,
        )
    except DriverError:
        return None


def _load_explicit_or_default(repo: Path, worklist_arg: str, context_arg: str) -> tuple[MilestoneConfig, dict[str, Any]] | None:
    raw_worklist = worklist_arg or os.environ.get(ENV_WORKLIST, "") or DEFAULT_WORKLIST
    raw_context = context_arg or os.environ.get(ENV_CONTEXT_PROFILES, "")
    worklist_path = _as_repo_path(repo, raw_worklist)
    if raw_context:
        context_path = _as_repo_path(repo, raw_context)
    else:
        paired = _paired_context_profiles(repo, worklist_path)
        context_path = paired[0] if paired else repo / DEFAULT_CONTEXT_PROFILES
    config = _try_load_config(repo, worklist_path, context_path)
    if config is None:
        return None
    return config, {
        "mode": "explicit_or_default",
        "worklist": relpath(repo, config.worklist_path),
        "context_profiles": relpath(repo, config.context_profiles_path),
        "fallback_used": False,
    }


def _find_config_for_slice(repo: Path, slice_id: str) -> tuple[MilestoneConfig, dict[str, Any]]:
    matches: list[tuple[MilestoneConfig, dict[str, Any]]] = []
    for worklist_path in _candidate_worklists(repo):
        try:
            if slice_id not in _slice_ids_in_worklist(worklist_path):
                continue
        except DriverError:
            continue
        for context_path in _paired_context_profiles(repo, worklist_path):
            config = _try_load_config(repo, worklist_path, context_path)
            if config is None:
                continue
            try:
                config.get_slice(slice_id)
            except DriverError:
                continue
            matches.append(
                (
                    config,
                    {
                        "mode": "slice_id_scan",
                        "worklist": relpath(repo, config.worklist_path),
                        "context_profiles": relpath(repo, config.context_profiles_path),
                        "fallback_used": True,
                    },
                )
            )
            break

    # De-duplicate by worklist/context pair.
    dedup: dict[tuple[Path, Path], tuple[MilestoneConfig, dict[str, Any]]] = {}
    for config, meta in matches:
        dedup[(config.worklist_path.resolve(), config.context_profiles_path.resolve())] = (config, meta)
    matches = list(dedup.values())

    if not matches:
        known: dict[str, list[str]] = {}
        for worklist_path in _candidate_worklists(repo):
            try:
                ids = sorted(_slice_ids_in_worklist(worklist_path))
            except DriverError:
                ids = []
            known[relpath(repo, worklist_path)] = ids
        raise DriverError(
            f"unknown slice id: {slice_id}; no discovered worklist declares it. "
            f"Pass --worklist/--context-profiles explicitly. Discovered worklists: "
            + json.dumps(known, sort_keys=True)
        )
    if len(matches) > 1:
        choices = [m[1] for m in matches]
        raise DriverError(
            f"ambiguous slice id {slice_id}: declared by multiple worklists. "
            f"Pass --worklist/--context-profiles explicitly. Matches: "
            + json.dumps(choices, sort_keys=True)
        )
    return matches[0]


def _load(args: argparse.Namespace) -> tuple[Path, MilestoneConfig, dict[str, Any] | None, dict[str, Any]]:
    repo = find_repo_root(getattr(args, "repo", "."))
    slice_id = str(getattr(args, "slice", "") or "")
    worklist_arg = str(getattr(args, "worklist", "") or "")
    context_arg = str(getattr(args, "context_profiles", "") or "")

    loaded = _load_explicit_or_default(repo, worklist_arg, context_arg)
    if loaded is not None:
        config, meta = loaded
        if not slice_id:
            return repo, config, None, meta
        try:
            return repo, config, config.get_slice(slice_id), meta
        except DriverError:
            # Fall through to slice-id based discovery.  This is the key
            # milestone-generic behavior: a default M1 worklist must not make an
            # M2/M3 preflight fail before the active worklist can be found.
            pass

    if not slice_id:
        raise DriverError(
            "could not load default worklist/context profiles; pass --worklist and --context-profiles"
        )
    config, meta = _find_config_for_slice(repo, slice_id)
    return repo, config, config.get_slice(slice_id), meta


def _attach_config_meta(report: dict[str, Any], config: MilestoneConfig, meta: Mapping[str, Any]) -> dict[str, Any]:
    report = dict(report)
    report["milestone"] = config.milestone
    report["worklist"] = relpath(config.repo, config.worklist_path)
    report["context_profiles"] = relpath(config.repo, config.context_profiles_path)
    report["preflight_config_resolution"] = dict(meta)
    return report


def run_repo_capabilities_report(repo: Path, *, out: Path | None = None) -> dict[str, Any]:
    report = build_repo_capability_snapshot(repo, None)
    if out:
        write_report(out, report)
    return report


def cmd_repo_capabilities(args: argparse.Namespace) -> int:
    repo, config, slice_entry, meta = _load(args)
    report = _attach_config_meta(build_repo_capability_snapshot(repo, slice_entry), config, meta)
    if args.report:
        write_report(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_patch_invariants(args: argparse.Namespace) -> int:
    repo, config, slice_entry, meta = _load(args)
    if slice_entry is None:
        raise DriverError("patch-invariants requires --slice")
    changed = args.changed_path if args.changed_path else git_changed_paths(repo, include_untracked=True)
    report = validate_patch_invariants(repo, slice_entry, changed_paths=changed)
    report = _attach_config_meta(report, config, meta)
    if args.report:
        write_report(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


def cmd_prompt_templates(args: argparse.Namespace) -> int:
    repo, config, _slice_entry, meta = _load(args)
    try:
        from vc4_codegen_prompt_render import validate_prompt_templates
    except ModuleNotFoundError:  # pragma: no cover
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vc4_codegen_prompt_render import validate_prompt_templates  # type: ignore
    report = _attach_config_meta(validate_prompt_templates(repo), config, meta)
    if args.report:
        write_report(Path(args.report), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


def _add_common_config_args(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument("--repo", default=default if suppress_defaults else ".")
    parser.add_argument("--worklist", default=default if suppress_defaults else "")
    parser.add_argument("--context-profiles", dest="context_profiles", default=default if suppress_defaults else "")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common_config_args(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("repo-capabilities")
    _add_common_config_args(p, suppress_defaults=True)
    p.add_argument("--slice", default="")
    p.add_argument("--report", default="")
    p.set_defaults(func=cmd_repo_capabilities)

    q = sub.add_parser("patch-invariants")
    _add_common_config_args(q, suppress_defaults=True)
    q.add_argument("--slice", required=True)
    q.add_argument("--changed-path", action="append", default=[])
    q.add_argument("--report", default="")
    q.set_defaults(func=cmd_patch_invariants)

    r = sub.add_parser("prompt-templates")
    _add_common_config_args(r, suppress_defaults=True)
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
