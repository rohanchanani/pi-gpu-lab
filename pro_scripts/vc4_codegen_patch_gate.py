#!/usr/bin/env python3
"""Validate and apply GPT Pro patches for VC4 codegen Milestone 1.

Normal GPT Pro implementation output is staged by gpt_web_driver.js as two files:

  response.json
  changes.patch

This gate validates the staged response, extracts changed paths from the patch,
checks them against the active slice allow/deny policy, rejects binary patches,
runs `git apply --check`, and only then applies the patch to the repo.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from vc4_codegen_state import (
        DriverError,
        MilestoneConfig,
        StateStore,
        eprint,
        find_repo_root,
        git,
        git_changed_paths,
        match_any_path,
        normalize_relpath,
        read_json_file,
        reject_if_forbidden,
        relpath,
        require_allowed,
        restore_paths,
        tail_text,
        write_json_file,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_state import (  # type: ignore
        DriverError,
        MilestoneConfig,
        StateStore,
        eprint,
        find_repo_root,
        git,
        git_changed_paths,
        match_any_path,
        normalize_relpath,
        read_json_file,
        reject_if_forbidden,
        relpath,
        require_allowed,
        restore_paths,
        tail_text,
        write_json_file,
    )


PATCH_PATH_PREFIX_RE = re.compile(r"^(?:a|b)/(.*)$")
DIFF_GIT_RE = re.compile(r"^diff --git (\S+) (\S+)\s*$")
OLD_NEW_RE = re.compile(r"^(---|\+\+\+)\s+(\S+)")
RENAME_RE = re.compile(r"^(rename from|rename to|copy from|copy to)\s+(.+)$")


class PatchGateError(DriverError):
    pass


def _strip_diff_prefix(raw: str) -> str | None:
    raw = raw.strip().strip('"')
    if raw == "/dev/null":
        return None
    m = PATCH_PATH_PREFIX_RE.match(raw)
    if m:
        raw = m.group(1)
    return normalize_relpath(raw)


def extract_patch_paths(patch_text: str) -> list[str]:
    """Extract changed repo-relative paths from a unified git patch."""

    paths: set[str] = set()
    for line in patch_text.splitlines():
        m = DIFF_GIT_RE.match(line)
        if m:
            for raw in (m.group(1), m.group(2)):
                path = _strip_diff_prefix(raw)
                if path:
                    paths.add(path)
            continue
        m = OLD_NEW_RE.match(line)
        if m:
            path = _strip_diff_prefix(m.group(2))
            if path:
                paths.add(path)
            continue
        m = RENAME_RE.match(line)
        if m:
            path = _strip_diff_prefix(m.group(2))
            if path:
                paths.add(path)
            continue
    return sorted(paths)


def reject_binary_patch(patch_text: str) -> None:
    markers = [
        "GIT binary patch",
        "Binary files ",
        "literal 0",
    ]
    for marker in markers:
        if marker in patch_text:
            raise PatchGateError(f"binary patch marker rejected: {marker!r}")


def load_staged_output(staging_dir: Path) -> tuple[dict[str, Any], Path, str]:
    response_path = staging_dir / "response.json"
    patch_path = staging_dir / "changes.patch"
    if not response_path.exists():
        raise PatchGateError(f"missing staged response.json in {staging_dir}")
    if not patch_path.exists():
        raise PatchGateError(f"missing staged changes.patch in {staging_dir}")
    response = read_json_file(response_path)
    if not isinstance(response, dict):
        raise PatchGateError("response.json must contain a JSON object")
    patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
    if not patch_text.strip():
        raise PatchGateError("changes.patch is empty")
    return response, patch_path, patch_text


def validate_patch_policy(
    *,
    patch_paths: Sequence[str],
    allowed_paths: Sequence[str],
    forbidden_paths: Sequence[str],
) -> None:
    if not patch_paths:
        raise PatchGateError("could not extract any changed paths from changes.patch")
    for path in patch_paths:
        # Normalize first so malformed paths are caught even before policy checks.
        normalized = normalize_relpath(path)
        reject_if_forbidden(normalized, forbidden_paths)
        require_allowed(normalized, allowed_paths)


def git_apply_check(repo: Path, patch_path: Path, *, reverse: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = ["git", "apply", "--check"]
    if reverse:
        cmd.append("--reverse")
    cmd.append(str(patch_path))
    return subprocess.run(cmd, cwd=str(repo), text=True, capture_output=True)


def git_apply(repo: Path, patch_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "apply", str(patch_path)], cwd=str(repo), text=True, capture_output=True)


def validate_and_apply(
    *,
    repo: Path,
    config: MilestoneConfig,
    slice_entry: Mapping[str, Any],
    staging_dir: Path,
    apply_patch: bool,
    write_report: Path | None = None,
) -> dict[str, Any]:
    response, patch_path, patch_text = load_staged_output(staging_dir)
    reject_binary_patch(patch_text)
    patch_paths = extract_patch_paths(patch_text)
    allowed = config.allowed_paths_for_slice(slice_entry)
    forbidden = config.forbidden_paths_for_slice(slice_entry)
    validate_patch_policy(patch_paths=patch_paths, allowed_paths=allowed, forbidden_paths=forbidden)

    check = git_apply_check(repo, patch_path)
    if check.returncode != 0:
        raise PatchGateError(
            "git apply --check failed for staged changes.patch\nstdout:\n"
            + check.stdout
            + "\nstderr:\n"
            + check.stderr
        )

    applied = False
    if apply_patch:
        before_paths = git_changed_paths(repo, include_untracked=True)
        proc = git_apply(repo, patch_path)
        if proc.returncode != 0:
            restore_paths(repo, [p for p in git_changed_paths(repo, include_untracked=True) if p not in before_paths])
            raise PatchGateError(
                "git apply failed unexpectedly after --check passed\nstdout:\n"
                + proc.stdout
                + "\nstderr:\n"
                + proc.stderr
            )
        applied = True

    report = {
        "ok": True,
        "slice_id": slice_entry.get("id"),
        "staging_dir": str(staging_dir),
        "response_summary": response.get("summary") or response.get("diagnosis") or "",
        "patch_path": str(patch_path),
        "changed_paths": patch_paths,
        "applied": applied,
    }
    if write_report:
        write_json_file(write_report, report)
    return report


def guard_worktree(
    *,
    repo: Path,
    config: MilestoneConfig,
    slice_entry: Mapping[str, Any],
    restore_disallowed: bool = False,
) -> dict[str, Any]:
    changed = git_changed_paths(repo, include_untracked=True)
    allowed = config.allowed_paths_for_slice(slice_entry)
    forbidden = config.forbidden_paths_for_slice(slice_entry)
    disallowed: list[str] = []
    forbidden_hits: list[str] = []
    for path in changed:
        if match_any_path(path, forbidden):
            forbidden_hits.append(path)
            continue
        if not match_any_path(path, allowed):
            disallowed.append(path)
    bad = sorted(set(disallowed + forbidden_hits))
    if bad and restore_disallowed:
        restore_paths(repo, bad)
    return {
        "ok": not bad,
        "changed_paths": changed,
        "disallowed_paths": disallowed,
        "forbidden_paths": forbidden_hits,
        "restored": bool(bad and restore_disallowed),
    }


def cmd_validate_apply(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    slice_entry = config.get_slice(args.slice)
    report = validate_and_apply(
        repo=repo,
        config=config,
        slice_entry=slice_entry,
        staging_dir=(repo / args.staging).resolve() if not Path(args.staging).is_absolute() else Path(args.staging).resolve(),
        apply_patch=not args.check_only,
        write_report=(repo / args.report).resolve() if args.report else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_guard_worktree(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    slice_entry = config.get_slice(args.slice)
    report = guard_worktree(repo=repo, config=config, slice_entry=slice_entry, restore_disallowed=args.restore_disallowed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repository root; default: current directory")
    parser.add_argument("--worklist", default="pro_scripts/vc4_codegen_m1_worklist.json")
    parser.add_argument("--context-profiles", default="pro_scripts/vc4_codegen_m1_context_profiles.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_apply = sub.add_parser("validate-apply", help="validate staged GPT output and apply changes.patch")
    p_apply.add_argument("--slice", required=True)
    p_apply.add_argument("--staging", required=True, help="staging directory containing response.json and changes.patch")
    p_apply.add_argument("--check-only", action="store_true", help="validate but do not apply the patch")
    p_apply.add_argument("--report", default="", help="optional JSON report path")
    p_apply.set_defaults(func=cmd_validate_apply)

    p_guard = sub.add_parser("guard-worktree", help="verify current changed paths stay in the slice allowlist")
    p_guard.add_argument("--slice", required=True)
    p_guard.add_argument("--restore-disallowed", action="store_true")
    p_guard.set_defaults(func=cmd_guard_worktree)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DriverError as exc:
        eprint(f"[vc4-patch-gate] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
