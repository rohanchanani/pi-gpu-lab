#!/usr/bin/env python3
"""Validate and apply GPT Pro patches for VC4 codegen Milestone 1.

Normal implementation/fix attempts must stage exactly:

  response.json
  changes.patch

Compiler edits must be represented by changes.patch.  This gate validates the
response metadata, path policy, binary-patch policy, and git-apply state before
any patch is applied to the repository.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from vc4_codegen_state import (
        DriverError,
        MilestoneConfig,
        eprint,
        find_repo_root,
        git_changed_paths,
        match_any_path,
        normalize_relpath,
        read_json_file,
        reject_if_forbidden,
        relpath,
        require_allowed,
        restore_paths,
        write_json_file,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_state import (  # type: ignore
        DriverError,
        MilestoneConfig,
        eprint,
        find_repo_root,
        git_changed_paths,
        match_any_path,
        normalize_relpath,
        read_json_file,
        reject_if_forbidden,
        relpath,
        require_allowed,
        restore_paths,
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
    for marker in ["GIT binary patch", "Binary files ", "literal 0"]:
        if marker in patch_text:
            raise PatchGateError(f"binary patch marker rejected: {marker!r}")


def validate_staging_file_set(staging_dir: Path) -> None:
    allowed = {"response.json", "changes.patch"}
    ignored_prefixes = (".gpt-web-run/", "metadata/", "logs/")
    unexpected: list[str] = []
    for path in staging_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(staging_dir).as_posix()
        if rel in allowed or any(rel.startswith(prefix) for prefix in ignored_prefixes):
            continue
        unexpected.append(rel)
    if unexpected:
        raise PatchGateError(
            "staging directory contains direct GPTWEB files outside response.json + changes.patch; "
            "normal Milestone 1 attempts must transport compiler changes only through changes.patch:\n"
            + "\n".join(sorted(unexpected)[:200])
        )


def load_staged_output(staging_dir: Path) -> tuple[dict[str, Any], Path, str]:
    validate_staging_file_set(staging_dir)
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


def validate_response_contract(response: Mapping[str, Any], *, slice_entry: Mapping[str, Any], patch_paths: Sequence[str]) -> None:
    sid = response.get("slice_id")
    if sid is not None and str(sid) != str(slice_entry.get("id")):
        raise PatchGateError(f"response.json slice_id {sid!r} does not match active slice {slice_entry.get('id')!r}")

    raw_paths = response.get("changed_paths")
    if raw_paths is None:
        raise PatchGateError("response.json must include changed_paths matching changes.patch")
    if not isinstance(raw_paths, list) or not all(isinstance(x, str) for x in raw_paths):
        raise PatchGateError("response.json.changed_paths must be an array of repo-relative strings")
    response_paths = sorted({normalize_relpath(x) for x in raw_paths})
    decoded_paths = sorted({normalize_relpath(x) for x in patch_paths})
    if response_paths != decoded_paths:
        raise PatchGateError(
            "response.json.changed_paths does not match decoded changes.patch paths\n"
            + "response.json changed_paths:\n"
            + "\n".join(response_paths)
            + "\nchanges.patch paths:\n"
            + "\n".join(decoded_paths)
        )
    for key in ("expected_gates", "tests_to_run"):
        if key in response and response[key] is not None and not isinstance(response[key], list):
            raise PatchGateError(f"response.json.{key} must be an array when present")


def validate_patch_policy(*, patch_paths: Sequence[str], allowed_paths: Sequence[str], forbidden_paths: Sequence[str]) -> None:
    if not patch_paths:
        raise PatchGateError("could not extract any changed paths from changes.patch")
    for path in patch_paths:
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
    validate_response_contract(response, slice_entry=slice_entry, patch_paths=patch_paths)
    validate_patch_policy(
        patch_paths=patch_paths,
        allowed_paths=config.allowed_paths_for_slice(slice_entry),
        forbidden_paths=config.forbidden_paths_for_slice(slice_entry),
    )

    check = git_apply_check(repo, patch_path)
    if check.returncode != 0:
        reverse = git_apply_check(repo, patch_path, reverse=True)
        already = reverse.returncode == 0
        message = "git apply --check failed for staged changes.patch"
        if already:
            message += "\npatch already appears applied in reverse-check mode; reset slice state or clean repo before rerunning"
        raise PatchGateError(
            message + "\nstdout:\n"
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
        "response_json": dict(response),
        "response_summary": response.get("summary") or response.get("diagnosis") or "",
        "response_changed_paths": response.get("changed_paths", []),
        "patch_path": str(patch_path),
        "changed_paths": patch_paths,
        "applied": applied,
    }
    if write_report:
        write_json_file(write_report, report)
    return report


def guard_worktree(*, repo: Path, config: MilestoneConfig, slice_entry: Mapping[str, Any], restore_disallowed: bool = False) -> dict[str, Any]:
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
