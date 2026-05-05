#!/usr/bin/env python3
"""Validate and apply GPT Pro patches for VC4 codegen Milestone 1.

Normal GPT Pro implementation output must be staged by gpt_web_driver.js as:

  response.json
  changes.patch
  .gpt-web-run/*     transport/debug metadata only

This gate is intentionally strict because slice 1 showed that direct source-file
transport through the web UI can mangle C++/JSON/MLIR text.  In strict mode the
patch gate rejects direct source files in staging and requires the changes.patch
GPTWEB block to have used encoding=git-patch-lines.
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
GPTWEB_BLOCK_RE = re.compile(
    r"^BEGIN_GPT_?WEB_FILE([^\r\n]*)\r?\n[\s\S]*?^END_GPT_?WEB_FILE([^\r\n]*)(?:\r?\n|$)",
    re.MULTILINE,
)
KV_RE_TEMPLATE = r"(?:^|\s){key}=(?:\"([^\"]*)\"|'([^']*)'|(\S+))"

ALLOWED_STAGING_FILES = {"response.json", "changes.patch"}
PYTHON_SYNTAX_SUFFIXES = {".py", ".cfg", ".cfg.py"}
MAX_REPORT_TEXT = 50000


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


def _parse_header_kv(header: str, key: str) -> str | None:
    re_key = re.escape(key)
    m = re.search(KV_RE_TEMPLATE.format(key=re_key), header or "")
    if not m:
        return None
    return (m.group(1) if m.group(1) is not None else m.group(2) if m.group(2) is not None else m.group(3) or "").strip()


def _header_path(header: str) -> str | None:
    value = _parse_header_kv(header, "path")
    if value:
        try:
            return normalize_relpath(value)
        except DriverError:
            return None
    # Backward-compatible positional path after BEGIN_GPTWEB_FILE.
    stripped = re.sub(r"(?:^|\s)(?:token|encoding|content_encoding)=\S+", "", header or "").strip()
    if not stripped:
        return None
    try:
        return normalize_relpath(stripped)
    except DriverError:
        return None


def _header_encoding(header: str) -> str:
    enc = _parse_header_kv(header, "encoding") or _parse_header_kv(header, "content_encoding") or ""
    return enc.strip().lower().replace("_", "-")


def staged_non_metadata_files(staging_dir: Path) -> list[str]:
    """Return unexpected direct files staged by GPT output.

    Only response.json, changes.patch, and .gpt-web-run/** metadata are allowed.
    Direct compiler/source files in staging are rejected because that was the
    transport mode that mangled C++ string literals in slice 1.
    """

    bad: list[str] = []
    if not staging_dir.exists():
        raise PatchGateError(f"staging directory does not exist: {staging_dir}")
    for path in sorted(staging_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(staging_dir).as_posix()
        if rel in ALLOWED_STAGING_FILES or rel.startswith(".gpt-web-run/"):
            continue
        bad.append(rel)
    return bad


def load_staged_output(staging_dir: Path) -> tuple[dict[str, Any], Path, str]:
    unexpected = staged_non_metadata_files(staging_dir)
    if unexpected:
        raise PatchGateError(
            "GPT output contained direct files outside response.json/changes.patch. "
            "This is forbidden; source changes must be carried only by changes.patch.\n"
            + "\n".join(unexpected[:200])
        )

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
    if "diff --git " not in patch_text:
        raise PatchGateError("changes.patch does not look like a git unified diff; missing 'diff --git'")
    return response, patch_path, patch_text


def validate_response_json(response: Mapping[str, Any], patch_paths: Sequence[str]) -> None:
    summary = response.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise PatchGateError("response.json must contain a non-empty string field 'summary'")
    changed = response.get("changed_paths")
    if not isinstance(changed, list) or not changed:
        raise PatchGateError("response.json must contain a non-empty changed_paths array matching changes.patch")
    try:
        changed_paths = sorted({normalize_relpath(str(p)) for p in changed})
    except DriverError as exc:
        raise PatchGateError(f"response.json changed_paths contains an invalid path: {exc}") from exc
    patch_set = sorted(set(patch_paths))
    if changed_paths != patch_set:
        raise PatchGateError(
            "response.json changed_paths does not match paths extracted from changes.patch\n"
            f"response.json: {changed_paths}\nchanges.patch: {patch_set}"
        )


def inspect_transport_metadata(staging_dir: Path) -> dict[str, Any]:
    """Inspect GPTWEB transport metadata for encoded changes.patch usage."""

    meta_dir = staging_dir / ".gpt-web-run"
    manifest_path = meta_dir / "manifest.json"
    answer_path = meta_dir / "answer.md"
    encoded_from_manifest: bool | None = None
    encoded_from_answer: bool | None = None
    manifest_note = ""

    if manifest_path.exists():
        try:
            manifest = read_json_file(manifest_path)
            metadata = manifest.get("fileMetadata", []) if isinstance(manifest, dict) else []
            if isinstance(metadata, list):
                for item in metadata:
                    if not isinstance(item, dict):
                        continue
                    rel = item.get("relPath") or item.get("path")
                    if rel == "changes.patch":
                        encoded_from_manifest = bool(item.get("gitPatchLinesDecoded"))
                        break
            if encoded_from_manifest is None and isinstance(manifest, dict):
                # Old drivers did not expose fileMetadata.  Keep going and try answer.md.
                manifest_note = "manifest present but has no fileMetadata for changes.patch"
        except Exception as exc:
            manifest_note = f"could not parse manifest: {exc}"

    if answer_path.exists():
        answer = answer_path.read_text(encoding="utf-8", errors="replace")
        for block in GPTWEB_BLOCK_RE.finditer(answer):
            header = (block.group(1) or "").strip()
            rel = _header_path(header)
            if rel != "changes.patch":
                continue
            encoded_from_answer = _header_encoding(header) in {"git-patch-lines", "patch-lines", "unified-diff-lines"}
            break

    encoded = encoded_from_manifest if encoded_from_manifest is not None else encoded_from_answer
    return {
        "metadata_dir": str(meta_dir),
        "manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "answer_path": str(answer_path) if answer_path.exists() else "",
        "encoded_from_manifest": encoded_from_manifest,
        "encoded_from_answer": encoded_from_answer,
        "encoded_changes_patch": encoded,
        "manifest_note": manifest_note,
    }


def validate_transport_encoding(staging_dir: Path, *, require_encoded_patch: bool) -> dict[str, Any]:
    info = inspect_transport_metadata(staging_dir)
    if not require_encoded_patch:
        info["strict"] = False
        return info
    info["strict"] = True
    encoded = info.get("encoded_changes_patch")
    if encoded is not True:
        raise PatchGateError(
            "changes.patch was not proven to come from a GPTWEB block with encoding=git-patch-lines. "
            "Rejecting to prevent Markdown/HTML/C++ string-literal mangling from recurring. "
            "Use --allow-unencoded-patch only for local hand-written smoke tests.\n"
            + json.dumps(info, indent=2, sort_keys=True)
        )
    return info


def validate_patch_policy(
    *,
    patch_paths: Sequence[str],
    allowed_paths: Sequence[str],
    forbidden_paths: Sequence[str],
) -> None:
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


def _run(repo: Path, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(x) for x in cmd], cwd=str(repo), text=True, capture_output=True)


def _path_exists_in_worktree(repo: Path, rel: str) -> bool:
    try:
        return (repo / normalize_relpath(rel)).exists()
    except DriverError:
        return False


def validate_changed_json(repo: Path, patch_paths: Sequence[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for rel in patch_paths:
        if not rel.endswith(".json") or not _path_exists_in_worktree(repo, rel):
            continue
        path = repo / rel
        try:
            json.loads(path.read_text(encoding="utf-8", errors="replace"))
            checks.append({"path": rel, "ok": True, "kind": "json"})
        except Exception as exc:
            raise PatchGateError(f"changed JSON file is invalid: {rel}: {exc}") from exc
    return checks


def validate_changed_python_syntax(repo: Path, patch_paths: Sequence[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for rel in patch_paths:
        suffixes = Path(rel).suffixes
        is_syntax_file = rel.endswith(".py") or rel.endswith(".cfg") or rel.endswith(".cfg.py")
        if not is_syntax_file or not _path_exists_in_worktree(repo, rel):
            continue
        path = repo / rel
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            compile(text, str(path), "exec")
            checks.append({"path": rel, "ok": True, "kind": "python-syntax"})
        except SyntaxError as exc:
            location = f"line {exc.lineno}" if exc.lineno else "unknown line"
            raise PatchGateError(f"Python/lit config syntax check failed for {rel} at {location}: {exc.msg}") from exc
    return checks


def validate_diff_check(repo: Path, patch_paths: Sequence[str]) -> dict[str, Any]:
    # git diff --check ignores entirely untracked files unless they have intent-to-add,
    # but it still catches whitespace damage in tracked files.  New files also get
    # covered by the syntax/JSON checks below where applicable.
    existing = [p for p in patch_paths if (repo / p).exists()]
    if not existing:
        return {"ok": True, "skipped": True, "reason": "no existing worktree paths for git diff --check"}
    proc = _run(repo, ["git", "diff", "--check", "--", *existing])
    if proc.returncode != 0:
        raise PatchGateError("git diff --check failed after patch apply\n" + proc.stdout + proc.stderr)
    return {"ok": True, "command": ["git", "diff", "--check", "--", *existing]}


def run_candidate_hygiene(repo: Path, patch_paths: Sequence[str]) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "git_diff_check": validate_diff_check(repo, patch_paths),
        "json": validate_changed_json(repo, patch_paths),
        "python_syntax": validate_changed_python_syntax(repo, patch_paths),
    }
    return checks


def _restore_after_failed_apply(repo: Path, patch_paths: Sequence[str], before_paths: Sequence[str]) -> None:
    # Clean up the paths the accepted patch was allowed to touch.  This is used
    # only after an apply/hygiene failure; the autorunner has a broader failed-
    # candidate cleanup after gate failures.
    try:
        restore_paths(repo, patch_paths)
    except Exception:
        after = git_changed_paths(repo, include_untracked=True)
        restore_paths(repo, [p for p in after if p not in set(before_paths)])


def validate_and_apply(
    *,
    repo: Path,
    config: MilestoneConfig,
    slice_entry: Mapping[str, Any],
    staging_dir: Path,
    apply_patch: bool,
    write_report: Path | None = None,
    require_encoded_patch: bool = True,
) -> dict[str, Any]:
    response, patch_path, patch_text = load_staged_output(staging_dir)
    reject_binary_patch(patch_text)
    patch_paths = extract_patch_paths(patch_text)
    validate_response_json(response, patch_paths)
    transport = validate_transport_encoding(staging_dir, require_encoded_patch=require_encoded_patch)
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
    hygiene: dict[str, Any] = {}
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
        try:
            hygiene = run_candidate_hygiene(repo, patch_paths)
        except Exception:
            _restore_after_failed_apply(repo, patch_paths, before_paths)
            raise
    else:
        # In check-only mode, only transport/schema/path/apply checks run.  Syntax
        # hygiene needs the candidate files materialized in the worktree.
        hygiene = {"skipped": "check_only"}

    report = {
        "ok": True,
        "slice_id": slice_entry.get("id"),
        "staging_dir": str(staging_dir),
        "response_summary": response.get("summary") or response.get("diagnosis") or "",
        "patch_path": str(patch_path),
        "changed_paths": patch_paths,
        "transport": transport,
        "hygiene": hygiene,
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
        require_encoded_patch=not args.allow_unencoded_patch,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def cmd_extract_paths(args: argparse.Namespace) -> int:
    text = Path(args.patch).read_text(encoding="utf-8", errors="replace")
    print(json.dumps(extract_patch_paths(text), indent=2))
    return 0


def cmd_guard_worktree(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    slice_entry = config.get_slice(args.slice)
    report = guard_worktree(repo=repo, config=config, slice_entry=slice_entry, restore_disallowed=args.restore_disallowed)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--worklist", default="pro_scripts/vc4_codegen_m1_worklist.json")
    parser.add_argument("--context-profiles", default="pro_scripts/vc4_codegen_m1_context_profiles.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_apply = sub.add_parser("validate-apply", help="validate staged response.json + changes.patch and optionally apply")
    p_apply.add_argument("--slice", required=True)
    p_apply.add_argument("--staging", required=True)
    p_apply.add_argument("--check-only", action="store_true")
    p_apply.add_argument("--report", default="")
    p_apply.add_argument(
        "--allow-unencoded-patch",
        action="store_true",
        help="local-only escape hatch: do not require GPTWEB encoding=git-patch-lines metadata",
    )
    p_apply.set_defaults(func=cmd_validate_apply)

    p_paths = sub.add_parser("extract-paths", help="print changed paths from a patch")
    p_paths.add_argument("patch")
    p_paths.set_defaults(func=cmd_extract_paths)

    p_guard = sub.add_parser("guard-worktree", help="check current worktree paths against a slice policy")
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
