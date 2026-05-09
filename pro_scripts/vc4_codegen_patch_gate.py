#!/usr/bin/env python3
"""Validate and apply GPT Pro patches for VC4 codegen Milestone 1.

Normal implementation/fix attempts should use the downloadable bundle transport:

  artifact_transport.json
  bundle.zip
  apply_bundle.sh

The legacy response.json + changes.patch transport is still accepted as a
fallback for manual recovery and old prompts.  In both transports this gate
validates metadata, path policy, forbidden paths, and repository state before
any candidate change is applied.
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
    from vc4_codegen_download_bundle_apply import BundleApplyError, apply_bundle
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
    from vc4_codegen_download_bundle_apply import BundleApplyError, apply_bundle  # type: ignore
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


GENERATED_TEST_SIDE_EFFECT_PATTERNS = (
    "compiler/test/**/Output/**",
    "compiler/test/**/.lit_test_times.txt",
)


def is_generated_test_side_effect_path(path: str) -> bool:
    return match_any_path(path, GENERATED_TEST_SIDE_EFFECT_PATTERNS)


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


def _ignored_staging_file(rel: str) -> bool:
    ignored_prefixes = (".gpt-web-run/", "metadata/", "logs/", "downloads/")
    return any(rel.startswith(prefix) for prefix in ignored_prefixes)


def detect_staging_transport(staging_dir: Path) -> str:
    if (staging_dir / "artifact_transport.json").exists() or (staging_dir / "bundle.zip").exists():
        return "download_bundle"
    if (staging_dir / "response.json").exists() or (staging_dir / "changes.patch").exists():
        return "legacy_patch"
    return "unknown"


def expected_attempt_from_staging_dir(staging_dir: Path) -> int | None:
    for part in reversed(staging_dir.parts):
        m = re.fullmatch(r"attempt-(\d+)", part)
        if m:
            return int(m.group(1))
    return None


def expected_download_name_fragment(slice_entry: Mapping[str, Any], expected_attempt: int | None) -> str | None:
    if expected_attempt is None:
        return None
    slice_id = str(slice_entry.get("id", ""))
    if not slice_id:
        return None
    milestone = "m"
    m = re.match(r"(m\d+)-", slice_id)
    if m:
        milestone = m.group(1)
    return f"vc4_codegen_{milestone}__{slice_id}__attempt-{expected_attempt:02d}__"


def validate_download_transport_metadata(
    transport: Mapping[str, Any],
    *,
    slice_entry: Mapping[str, Any],
    expected_attempt: int | None,
) -> None:
    fragment = expected_download_name_fragment(slice_entry, expected_attempt)
    if fragment is None:
        return
    stale: list[str] = []
    for key in ("artifact_prefix", "bundle_zip", "apply_script"):
        value = transport.get(key)
        if isinstance(value, str) and value and fragment not in value:
            stale.append(f"{key}={value}")
    attempts = transport.get("attempts")
    if isinstance(attempts, list):
        for item in attempts:
            if isinstance(item, dict):
                filename = item.get("filename")
                if isinstance(filename, str) and filename and fragment not in filename:
                    stale.append(f"attempts[].filename={filename}")
    downloaded = transport.get("downloaded_files")
    if isinstance(downloaded, dict):
        for key in ("bundle_zip", "apply_script"):
            value = downloaded.get(key)
            if isinstance(value, str) and value:
                name = Path(value).name
                if name not in {"bundle.zip", "apply_bundle.sh"} and fragment not in name:
                    stale.append(f"downloaded_files.{key}={value}")
    if stale:
        raise PatchGateError(
            "download-bundle artifact metadata uses stale or wrong attempt filenames. "
            f"Expected all generated artifact names to contain {fragment!r}. "
            "The next GPT attempt must ignore prior failure-packet artifact filenames and use only the current prompt's DOWNLOAD_CONTRACT_JSON.\n"
            + "\n".join(stale[:20])
        )


def validate_staging_file_set(staging_dir: Path, *, transport: str = "legacy_patch") -> None:
    if transport == "download_bundle":
        allowed = {"artifact_transport.json", "bundle.zip", "apply_bundle.sh"}
        unexpected: list[str] = []
        for path in staging_dir.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(staging_dir).as_posix()
            if rel in allowed or _ignored_staging_file(rel):
                continue
            # Legacy answer/debug files may appear if ChatGPT also emitted text blocks;
            # keep them for diagnostics but do not let source changes ride through them.
            if rel in {"response.json", "changes.patch"}:
                unexpected.append(rel)
                continue
            unexpected.append(rel)
        if unexpected:
            raise PatchGateError(
                "download-bundle staging directory contains unexpected source-bearing files; "
                "the bundle transport must carry candidate edits only through bundle.zip:\n"
                + "\n".join(sorted(unexpected)[:200])
            )
        return

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
            "legacy Milestone 1 patch attempts must transport compiler changes only through changes.patch:\n"
            + "\n".join(sorted(unexpected)[:200])
        )


def load_download_bundle_transport(staging_dir: Path) -> tuple[Path, Path | None, dict[str, Any]]:
    validate_staging_file_set(staging_dir, transport="download_bundle")
    transport_path = staging_dir / "artifact_transport.json"
    transport: dict[str, Any] = {}
    if transport_path.exists():
        data = read_json_file(transport_path)
        if not isinstance(data, dict):
            raise PatchGateError("artifact_transport.json must contain a JSON object")
        transport = dict(data)

    def resolve_candidate(value: Any, fallback: str) -> Path:
        if isinstance(value, str) and value.strip():
            candidate = staging_dir / normalize_relpath(value)
            if candidate.exists():
                return candidate
        return staging_dir / fallback

    downloaded = transport.get("downloaded_files") if isinstance(transport.get("downloaded_files"), dict) else {}
    bundle_value = downloaded.get("bundle_zip") or transport.get("bundle_zip_path") or transport.get("bundle_zip")
    script_value = downloaded.get("apply_script") or transport.get("apply_script_path") or transport.get("apply_script")
    bundle_path = resolve_candidate(bundle_value, "bundle.zip")
    script_path = resolve_candidate(script_value, "apply_bundle.sh") if script_value or (staging_dir / "apply_bundle.sh").exists() else (staging_dir / "apply_bundle.sh")
    if not bundle_path.exists():
        raise PatchGateError(f"download bundle transport is missing bundle.zip: looked for {bundle_path}")
    if not script_path.exists():
        raise PatchGateError(f"download bundle transport is missing apply_bundle.sh: looked for {script_path}")
    return bundle_path, script_path, transport

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
        if is_generated_test_side_effect_path(normalized):
            raise PatchGateError(f"generated lit side-effect path is not a source product: {normalized}")
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
    transport = detect_staging_transport(staging_dir)
    if transport == "download_bundle":
        expected_attempt = expected_attempt_from_staging_dir(staging_dir)
        bundle_path, script_path, transport_meta = load_download_bundle_transport(staging_dir)
        validate_download_transport_metadata(
            transport_meta,
            slice_entry=slice_entry,
            expected_attempt=expected_attempt,
        )
        try:
            report = apply_bundle(
                repo=repo,
                config=config,
                slice_entry=slice_entry,
                bundle_path=bundle_path,
                apply_script=script_path,
                expect_attempt=expected_attempt,
                check_only=not apply_patch,
                write_report=write_report,
            )
        except BundleApplyError as exc:
            raise PatchGateError(str(exc)) from exc
        report["staging_dir"] = str(staging_dir)
        report["artifact_transport_json"] = transport_meta
        report["expected_attempt"] = expected_attempt
        if write_report:
            write_json_file(write_report, report)
        return report
    if transport == "unknown":
        raise PatchGateError(
            f"staging directory contains neither downloadable bundle artifacts nor legacy response.json + changes.patch: {staging_dir}"
        )

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
        "transport": "legacy_patch",
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
    generated_side_effects: list[str] = []
    for path in changed:
        if is_generated_test_side_effect_path(path):
            generated_side_effects.append(path)
            continue
        if match_any_path(path, forbidden):
            forbidden_hits.append(path)
            continue
        if not match_any_path(path, allowed):
            disallowed.append(path)
    bad = sorted(set(disallowed + forbidden_hits + generated_side_effects))
    if bad and restore_disallowed:
        restore_paths(repo, bad)
    return {
        "ok": not bad,
        "changed_paths": changed,
        "disallowed_paths": disallowed,
        "forbidden_paths": forbidden_hits,
        "generated_side_effects": generated_side_effects,
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
