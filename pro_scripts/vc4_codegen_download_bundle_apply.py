#!/usr/bin/env python3
"""Validate and apply VC4 GPT downloadable bundle artifacts.

This module is intentionally the trusted local side of the downloadable
artifact transport.  GPT Pro may create a zip and a shell launcher, but this
script validates the zip manifest, slice path policy, file hashes, and optional
launcher shape before any repository file is written.  The GPT-generated shell
script is never needed for privileged logic; it is accepted only as a tiny
human-readable launcher for manual recovery.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    from vc4_codegen_state import (
        DriverError,
        MilestoneConfig,
        eprint,
        find_repo_root,
        git_changed_paths,
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
        normalize_relpath,
        read_json_file,
        reject_if_forbidden,
        relpath,
        require_allowed,
        restore_paths,
        write_json_file,
    )

TRANSPORT = "vc4_codegen_download_bundle_v1"
MANIFEST_NAME = "manifest.json"
REPO_PREFIX = "repo/"
ALLOWED_MODES = {"0644", "0755", "100644", "100755"}
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_BUNDLE_BYTES = 64 * 1024 * 1024


class BundleApplyError(DriverError):
    pass


@dataclass(frozen=True)
class BundleEntry:
    path: str
    action: str
    sha256: str | None = None
    mode: str = "0644"
    zip_name: str | None = None


def sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _load_json_from_zip(zf: zipfile.ZipFile, name: str) -> Any:
    try:
        with zf.open(name) as fh:
            raw = fh.read(MAX_FILE_BYTES + 1)
    except KeyError as exc:
        raise BundleApplyError(f"bundle zip is missing {name}") from exc
    if len(raw) > MAX_FILE_BYTES:
        raise BundleApplyError(f"{name} is too large")
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise BundleApplyError(f"{name} is not valid UTF-8 JSON: {exc}") from exc
    return data


def _is_zipinfo_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o777777
    return stat.S_ISLNK(mode)


def _safe_zip_member(name: str) -> str:
    value = str(name or "").replace("\\", "/").strip()
    if not value:
        raise BundleApplyError("bundle contains an empty zip member name")
    if value.startswith("/"):
        raise BundleApplyError(f"bundle contains absolute zip member path: {value}")
    parts = [p for p in PurePosixPath(value).parts if p not in {"", "."}]
    if any(p == ".." for p in parts):
        raise BundleApplyError(f"bundle contains parent-directory zip member path: {value}")
    return "/".join(parts)


def _normalize_bundle_repo_path(raw: str) -> str:
    rel = normalize_relpath(str(raw).replace("\\", "/"))
    if rel == ".vc4_auto" or rel.startswith(".vc4_auto/"):
        raise BundleApplyError(f"bundle path is reserved for automation state: {rel}")
    if rel == ".git" or rel.startswith(".git/"):
        raise BundleApplyError(f"bundle path is reserved for git metadata: {rel}")
    return rel


def _normalize_mode(raw: Any) -> str:
    value = str(raw if raw is not None else "0644").strip()
    if value in {"100644", "0644", "644"}:
        return "0644"
    if value in {"100755", "0755", "755"}:
        return "0755"
    if value not in ALLOWED_MODES:
        raise BundleApplyError(f"unsupported file mode {value!r}; expected 0644 or 0755")
    return value[-4:]


def _entry_from_manifest_item(item: Any) -> BundleEntry:
    if isinstance(item, str):
        path = _normalize_bundle_repo_path(item)
        return BundleEntry(path=path, action="write", zip_name=REPO_PREFIX + path)
    if not isinstance(item, Mapping):
        raise BundleApplyError("manifest changed_paths entries must be strings or objects")
    path_raw = item.get("path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        raise BundleApplyError("manifest changed_paths object missing non-empty string path")
    path = _normalize_bundle_repo_path(path_raw)
    action = str(item.get("action", "write")).strip().lower()
    if action not in {"write", "delete"}:
        raise BundleApplyError(f"unsupported manifest action for {path}: {action!r}")
    digest = item.get("sha256")
    if digest is not None:
        digest = str(digest).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BundleApplyError(f"manifest sha256 for {path} is not a 64-character lowercase hex digest")
    mode = _normalize_mode(item.get("mode", "0644"))
    zip_name_raw = item.get("zip_name") or item.get("source") or (REPO_PREFIX + path)
    zip_name = _safe_zip_member(str(zip_name_raw)) if action == "write" else None
    if zip_name and not zip_name.startswith(REPO_PREFIX):
        raise BundleApplyError(f"write entry for {path} must source from repo/: {zip_name}")
    return BundleEntry(path=path, action=action, sha256=digest, mode=mode, zip_name=zip_name)


def _manifest_entries(manifest: Mapping[str, Any]) -> list[BundleEntry]:
    raw_changed = manifest.get("changed_paths")
    if not isinstance(raw_changed, list) or not raw_changed:
        raise BundleApplyError("manifest.changed_paths must be a non-empty array")
    entries = [_entry_from_manifest_item(item) for item in raw_changed]
    raw_deleted = manifest.get("deleted_paths")
    if raw_deleted is not None:
        if not isinstance(raw_deleted, list) or not all(isinstance(x, str) for x in raw_deleted):
            raise BundleApplyError("manifest.deleted_paths must be an array of strings when present")
        for raw in raw_deleted:
            entries.append(BundleEntry(path=_normalize_bundle_repo_path(raw), action="delete"))

    seen: dict[str, str] = {}
    deduped: list[BundleEntry] = []
    for entry in entries:
        if entry.path in seen:
            if seen[entry.path] != entry.action:
                raise BundleApplyError(f"manifest lists {entry.path} with conflicting actions")
            # Duplicate same-action entries are almost always accidental and make
            # changed-path accounting ambiguous, so reject them loudly.
            raise BundleApplyError(f"manifest lists {entry.path} more than once")
        seen[entry.path] = entry.action
        deduped.append(entry)
    return deduped


def _validate_manifest_header(manifest: Any, *, slice_entry: Mapping[str, Any], expect_attempt: int | None) -> Mapping[str, Any]:
    if not isinstance(manifest, Mapping):
        raise BundleApplyError("bundle manifest must be a JSON object")
    if int(manifest.get("schema_version", 0) or 0) != 1:
        raise BundleApplyError("bundle manifest schema_version must be 1")
    if str(manifest.get("transport", "")) != TRANSPORT:
        raise BundleApplyError(f"bundle manifest transport must be {TRANSPORT!r}")
    if str(manifest.get("slice_id", "")) != str(slice_entry.get("id")):
        raise BundleApplyError(
            f"bundle manifest slice_id {manifest.get('slice_id')!r} does not match active slice {slice_entry.get('id')!r}"
        )
    if expect_attempt is not None and manifest.get("attempt") is not None:
        try:
            got = int(manifest.get("attempt"))
        except Exception as exc:
            raise BundleApplyError("bundle manifest attempt must be an integer when present") from exc
        if got != int(expect_attempt):
            raise BundleApplyError(f"bundle manifest attempt {got} does not match expected attempt {expect_attempt}")
    for key in ("diagnosis", "risk_notes", "tests_to_run"):
        if key in manifest and manifest[key] is not None and not isinstance(manifest[key], list):
            raise BundleApplyError(f"bundle manifest {key} must be an array when present")
    return manifest


def _validate_launcher_script(script_path: Path | None, *, bundle_path: Path, slice_entry: Mapping[str, Any], expect_attempt: int | None) -> dict[str, Any]:
    if script_path is None:
        return {"present": False, "validated": False, "reason": "no apply script supplied"}
    if not script_path.exists():
        raise BundleApplyError(f"apply script not found: {script_path}")
    text = script_path.read_text(encoding="utf-8", errors="replace")
    if len(text) > 64_000:
        raise BundleApplyError("apply script is too large for a trusted launcher")
    if "\0" in text:
        raise BundleApplyError("apply script contains NUL bytes")
    forbidden = ["rm -rf", "curl ", "wget ", "nc ", "python -c", "python3 -c", "eval ", "source ", ". "]
    hits = [s for s in forbidden if s in text]
    if hits:
        raise BundleApplyError("apply script is not a narrow launcher; forbidden snippet(s): " + ", ".join(hits))
    if "vc4_codegen_download_bundle_apply.py" not in text:
        raise BundleApplyError("apply script must delegate to pro_scripts/vc4_codegen_download_bundle_apply.py")
    if str(slice_entry.get("id")) not in text:
        raise BundleApplyError("apply script must include the active slice id")
    if expect_attempt is not None and str(expect_attempt) not in text:
        raise BundleApplyError("apply script must include the expected attempt number")
    if bundle_path.name not in text and ".zip" not in text:
        raise BundleApplyError("apply script must reference a bundle zip filename")
    return {"present": True, "validated": True, "path": str(script_path), "bytes": len(text)}


def validate_bundle(
    *,
    repo: Path,
    config: MilestoneConfig,
    slice_entry: Mapping[str, Any],
    bundle_path: Path,
    apply_script: Path | None = None,
    expect_attempt: int | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    bundle_path = bundle_path.resolve()
    if not bundle_path.exists():
        raise BundleApplyError(f"bundle zip not found: {bundle_path}")
    if bundle_path.stat().st_size > MAX_BUNDLE_BYTES:
        raise BundleApplyError(f"bundle zip is too large: {bundle_path.stat().st_size} bytes")

    allowed = config.allowed_paths_for_slice(slice_entry)
    forbidden = config.forbidden_paths_for_slice(slice_entry)
    with zipfile.ZipFile(bundle_path) as zf:
        normalized_names = [_safe_zip_member(info.filename) for info in zf.infolist() if not info.is_dir()]
        infos = {_safe_zip_member(info.filename): info for info in zf.infolist() if not info.is_dir()}
        if MANIFEST_NAME not in infos:
            raise BundleApplyError("bundle zip must contain manifest.json at top level")
        for name, info in infos.items():
            if _is_zipinfo_symlink(info):
                raise BundleApplyError(f"bundle zip member is a symlink and is rejected: {name}")
            if not (name == MANIFEST_NAME or name.startswith(REPO_PREFIX)):
                raise BundleApplyError(f"unexpected bundle zip member outside manifest.json and repo/: {name}")
            if info.file_size > MAX_FILE_BYTES:
                raise BundleApplyError(f"bundle zip member is too large: {name}")

        manifest = _validate_manifest_header(_load_json_from_zip(zf, MANIFEST_NAME), slice_entry=slice_entry, expect_attempt=expect_attempt)
        entries = _manifest_entries(manifest)
        manifest_repo_names = {entry.zip_name for entry in entries if entry.action == "write" and entry.zip_name}
        actual_repo_names = {name for name in normalized_names if name.startswith(REPO_PREFIX)}
        if manifest_repo_names != actual_repo_names:
            missing = sorted(manifest_repo_names - actual_repo_names)
            extra = sorted(actual_repo_names - manifest_repo_names)
            raise BundleApplyError(
                "bundle repo/ members must exactly match write entries in manifest"
                + ("\nmissing from zip:\n" + "\n".join(missing) if missing else "")
                + ("\nextra in zip:\n" + "\n".join(extra) if extra else "")
            )

        entry_reports: list[dict[str, Any]] = []
        for entry in entries:
            reject_if_forbidden(entry.path, forbidden)
            require_allowed(entry.path, allowed)
            item: dict[str, Any] = {"path": entry.path, "action": entry.action, "mode": entry.mode}
            if entry.action == "write":
                assert entry.zip_name is not None
                data = zf.read(entry.zip_name)
                if b"\0" in data:
                    raise BundleApplyError(f"bundle file contains NUL bytes and is rejected as binary: {entry.path}")
                digest = sha256_bytes(data)
                if entry.sha256 and digest != entry.sha256:
                    raise BundleApplyError(f"sha256 mismatch for {entry.path}: manifest={entry.sha256} actual={digest}")
                item.update({"sha256": digest, "bytes": len(data), "zip_name": entry.zip_name})
            entry_reports.append(item)

    script_report = _validate_launcher_script(apply_script, bundle_path=bundle_path, slice_entry=slice_entry, expect_attempt=expect_attempt)
    return {
        "schema_version": 1,
        "ok": True,
        "transport": TRANSPORT,
        "slice_id": slice_entry.get("id"),
        "bundle_path": str(bundle_path),
        "apply_script": script_report,
        "manifest": dict(manifest),
        "entries": entry_reports,
        "changed_paths": [e.path for e in entries],
    }


def apply_bundle(
    *,
    repo: Path,
    config: MilestoneConfig,
    slice_entry: Mapping[str, Any],
    bundle_path: Path,
    apply_script: Path | None = None,
    expect_attempt: int | None = None,
    check_only: bool = False,
    write_report: Path | None = None,
) -> dict[str, Any]:
    report = validate_bundle(
        repo=repo,
        config=config,
        slice_entry=slice_entry,
        bundle_path=bundle_path,
        apply_script=apply_script,
        expect_attempt=expect_attempt,
    )
    report["applied"] = False
    report["check_only"] = bool(check_only)
    if check_only:
        if write_report:
            write_json_file(write_report, report)
        return report

    before = set(git_changed_paths(repo, include_untracked=True))
    entries = _manifest_entries(report["manifest"])
    try:
        with zipfile.ZipFile(bundle_path.resolve()) as zf:
            for entry in entries:
                target = repo / entry.path
                if entry.action == "delete":
                    if target.exists() or target.is_symlink():
                        target.unlink()
                    continue
                assert entry.zip_name is not None
                data = zf.read(entry.zip_name)
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
                try:
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(data)
                    os.chmod(tmp_name, 0o755 if entry.mode == "0755" else 0o644)
                    os.replace(tmp_name, target)
                finally:
                    try:
                        if os.path.exists(tmp_name):
                            os.unlink(tmp_name)
                    except OSError:
                        pass
    except Exception:
        current = set(git_changed_paths(repo, include_untracked=True))
        restore_paths(repo, sorted(current - before))
        raise

    after = git_changed_paths(repo, include_untracked=True)
    report["applied"] = True
    report["repo_changed_paths_after_apply"] = after
    if write_report:
        write_json_file(write_report, report)
    return report


def cmd_validate_apply(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    slice_entry = config.get_slice(args.slice)
    bundle = Path(args.bundle)
    if not bundle.is_absolute():
        bundle = repo / bundle
    script = Path(args.apply_script) if args.apply_script else None
    if script is not None and not script.is_absolute():
        script = repo / script
    report = apply_bundle(
        repo=repo,
        config=config,
        slice_entry=slice_entry,
        bundle_path=bundle,
        apply_script=script,
        expect_attempt=args.expect_attempt if args.expect_attempt > 0 else None,
        check_only=args.check_only,
        write_report=(repo / args.report).resolve() if args.report else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--worklist", default="pro_scripts/vc4_codegen_m1_worklist.json")
    parser.add_argument("--context-profiles", default="pro_scripts/vc4_codegen_m1_context_profiles.json")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("validate-apply", help="validate and optionally apply a GPT downloadable bundle")
    p.add_argument("--slice", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--apply-script", default="")
    p.add_argument("--expect-attempt", type=int, default=0)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--report", default="")
    p.set_defaults(func=cmd_validate_apply)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DriverError as exc:
        eprint(f"[vc4-bundle-apply] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
