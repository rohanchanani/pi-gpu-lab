#!/usr/bin/env python3
"""Shared state and repository helpers for VC4 codegen Milestone 1 automation.

This module is intentionally dependency-free.  It is used by the Stage 2
orchestrator, patch gate, gate runner, and failure classifier.  The public
surface is small and conservative: repository discovery, JSON loading/writing,
worklist access, state persistence, path normalization, and git helpers.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


class DriverError(RuntimeError):
    """Fatal automation error with a message suitable for the terminal."""


REPO_SENTINELS = (".git", "compiler", "pro_scripts")
DEFAULT_WORKLIST = Path("pro_scripts/vc4_codegen_m1_worklist.json")
DEFAULT_CONTEXT_PROFILES = Path("pro_scripts/vc4_codegen_m1_context_profiles.json")
DEFAULT_STATE_ROOT = Path(".vc4_auto/codegen_m1")


@dataclasses.dataclass(frozen=True)
class GitResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclasses.dataclass(frozen=True)
class AttemptPaths:
    slice_id: str
    attempt: int
    attempt_name: str
    prompt_path: Path
    staging_dir: Path
    log_dir: Path
    failure_packet_path: Path


@dataclasses.dataclass
class MilestoneConfig:
    repo: Path
    worklist_path: Path
    context_profiles_path: Path
    worklist: dict[str, Any]
    context_profiles: dict[str, Any]

    @classmethod
    def load(
        cls,
        repo: Path,
        *,
        worklist_path: Path | str | None = None,
        context_profiles_path: Path | str | None = None,
    ) -> "MilestoneConfig":
        repo = repo.resolve()
        wp = repo / (Path(worklist_path) if worklist_path else DEFAULT_WORKLIST)
        cp = repo / (Path(context_profiles_path) if context_profiles_path else DEFAULT_CONTEXT_PROFILES)
        if not wp.exists():
            raise DriverError(f"missing worklist: {relpath(repo, wp)}")
        if not cp.exists():
            raise DriverError(f"missing context profiles: {relpath(repo, cp)}")
        worklist = read_json_file(wp)
        profiles = read_json_file(cp)
        validate_worklist_shape(worklist)
        validate_context_profiles_shape(profiles)
        return cls(repo=repo, worklist_path=wp, context_profiles_path=cp, worklist=worklist, context_profiles=profiles)

    @property
    def milestone(self) -> str:
        return str(self.worklist.get("milestone", "vc4-codegen-m1"))

    @property
    def defaults(self) -> dict[str, Any]:
        value = self.worklist.get("defaults", {})
        if not isinstance(value, dict):
            raise DriverError("worklist.defaults must be an object")
        return value

    @property
    def slices(self) -> list[dict[str, Any]]:
        slices = self.worklist.get("slices", [])
        if not isinstance(slices, list):
            raise DriverError("worklist.slices must be an array")
        return slices

    def slice_ids(self) -> list[str]:
        return [str(s["id"]) for s in self.slices]

    def get_slice(self, slice_id: str) -> dict[str, Any]:
        for s in self.slices:
            if s.get("id") == slice_id:
                return s
        raise DriverError(f"unknown slice id: {slice_id}")

    def default_forbidden_paths(self) -> list[str]:
        raw = self.defaults.get("forbidden_paths", [])
        if not isinstance(raw, list):
            raise DriverError("worklist.defaults.forbidden_paths must be an array")
        return [str(x) for x in raw]

    def forbidden_paths_for_slice(self, slice_entry: Mapping[str, Any]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for pattern in [*self.default_forbidden_paths(), *as_str_list(slice_entry.get("forbidden_paths", []))]:
            if pattern not in seen:
                seen.add(pattern)
                out.append(pattern)
        return out

    def allowed_paths_for_slice(self, slice_entry: Mapping[str, Any]) -> list[str]:
        return as_str_list(slice_entry.get("allowed_paths", []))

    def state_root(self) -> Path:
        raw = self.defaults.get("state_root", str(DEFAULT_STATE_ROOT))
        return self.repo / Path(str(raw))

    def build_dir(self) -> Path:
        raw = self.defaults.get("build_dir", "compiler/build")
        return self.repo / Path(str(raw))

    def gpt_web_driver(self) -> Path:
        raw = self.defaults.get("gpt_web_driver", "pro_scripts/gpt_web_driver.js")
        return self.repo / Path(str(raw))

    def auto_commit_on_pass(self) -> bool:
        return bool(self.defaults.get("auto_commit_on_pass", True))


class StateStore:
    """Persistent Milestone 1 state under .vc4_auto/codegen_m1."""

    def __init__(self, config: MilestoneConfig):
        self.config = config
        self.repo = config.repo
        self.root = config.state_root()
        self.path = self.root / "state.json"

    def ensure_dirs(self) -> None:
        for child in ["prompts", "staging", "logs", "failure_packets", "candidates", "tmp"]:
            (self.root / child).mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        self.ensure_dirs()
        if not self.path.exists():
            return {
                "schema_version": 1,
                "milestone": self.config.milestone,
                "created_at": iso_now(),
                "updated_at": iso_now(),
                "slices": {},
                "attempts": [],
            }
        data = read_json_file(self.path)
        if not isinstance(data, dict):
            raise DriverError(f"state file must contain a JSON object: {relpath(self.repo, self.path)}")
        data.setdefault("schema_version", 1)
        data.setdefault("milestone", self.config.milestone)
        data.setdefault("slices", {})
        data.setdefault("attempts", [])
        return data

    def save(self, data: Mapping[str, Any]) -> None:
        payload = dict(data)
        payload["updated_at"] = iso_now()
        self.ensure_dirs()
        atomic_write_text(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def reset_slice(self, slice_id: str) -> None:
        data = self.load()
        slices = data.setdefault("slices", {})
        if isinstance(slices, MutableMapping):
            slices.pop(slice_id, None)
        attempts = data.setdefault("attempts", [])
        if isinstance(attempts, list):
            data["attempts"] = [a for a in attempts if not (isinstance(a, dict) and a.get("slice_id") == slice_id)]
        self.save(data)

    def slice_status(self, slice_id: str) -> str:
        entry = self.load().get("slices", {}).get(slice_id)
        if isinstance(entry, dict):
            return str(entry.get("status", "pending"))
        return "pending"

    def dependency_status(self, slice_entry: Mapping[str, Any]) -> dict[str, str]:
        return {dep: self.slice_status(str(dep)) for dep in slice_entry.get("depends_on", [])}

    def next_pending_slice(self) -> dict[str, Any] | None:
        for slice_entry in self.config.slices:
            sid = str(slice_entry["id"])
            if self.slice_status(sid) == "passed":
                continue
            dep_status = self.dependency_status(slice_entry)
            if all(status == "passed" for status in dep_status.values()):
                return slice_entry
        return None

    def next_attempt_index(self, slice_id: str) -> int:
        data = self.load()
        attempts = data.get("attempts", [])
        existing = [int(a.get("attempt", 0)) for a in attempts if isinstance(a, dict) and a.get("slice_id") == slice_id]
        return (max(existing) + 1) if existing else 1

    def attempt_paths(self, slice_id: str, attempt: int) -> AttemptPaths:
        self.ensure_dirs()
        attempt_name = f"attempt-{attempt:02d}"
        prompt_dir = self.root / "prompts" / slice_id
        staging_dir = self.root / "staging" / slice_id / attempt_name
        log_dir = self.root / "logs" / slice_id / attempt_name
        failure_dir = self.root / "failure_packets" / slice_id
        for p in [prompt_dir, staging_dir, log_dir, failure_dir]:
            p.mkdir(parents=True, exist_ok=True)
        return AttemptPaths(
            slice_id=slice_id,
            attempt=attempt,
            attempt_name=attempt_name,
            prompt_path=prompt_dir / f"{attempt_name}.md",
            staging_dir=staging_dir,
            log_dir=log_dir,
            failure_packet_path=failure_dir / f"{attempt_name}.json",
        )

    def record_attempt(self, *, slice_id: str, attempt: int, status: str, details: Mapping[str, Any] | None = None) -> None:
        data = self.load()
        attempts = data.setdefault("attempts", [])
        if not isinstance(attempts, list):
            data["attempts"] = attempts = []
        attempts.append({
            "slice_id": slice_id,
            "attempt": attempt,
            "status": status,
            "time": iso_now(),
            "details": dict(details or {}),
        })
        self.save(data)

    def mark_slice_passed(self, *, slice_id: str, gate_results: Sequence[Mapping[str, Any]] | None = None, repo_head: str | None = None) -> None:
        data = self.load()
        slices = data.setdefault("slices", {})
        if not isinstance(slices, MutableMapping):
            data["slices"] = slices = {}
        slices[slice_id] = {
            "status": "passed",
            "passed_at": iso_now(),
            "repo_head": repo_head or git_head(self.repo, allow_missing=True),
            "gates": [dict(x) for x in (gate_results or [])],
        }
        self.save(data)

    def mark_slice_failed(self, *, slice_id: str, failure_packet: Path | str | None = None) -> None:
        data = self.load()
        slices = data.setdefault("slices", {})
        if not isinstance(slices, MutableMapping):
            data["slices"] = slices = {}
        entry = dict(slices.get(slice_id, {}))
        entry.update({
            "status": "failed",
            "failed_at": iso_now(),
        })
        if failure_packet:
            entry["failure_packet"] = str(failure_packet)
        slices[slice_id] = entry
        self.save(data)


# ---------------------------------------------------------------------------
# JSON / filesystem helpers
# ---------------------------------------------------------------------------


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DriverError(f"invalid JSON in {path}: {exc}") from exc


def write_json_file(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DriverError(f"expected a JSON array of strings, got {type(value).__name__}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise DriverError(f"expected string list item, got {type(item).__name__}")
        out.append(item)
    return out


def validate_worklist_shape(worklist: Mapping[str, Any]) -> None:
    if not isinstance(worklist, Mapping):
        raise DriverError("worklist must be a JSON object")
    if "slices" not in worklist or not isinstance(worklist["slices"], list):
        raise DriverError("worklist requires a 'slices' array")
    ids: list[str] = []
    for i, s in enumerate(worklist["slices"]):
        if not isinstance(s, Mapping):
            raise DriverError(f"worklist slice #{i} must be an object")
        sid = s.get("id")
        if not isinstance(sid, str) or not sid:
            raise DriverError(f"worklist slice #{i} requires non-empty string id")
        ids.append(sid)
        for key in ["title", "context_profile", "intent", "allowed_paths", "forbidden_paths", "gates"]:
            if key not in s:
                raise DriverError(f"slice {sid} missing required key '{key}'")
    if len(ids) != len(set(ids)):
        raise DriverError("worklist contains duplicate slice ids")
    id_set = set(ids)
    for s in worklist["slices"]:
        for dep in s.get("depends_on", []):
            if dep not in id_set:
                raise DriverError(f"slice {s['id']} depends on unknown slice {dep}")


def validate_context_profiles_shape(profiles: Mapping[str, Any]) -> None:
    if not isinstance(profiles, Mapping):
        raise DriverError("context profiles must be a JSON object")
    if "profiles" not in profiles or not isinstance(profiles["profiles"], Mapping):
        raise DriverError("context profiles require a 'profiles' object")


def find_repo_root(start: Path | str = ".") -> Path:
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for candidate in [path, *path.parents]:
        if (candidate / ".git").exists() and (candidate / "compiler").exists():
            return candidate
        if all((candidate / sentinel).exists() for sentinel in REPO_SENTINELS if sentinel != ".git"):
            return candidate
    raise DriverError(f"could not find repo root from {start}")


def relpath(repo: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except Exception:
        return str(path)


def normalize_relpath(raw: str | os.PathLike[str]) -> str:
    s = str(raw).replace("\x00", "").replace("\\", "/").strip()
    while s.startswith("./"):
        s = s[2:]
    if not s:
        raise DriverError("empty relative path")
    p = PurePosixPath(s)
    if p.is_absolute():
        raise DriverError(f"absolute path is not allowed: {s}")
    parts = [part for part in p.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise DriverError(f"parent-directory path component is not allowed: {s}")
    if not parts:
        raise DriverError(f"invalid relative path: {s}")
    return "/".join(parts)


def match_path(path: str, pattern: str) -> bool:
    path = normalize_relpath(path)
    pattern = str(pattern).replace("\\", "/").strip()
    while pattern.startswith("./"):
        pattern = pattern[2:]
    # fnmatch is intentionally used here because Stage 1 patterns are shell-like
    # and include **.  fnmatch's '*' matches '/', which is acceptable for this
    # policy because allow/deny patterns are intentionally coarse-grained.
    return fnmatch.fnmatchcase(path, pattern)


def match_any_path(path: str, patterns: Iterable[str]) -> bool:
    return any(match_path(path, pattern) for pattern in patterns)


def reject_if_forbidden(path: str, patterns: Iterable[str]) -> None:
    if match_any_path(path, patterns):
        raise DriverError(f"path is forbidden by policy: {path}")


def require_allowed(path: str, allowed_patterns: Iterable[str]) -> None:
    patterns = list(allowed_patterns)
    if not patterns:
        raise DriverError(f"no allowed_paths configured; refusing path: {path}")
    if not match_any_path(path, patterns):
        raise DriverError(f"path is outside allowed_paths: {path}")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def git(repo: Path, args: Sequence[str], *, check: bool = True, capture: bool = True) -> GitResult:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        text=True,
        capture_output=capture,
    )
    result = GitResult(args=list(args), returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")
    if check and result.returncode != 0:
        raise DriverError(
            "git command failed: git "
            + " ".join(args)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    return result


def git_head(repo: Path, *, allow_missing: bool = False) -> str:
    result = git(repo, ["rev-parse", "HEAD"], check=not allow_missing)
    if result.ok:
        return result.stdout.strip()
    return ""


def git_status_paths(repo: Path, *, include_vc4_auto: bool = False) -> list[str]:
    result = git(repo, ["status", "--porcelain=v1"], check=True)
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ")[-1]
        try:
            path = normalize_relpath(raw)
        except DriverError:
            continue
        if not include_vc4_auto and path.startswith(".vc4_auto/"):
            continue
        paths.append(path)
    return paths


def git_changed_paths(repo: Path, *, include_untracked: bool = True, include_vc4_auto: bool = False) -> list[str]:
    paths: set[str] = set()
    for args in (["diff", "--name-only"], ["diff", "--cached", "--name-only"]):
        result = git(repo, args, check=True)
        for line in result.stdout.splitlines():
            try:
                p = normalize_relpath(line)
            except DriverError:
                continue
            if include_vc4_auto or not p.startswith(".vc4_auto/"):
                paths.add(p)
    if include_untracked:
        result = git(repo, ["ls-files", "--others", "--exclude-standard"], check=True)
        for line in result.stdout.splitlines():
            try:
                p = normalize_relpath(line)
            except DriverError:
                continue
            if include_vc4_auto or not p.startswith(".vc4_auto/"):
                paths.add(p)
    return sorted(paths)


def ensure_auto_excluded(repo: Path) -> None:
    exclude = repo / ".git" / "info" / "exclude"
    if not exclude.exists():
        return
    text = exclude.read_text(encoding="utf-8", errors="ignore")
    additions: list[str] = []
    for pattern in [".vc4_auto/", ".vc4_auto/**"]:
        if pattern not in text:
            additions.append(pattern)
    if additions:
        with exclude.open("a", encoding="utf-8") as f:
            f.write("\n# VC4 codegen milestone automation scratch state\n")
            for pattern in additions:
                f.write(pattern + "\n")


def ensure_clean_repo(repo: Path, *, allow_dirty: bool = False, allowed_dirty_patterns: Iterable[str] = ()) -> None:
    dirty = git_status_paths(repo)
    if not dirty:
        return
    allowed = list(allowed_dirty_patterns)
    disallowed = [p for p in dirty if not match_any_path(p, allowed)] if allowed else dirty
    if disallowed and not allow_dirty:
        raise DriverError(
            "repo is dirty before automation. Commit/stash changes first, or pass --allow-dirty.\n"
            + "\n".join(disallowed[:200])
        )


def stage_and_commit(repo: Path, *, paths: Sequence[str], message: str, allow_empty: bool = False) -> bool:
    normalized = [normalize_relpath(p) for p in paths if str(p).strip()]
    if normalized:
        git(repo, ["add", "-A", "--", *normalized], check=True)
    staged = git(repo, ["diff", "--cached", "--name-only"], check=True).stdout.strip().splitlines()
    if not staged and not allow_empty:
        return False
    args = ["commit", "-m", message]
    if allow_empty:
        args.insert(1, "--allow-empty")
    git(repo, args, check=True, capture=True)
    return True


def restore_paths(repo: Path, paths: Iterable[str]) -> None:
    tracked: list[str] = []
    untracked: list[Path] = []
    for raw in paths:
        rel = normalize_relpath(raw)
        tracked_result = git(repo, ["ls-files", "--error-unmatch", "--", rel], check=False)
        if tracked_result.ok:
            tracked.append(rel)
        else:
            untracked.append(repo / rel)
    if tracked:
        git(repo, ["restore", "--staged", "--worktree", "--", *tracked], check=False)
    for path in untracked:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------


def eprint(*parts: object) -> None:
    print(*parts, file=sys.stderr, flush=True)


def print_kv(key: str, value: object) -> None:
    print(f"{key}: {value}", flush=True)


def tail_text(text: str, *, max_lines: int = 80) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def tail_file(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists():
        return f"<missing log file: {path}>"
    return tail_text(path.read_text(encoding="utf-8", errors="replace"), max_lines=max_lines)


def executable_in_path(name: str) -> str | None:
    return shutil.which(name)


# ---------------------------------------------------------------------------
# Failure packet helper
# ---------------------------------------------------------------------------


def write_failure_packet(
    path: Path,
    *,
    slice_entry: Mapping[str, Any],
    stage: str,
    message: str,
    command: Sequence[str] | None = None,
    exit_code: int | None = None,
    timed_out: bool = False,
    log_path: Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    packet = {
        "schema_version": 1,
        "time": iso_now(),
        "slice_id": slice_entry.get("id"),
        "slice_title": slice_entry.get("title"),
        "stage": stage,
        "message": message,
        "command": list(command or []),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "log_path": str(log_path) if log_path else None,
        "log_tail": tail_file(log_path, max_lines=120) if log_path else "",
        "intent": slice_entry.get("intent"),
        "non_goals": slice_entry.get("non_goals", []),
        "extra": dict(extra or {}),
    }
    write_json_file(path, packet)
    return packet
