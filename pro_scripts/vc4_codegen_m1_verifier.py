#!/usr/bin/env python3
"""Typed deterministic verifier for VC4 Codegen Milestones 1 and 2 slices.

The verifier consumes a declarative JSON spec whose top-level ``slices`` map
contains a list of typed ``verifications`` for each slice.  Each verification
uses a named mechanism with mechanism-specific fields.  The verifier returns a
machine-readable packet that can be routed by automation, and it also supports
contract auditing and human explanations.

The script intentionally depends only on the Python standard library.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
DEFAULT_SPEC = "pro_scripts/vc4_codegen_m1_verifications.json"
DEFAULT_WORKLIST = "pro_scripts/vc4_codegen_m1_worklist.json"
TAIL_CHARS = 6000


class VerificationError(Exception):
    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


@dataclasses.dataclass
class VerificationResult:
    ok: bool
    slice_id: str
    verification_id: str
    mechanism: str
    description: str = ""
    category: str = ""
    route_hint: str = "gpt_pro"
    required: bool = True
    skipped: bool = False
    skip_reason: str = ""
    duration_sec: float = 0.0
    message: str = ""
    expected: Any = None
    actual: Any = None
    command: Optional[List[str]] = None
    exit_code: Optional[int] = None
    cwd: Optional[str] = None
    log_path: Optional[str] = None
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timed_out: bool = False
    timeout_sec: Optional[int] = None

    def to_packet(self) -> Dict[str, Any]:
        packet: Dict[str, Any] = {
            "ok": self.ok,
            "slice_id": self.slice_id,
            "verification_id": self.verification_id,
            "mechanism": self.mechanism,
            "description": self.description,
            "category": self.category or self.mechanism,
            "route_hint": self.route_hint,
            "required": self.required,
            "duration_sec": round(self.duration_sec, 3),
        }
        if self.skipped:
            packet["skipped"] = True
            packet["skip_reason"] = self.skip_reason
        if self.message:
            packet["message"] = self.message
        if self.expected is not None:
            packet["expected"] = self.expected
        if self.actual is not None:
            packet["actual"] = self.actual
        if self.command is not None:
            packet["command"] = self.command
        if self.exit_code is not None:
            packet["exit_code"] = self.exit_code
        if self.cwd is not None:
            packet["cwd"] = self.cwd
        if self.log_path is not None:
            packet["log_path"] = self.log_path
        if self.stdout_tail is not None:
            packet["stdout_tail"] = self.stdout_tail
        if self.stderr_tail is not None:
            packet["stderr_tail"] = self.stderr_tail
        if self.timed_out:
            packet["timed_out"] = True
        if self.timeout_sec is not None:
            packet["timeout_sec"] = self.timeout_sec
        if self.details:
            packet["details"] = self.details
        return packet


@dataclasses.dataclass
class CommandResult:
    ok: bool
    argv: List[str]
    cwd: Path
    exit_code: int
    stdout: str
    stderr: str
    log_path: Optional[Path]
    duration_sec: float
    timed_out: bool = False
    timeout_sec: Optional[int] = None


class VerifierContext:
    def __init__(self, repo: Path, spec: Dict[str, Any], args: argparse.Namespace):
        self.repo = repo.resolve()
        self.spec = spec
        self.args = args
        defaults = spec.get("defaults", {}) if isinstance(spec.get("defaults", {}), dict) else {}
        self.state_root = self.repo / args.state_root if args.state_root else self.repo / defaults.get(
            "state_root", ".vc4_auto/codegen_m1/verifier"
        )
        self.build_dir = self.repo / defaults.get("build_dir", "compiler/build")
        self.env = os.environ.copy()
        path_prefix = defaults.get("path_prefix", ["compiler/build/bin"])
        if isinstance(path_prefix, list):
            prefix_paths = [str((self.repo / p).resolve()) for p in path_prefix]
            self.env["PATH"] = os.pathsep.join(prefix_paths + [self.env.get("PATH", "")])

        # Candidate support scripts are shared by M1 and M2.  M1 defaults to
        # .vc4_auto/codegen_m1; M2 specs set candidate_state_root to
        # .vc4_auto/codegen_m2 so fixture_matrix/candidate_phase do not collide
        # with completed M1 artifacts.
        candidate_state_root = defaults.get("candidate_state_root")
        if isinstance(candidate_state_root, str) and candidate_state_root:
            self.env["VC4_CODEGEN_STATE_ROOT"] = candidate_state_root

        self.timeout_sec = int(args.timeout_sec or defaults.get("timeout_sec", 7200))
        self.hardware_timeout_sec = int(
            args.hardware_timeout_sec
            or os.environ.get("VC4_HARDWARE_TIMEOUT_SEC")
            or defaults.get("hardware_timeout_sec", 120)
        )
        self.dry_run = bool(args.dry_run)
        self.no_hardware = bool(args.no_hardware)
        self.keep_going = bool(args.keep_going)
        self.only_mechanisms = set(str(x) for x in getattr(args, "only_mechanism", []) or [])
        self.verbose = bool(args.verbose)
        self.report_dir = self.state_root / "reports"
        self.log_dir = self.state_root / "logs"

    def rel(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.repo))
        except Exception:
            return str(path)

    def repo_path(self, relpath: str) -> Path:
        norm = normalize_repo_relpath(relpath)
        return self.repo / norm

    def resolve_tool(self, name: str) -> str:
        explicit = self.repo / name
        if explicit.exists() and os.access(explicit, os.X_OK):
            return str(explicit)
        build_bin = self.build_dir / "bin" / name
        if build_bin.exists() and os.access(build_bin, os.X_OK):
            return str(build_bin)
        found = shutil.which(name, path=self.env.get("PATH"))
        if found:
            return found
        raise VerificationError(
            f"tool not found: {name}",
            details={"tool": name, "searched": [str(build_bin), "PATH"]},
        )

    def command_log_path(self, slice_id: str, verification_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{slice_id}_{verification_id}")
        return self.log_dir / f"{safe}.log"

    def run_command(
        self,
        argv: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        timeout_sec: Optional[int] = None,
        log_path: Optional[Path] = None,
        input_text: Optional[str] = None,
    ) -> CommandResult:
        real_cwd = cwd.resolve() if cwd else self.repo
        cmd = [str(x) for x in argv]
        if self.dry_run:
            return CommandResult(True, cmd, real_cwd, 0, "", "", log_path, 0.0)
        started = time.time()
        real_cwd.mkdir(parents=True, exist_ok=True)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        effective_timeout = timeout_sec or self.timeout_sec
        proc = subprocess.Popen(
            cmd,
            cwd=str(real_cwd),
            env=self.env,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(input=input_text, timeout=effective_timeout)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = coerce_process_text(e.stdout)
            stderr = coerce_process_text(e.stderr)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError:
                proc.terminate()
            try:
                more_stdout, more_stderr = proc.communicate(timeout=2.0)
                stdout += coerce_process_text(more_stdout)
                stderr += coerce_process_text(more_stderr)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    proc.kill()
                more_stdout, more_stderr = proc.communicate()
                stdout += coerce_process_text(more_stdout)
                stderr += coerce_process_text(more_stderr)
            exit_code = 124

        duration = time.time() - started
        stdout = coerce_process_text(stdout)
        stderr = coerce_process_text(stderr)
        if log_path:
            with log_path.open("w", encoding="utf-8") as f:
                f.write(f"# cwd: {real_cwd}\n")
                f.write("# command: " + " ".join(shell_quote(x) for x in cmd) + "\n")
                f.write(f"# exit_code: {exit_code}\n")
                if timed_out:
                    f.write("# timed_out: true\n")
                    f.write(f"# timeout_sec: {effective_timeout}\n")
                f.write(f"# duration_sec: {duration:.3f}\n")
                f.write("\n## stdout\n")
                f.write(stdout)
                f.write("\n## stderr\n")
                f.write(stderr)
        return CommandResult(
            ok=(exit_code == 0 and not timed_out),
            argv=cmd,
            cwd=real_cwd,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            log_path=log_path,
            duration_sec=duration,
            timed_out=timed_out,
            timeout_sec=int(effective_timeout) if timed_out else None,
        )


def coerce_process_text(value: Any) -> str:
    """Normalize subprocess stdout/stderr, including timeout byte payloads."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def is_hardware_like_verification(v: Mapping[str, Any]) -> bool:
    """Return true for verifications likely to own pi-install/serial hangs."""
    mechanism = str(v.get("mechanism", ""))
    if mechanism == "hardware_run":
        return True
    if bool(v.get("requires_hardware", False)):
        return True
    if mechanism == "candidate_phase" and str(v.get("phase", "")) in {"build", "run"}:
        # The build phase should avoid pi-install, but older fixture Makefiles or
        # runner regressions can accidentally invoke it.  Treat it as
        # hardware-adjacent so timeout reports route to the semantic owner.
        return True
    return False


def verification_timeout_sec(ctx: "VerifierContext", v: Mapping[str, Any]) -> int:
    if "timeout_sec" in v:
        return int(v.get("timeout_sec", ctx.timeout_sec))
    if is_hardware_like_verification(v):
        return int(ctx.hardware_timeout_sec)
    return int(ctx.timeout_sec)


def shell_quote(s: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", s):
        return s
    return "'" + s.replace("'", "'\\''") + "'"


def normalize_repo_relpath(path: str) -> str:
    p = str(path).replace("\\", "/").strip()
    if not p:
        raise VerificationError("empty path")
    if p.startswith("/"):
        raise VerificationError(f"absolute paths are not allowed in verifier specs: {path}")
    parts = [part for part in p.split("/") if part and part != "."]
    if any(part == ".." for part in parts):
        raise VerificationError(f"parent directory components are not allowed: {path}")
    return "/".join(parts)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SystemExit(f"missing JSON file: {path}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid JSON in {path}: {e}") from e


def tail(text: str, limit: int = TAIL_CHARS) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def get_slices(spec: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    slices = spec.get("slices")
    if isinstance(slices, dict):
        return {str(k): v for k, v in slices.items() if isinstance(v, dict)}
    if isinstance(slices, list):
        out: Dict[str, Dict[str, Any]] = {}
        for item in slices:
            if isinstance(item, dict) and item.get("id"):
                out[str(item["id"])] = item
        return out
    return {}


def iter_repo_files(repo: Path) -> Iterable[str]:
    for root, dirs, files in os.walk(repo):
        root_path = Path(root)
        rel_root = root_path.relative_to(repo)
        parts = set(rel_root.parts)
        if ".git" in parts:
            dirs[:] = []
            continue
        # Generated state can be huge. Only walk it when explicitly named by exact path.
        if ".vc4_auto" in parts:
            dirs[:] = []
            continue
        for name in files:
            yield str((rel_root / name).as_posix())


def glob_repo(repo: Path, pattern: str) -> List[str]:
    pat = normalize_repo_relpath(pattern)
    matches = [p for p in iter_repo_files(repo) if fnmatch.fnmatch(p, pat)]
    matches.sort()
    return matches


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# MLIR launch ABI extraction helpers.  This is intentionally light-weight: it
# extracts the dictionary shape that is stable in Milestone 1 inputs without
# becoming a full MLIR parser.
# ---------------------------------------------------------------------------


def extract_balanced_after(text: str, key: str, open_ch: str, close_ch: str) -> Optional[str]:
    idx = text.find(key)
    if idx < 0:
        return None
    start = text.find(open_ch, idx)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def split_top_level_dicts(array_text: str) -> List[str]:
    out: List[str] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False
    for i, ch in enumerate(array_text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(array_text[start : i + 1])
                start = None
    return out


def parse_launch_abi_entry(entry: str) -> Dict[str, Any]:
    def str_field(name: str) -> Optional[str]:
        m = re.search(rf"\b{name}\s*=\s*\"([^\"]+)\"", entry)
        return m.group(1) if m else None

    def int_field(name: str) -> Optional[int]:
        m = re.search(rf"\b{name}\s*=\s*(-?\d+)\s*:\s*i32", entry)
        return int(m.group(1)) if m else None

    def builtin_kind() -> Optional[str]:
        m = re.search(r"\bkind\s*=\s*#vc4\.builtin_kind<([^>]+)>", entry)
        return m.group(1) if m else None

    result: Dict[str, Any] = {}
    for field in ["name", "kind", "direction", "type", "elem_type", "materialization"]:
        val = str_field(field)
        if val is not None:
            result[field] = val
    bkind = builtin_kind()
    if bkind is not None:
        result["kind"] = bkind
    idx = int_field("uniform_index")
    if idx is not None:
        result["uniform_index"] = idx
    return result


def extract_launch_abi(input_mlir: Path) -> Dict[str, Any]:
    text = read_text(input_mlir)
    abi_text = extract_balanced_after(text, '"vc4.launch_abi"', "{", "}")
    if abi_text is None:
        # Many tests spell the attribute inside a larger attributes dict; fall back
        # to the first public_name-bearing dictionary-ish slice.
        m = re.search(r'"vc4\.launch_abi"\s*=\s*\{', text)
        if m:
            abi_text = extract_balanced_after(text[m.start() :], '"vc4.launch_abi"', "{", "}")
    if abi_text is None:
        raise VerificationError(
            "input MLIR does not contain a vc4.launch_abi dictionary",
            details={"input_mlir": str(input_mlir)},
        )
    public = re.search(r"\bpublic_name\s*=\s*\"([^\"]+)\"", abi_text)
    uniform_words = re.search(r"\buniform_words_per_qpu\s*=\s*(\d+)\s*:\s*i32", abi_text)
    tail_policy = re.search(r"\btail_policy\s*=\s*\"([^\"]+)\"", abi_text)
    args_text = extract_balanced_after(abi_text, "args", "[", "]") or ""
    builtins_text = extract_balanced_after(abi_text, "builtins", "[", "]") or ""
    args = [parse_launch_abi_entry(e) for e in split_top_level_dicts(args_text)]
    builtins = [parse_launch_abi_entry(e) for e in split_top_level_dicts(builtins_text)]
    indices = []
    for entry in args + builtins:
        if "uniform_index" in entry:
            indices.append(int(entry["uniform_index"]))
    return {
        "public_name": public.group(1) if public else None,
        "tail_policy": tail_policy.group(1) if tail_policy else None,
        "uniform_words_per_qpu": int(uniform_words.group(1)) if uniform_words else None,
        "args": args,
        "builtins": builtins,
        "uniform_indices": indices,
    }


def find_c_prototypes(text: str, name: str) -> List[str]:
    # Good enough for generated C headers: capture prototypes/definitions whose
    # declaration starts at a C-ish return type and names the public function.
    pattern = rf"(?:^|\n)\s*(?:extern\s+)?(?:int|void|uint32_t|unsigned|long|struct\s+\w+\s*\*)[\w\s\*]*\b{re.escape(name)}\s*\(([^;{{}}]*)\)\s*(?:;|\{{)"
    return [m.group(0).strip() for m in re.finditer(pattern, text)]


def prototype_params(proto: str) -> List[str]:
    m = re.search(r"\((.*)\)", proto, flags=re.S)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw or raw == "void":
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def param_name(param: str) -> Optional[str]:
    # Strip array suffix and pick the final identifier.
    p = re.sub(r"\[[^\]]*\]", "", param).strip()
    names = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", p)
    if not names:
        return None
    # Skip common type keywords; the final remaining identifier is the variable.
    keywords = {"const", "volatile", "struct", "enum", "union", "unsigned", "signed", "int", "void", "uint32_t", "size_t", "float", "double", "long", "short"}
    filtered = [n for n in names if n not in keywords]
    return filtered[-1] if filtered else names[-1]


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------


def make_success(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any], *, message: str = "", details: Optional[Dict[str, Any]] = None, duration: float = 0.0) -> VerificationResult:
    return VerificationResult(
        ok=True,
        slice_id=slice_id,
        verification_id=str(v.get("id", "<unnamed>")),
        mechanism=str(v.get("mechanism", "<unknown>")),
        description=str(v.get("description", "")),
        category=str(v.get("category", v.get("mechanism", "verification"))),
        route_hint=str(v.get("route_hint", "gpt_pro")),
        required=bool(v.get("required", True)),
        duration_sec=duration,
        message=message,
        details=details,
    )


def make_failure(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any], message: str, *, expected: Any = None, actual: Any = None, details: Optional[Dict[str, Any]] = None, command_result: Optional[CommandResult] = None, duration: float = 0.0) -> VerificationResult:
    kwargs: Dict[str, Any] = {}
    timed_out = False
    timeout_value: Optional[int] = None
    category = str(v.get("category", v.get("mechanism", "verification_failed")))
    route_hint = str(v.get("route_hint", "gpt_pro"))

    if command_result is not None:
        kwargs.update(
            command=command_result.argv,
            exit_code=command_result.exit_code,
            cwd=str(command_result.cwd),
            log_path=str(command_result.log_path) if command_result.log_path else None,
            stdout_tail=tail(command_result.stdout),
            stderr_tail=tail(command_result.stderr),
        )
        duration = command_result.duration_sec
        if command_result.timed_out:
            timed_out = True
            timeout_value = command_result.timeout_sec
            hardware_like = is_hardware_like_verification(v)
            category = "hardware_timeout" if hardware_like else "command_timeout"
            route_hint = "gpt_pro" if hardware_like else route_hint
            if hardware_like:
                message = (
                    f"hardware verification command timed out after "
                    f"{timeout_value}s; likely pi-install/serial hardware hang"
                )
            else:
                message = f"verification command timed out after {timeout_value}s"

            timeout_actual = {
                "exit_code": command_result.exit_code,
                "timeout_sec": timeout_value,
                "cmd": command_result.argv,
            }
            if actual is None:
                actual = timeout_actual
            elif isinstance(actual, dict):
                actual = {**actual, **timeout_actual}

            timeout_details = {
                "stdout_tail": tail(command_result.stdout),
                "stderr_tail": tail(command_result.stderr),
            }
            if details is None:
                details = timeout_details
            else:
                details = {**details, **timeout_details}

    return VerificationResult(
        ok=False,
        slice_id=slice_id,
        verification_id=str(v.get("id", "<unnamed>")),
        mechanism=str(v.get("mechanism", "<unknown>")),
        description=str(v.get("description", "")),
        category=category,
        route_hint=route_hint,
        required=bool(v.get("required", True)),
        duration_sec=duration,
        message=message,
        expected=expected,
        actual=actual,
        details=details,
        timed_out=timed_out,
        timeout_sec=timeout_value,
        **kwargs,
    )

def skip_result(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any], reason: str) -> VerificationResult:
    return VerificationResult(
        ok=True,
        slice_id=slice_id,
        verification_id=str(v.get("id", "<unnamed>")),
        mechanism=str(v.get("mechanism", "<unknown>")),
        description=str(v.get("description", "")),
        category=str(v.get("category", v.get("mechanism", "verification_skipped"))),
        route_hint=str(v.get("route_hint", "gpt_pro")),
        required=bool(v.get("required", True)),
        skipped=True,
        skip_reason=reason,
        message=reason,
    )


def mechanism_source_products(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    files = [normalize_repo_relpath(str(x)) for x in as_list(v.get("files"))]
    globs = [normalize_repo_relpath(str(x)) for x in as_list(v.get("globs"))]
    min_glob_matches = v.get("min_glob_matches", {})
    missing_files = [p for p in files if not ctx.repo_path(p).exists()]
    glob_matches: Dict[str, List[str]] = {}
    missing_globs: List[Dict[str, Any]] = []
    for pattern in globs:
        matches = glob_repo(ctx.repo, pattern)
        glob_matches[pattern] = matches
        required = int(min_glob_matches.get(pattern, 1)) if isinstance(min_glob_matches, dict) else 1
        if len(matches) < required:
            missing_globs.append({"glob": pattern, "required_min": required, "actual_count": len(matches)})
    if missing_files or missing_globs:
        return make_failure(
            ctx,
            slice_id,
            v,
            "required source products are missing",
            expected={"files": files, "globs": globs},
            actual={"missing_files": missing_files, "missing_globs": missing_globs, "glob_matches": glob_matches},
            duration=time.time() - started,
        )
    return make_success(ctx, slice_id, v, message="source products present", details={"files": files, "glob_matches": glob_matches}, duration=time.time() - started)


def mechanism_forbidden_absent(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    globs = [normalize_repo_relpath(str(x)) for x in as_list(v.get("globs"))]
    mode = str(v.get("mode", "filesystem"))
    found: Dict[str, List[str]] = {}
    if mode == "git-status":
        argv = ["git", "status", "--short", "--"] + globs
        result = ctx.run_command(argv, cwd=ctx.repo, log_path=ctx.command_log_path(slice_id, str(v.get("id", "forbidden_absent"))))
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        if lines:
            return make_failure(ctx, slice_id, v, "forbidden paths have git status changes", expected={"clean_globs": globs}, actual={"status_lines": lines}, command_result=result)
        return make_success(ctx, slice_id, v, message="forbidden git-status globs clean", duration=result.duration_sec)
    for pattern in globs:
        matches = glob_repo(ctx.repo, pattern)
        if matches:
            found[pattern] = matches
    if found:
        return make_failure(ctx, slice_id, v, "forbidden files exist", expected={"absent_globs": globs}, actual={"matches": found}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="forbidden globs absent", duration=time.time() - started)


def mechanism_tool_available(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    tools = [str(x) for x in as_list(v.get("tools"))]
    any_tools = [str(x) for x in as_list(v.get("any_tools"))]
    found: Dict[str, str] = {}
    missing: List[str] = []
    for tool in tools:
        try:
            found[tool] = ctx.resolve_tool(tool)
        except VerificationError:
            missing.append(tool)
    any_found: Optional[Tuple[str, str]] = None
    if any_tools:
        for tool in any_tools:
            try:
                any_found = (tool, ctx.resolve_tool(tool))
                break
            except VerificationError:
                pass
        if any_found is None:
            missing.append("one of: " + ", ".join(any_tools))
        else:
            found[any_found[0]] = any_found[1]
    if missing:
        return make_failure(ctx, slice_id, v, "required tools are missing", expected={"tools": tools, "any_tools": any_tools}, actual={"missing": missing, "found": found}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="tools available", details={"found": found}, duration=time.time() - started)


def expand_command(ctx: VerifierContext, argv: Sequence[Any], v: Mapping[str, Any]) -> List[str]:
    mapping = {
        "{repo}": str(ctx.repo),
        "{build_dir}": str(ctx.build_dir),
        "{state_root}": str(ctx.state_root),
        "{python}": sys.executable,
        "{python3}": sys.executable,
    }
    out: List[str] = []
    for x in argv:
        s = str(x)
        for key, val in mapping.items():
            s = s.replace(key, val)
        out.append(s)
    return out


def mechanism_command(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    argv = v.get("argv") or v.get("command")
    if isinstance(argv, str):
        # Shell-like strings are intentionally unsupported to keep specs typed.
        return make_failure(ctx, slice_id, v, "command.argv must be a JSON array, not a shell string", expected="array", actual=argv)
    if not isinstance(argv, list) or not argv:
        return make_failure(ctx, slice_id, v, "command verification requires non-empty argv")
    cmd = expand_command(ctx, argv, v)
    cwd = ctx.repo_path(str(v.get("cwd", "."))) if v.get("cwd") else ctx.repo
    log_path = ctx.command_log_path(slice_id, str(v.get("id", "command")))
    result = ctx.run_command(cmd, cwd=cwd, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    expected_exit = int(v.get("expect_exit_code", 0))
    stdout_contains = [str(x) for x in as_list(v.get("stdout_contains"))]
    stderr_contains = [str(x) for x in as_list(v.get("stderr_contains"))]
    missing_stdout = [x for x in stdout_contains if x not in result.stdout]
    missing_stderr = [x for x in stderr_contains if x not in result.stderr]
    if result.exit_code != expected_exit or missing_stdout or missing_stderr:
        return make_failure(
            ctx,
            slice_id,
            v,
            "command verification failed",
            expected={"exit_code": expected_exit, "stdout_contains": stdout_contains, "stderr_contains": stderr_contains},
            actual={"exit_code": result.exit_code, "missing_stdout": missing_stdout, "missing_stderr": missing_stderr},
            command_result=result,
        )
    return make_success(ctx, slice_id, v, message="command passed", duration=result.duration_sec, details={"command": result.argv, "log_path": str(result.log_path) if result.log_path else None})


def mechanism_build(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    target = str(v.get("target", ""))
    if not target:
        return make_failure(ctx, slice_id, v, "build verification requires target")
    argv = ["ninja", "-C", str(ctx.build_dir), target]
    log_path = ctx.command_log_path(slice_id, str(v.get("id", f"build_{target}")))
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    if not result.ok:
        return make_failure(ctx, slice_id, v, f"build target failed: {target}", expected={"target": target}, actual={"exit_code": result.exit_code}, command_result=result)
    return make_success(ctx, slice_id, v, message=f"build target passed: {target}", duration=result.duration_sec, details={"log_path": str(log_path)})


def mechanism_lit(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    root = str(v.get("root", ""))
    if not root:
        return make_failure(ctx, slice_id, v, "lit verification requires root")
    lit_tool = str(v.get("tool", "lit"))
    try:
        lit = ctx.resolve_tool(lit_tool)
    except VerificationError:
        lit = ctx.resolve_tool("llvm-lit")
    argv = [lit, "-sv", str(ctx.repo_path(root))]
    log_path = ctx.command_log_path(slice_id, str(v.get("id", "lit")))
    result = ctx.run_command(argv, cwd=ctx.repo / "compiler/build/test" if (ctx.repo / "compiler/build/test").exists() else ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    if not result.ok:
        return make_failure(ctx, slice_id, v, "lit suite failed", expected={"root": root}, actual={"exit_code": result.exit_code}, command_result=result)
    return make_success(ctx, slice_id, v, message="lit suite passed", duration=result.duration_sec, details={"root": root, "log_path": str(log_path)})


def mechanism_vc4_codegen_generate(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    input_path = str(v.get("input", ""))
    bundle = str(v.get("bundle", ""))
    if not input_path or not bundle:
        return make_failure(ctx, slice_id, v, "vc4_codegen_generate requires input and bundle")
    input_abs = ctx.repo_path(input_path)
    bundle_abs = ctx.repo_path(bundle) if not bundle.startswith(".vc4_auto/") else ctx.repo / bundle
    if not input_abs.exists():
        return make_failure(ctx, slice_id, v, "codegen input does not exist", expected=input_path, actual={"exists": False})
    if bool(v.get("clean", True)) and bundle_abs.exists() and not ctx.dry_run:
        shutil.rmtree(bundle_abs)
    if not ctx.dry_run:
        bundle_abs.mkdir(parents=True, exist_ok=True)
    tool = str(v.get("tool", "vc4-codegen"))
    try:
        exe = ctx.resolve_tool(tool)
    except VerificationError as e:
        return make_failure(ctx, slice_id, v, str(e), expected={"tool": tool}, actual=e.details)
    argv = [exe, str(input_abs), "--emit-bundle", str(bundle_abs)]
    log_path = ctx.command_log_path(slice_id, str(v.get("id", "vc4_codegen_generate")))
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    required_files = [str(x) for x in as_list(v.get("required_files"))] or ["kernel.qasm", "kernel_launch.c", "kernel_launch.h", "manifest.json"]
    missing = [f for f in required_files if not (bundle_abs / f).exists()]
    if not result.ok or missing:
        return make_failure(ctx, slice_id, v, "vc4-codegen generation failed or produced incomplete bundle", expected={"required_files": required_files}, actual={"exit_code": result.exit_code, "missing": missing}, command_result=result)
    return make_success(ctx, slice_id, v, message="vc4-codegen bundle generated", duration=result.duration_sec, details={"bundle": ctx.rel(bundle_abs), "log_path": str(log_path)})


def mechanism_artifact_bundle(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    bundle = str(v.get("bundle", ""))
    if not bundle:
        return make_failure(ctx, slice_id, v, "artifact_bundle requires bundle")
    bundle_abs = ctx.repo / bundle if bundle.startswith(".vc4_auto/") else ctx.repo_path(bundle)
    required_files = [str(x) for x in as_list(v.get("required_files"))] or ["kernel.qasm", "kernel_launch.c", "kernel_launch.h", "manifest.json"]
    missing = [f for f in required_files if not (bundle_abs / f).exists()]
    manifest_data: Any = None
    manifest_errors: List[str] = []
    manifest_path = bundle_abs / "manifest.json"
    if manifest_path.exists():
        try:
            manifest_data = json.loads(read_text(manifest_path))
        except json.JSONDecodeError as e:
            manifest_errors.append(str(e))
    else:
        manifest_errors.append("manifest.json missing")
    manifest_required_keys = [str(x) for x in as_list(v.get("manifest_required_keys"))]
    missing_keys: List[str] = []
    if isinstance(manifest_data, dict):
        missing_keys = [key for key in manifest_required_keys if key not in manifest_data]
        manifest_requires = v.get("manifest_requires") or {}
        mismatches = {k: {"expected": val, "actual": manifest_data.get(k)} for k, val in manifest_requires.items() if manifest_data.get(k) != val} if isinstance(manifest_requires, dict) else {}
    else:
        mismatches = {}
    if missing or manifest_errors or missing_keys or mismatches:
        return make_failure(ctx, slice_id, v, "artifact bundle contract failed", expected={"required_files": required_files, "manifest_required_keys": manifest_required_keys, "manifest_requires": v.get("manifest_requires", {})}, actual={"missing_files": missing, "manifest_errors": manifest_errors, "missing_keys": missing_keys, "mismatches": mismatches}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="artifact bundle contract passed", duration=time.time() - started, details={"bundle": ctx.rel(bundle_abs)})


def mechanism_qasm_text(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    path = str(v.get("path", ""))
    if not path:
        return make_failure(ctx, slice_id, v, "qasm_text requires path")
    qasm_path = ctx.repo / path if path.startswith(".vc4_auto/") else ctx.repo_path(path)
    if not qasm_path.exists():
        return make_failure(ctx, slice_id, v, "qasm file does not exist", expected=path, actual={"exists": False})
    text = read_text(qasm_path)
    contains = [str(x) for x in as_list(v.get("contains"))]
    not_contains = [str(x) for x in as_list(v.get("not_contains"))]
    ordered = [str(x) for x in as_list(v.get("ordered_contains"))]
    missing = [x for x in contains if x not in text]
    forbidden = [x for x in not_contains if x in text]
    ordered_missing: List[str] = []
    pos = 0
    for token in ordered:
        idx = text.find(token, pos)
        if idx < 0:
            ordered_missing.append(token)
        else:
            pos = idx + len(token)
    if missing or forbidden or ordered_missing:
        return make_failure(ctx, slice_id, v, "qasm text expectations failed", expected={"contains": contains, "not_contains": not_contains, "ordered_contains": ordered}, actual={"missing": missing, "forbidden_present": forbidden, "ordered_missing": ordered_missing, "text_tail": tail(text)}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="qasm text expectations passed", duration=time.time() - started)


def mechanism_vc4asm_assemble(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    qasm = str(v.get("qasm", ""))
    if not qasm:
        return make_failure(ctx, slice_id, v, "vc4asm_assemble requires qasm")
    qasm_abs = ctx.repo / qasm if qasm.startswith(".vc4_auto/") else ctx.repo_path(qasm)
    if not qasm_abs.exists():
        return make_failure(ctx, slice_id, v, "qasm file missing", expected=qasm, actual={"exists": False})
    out_c = str(v.get("out_c", str(qasm_abs.with_name("kernelshader.c"))))
    out_h = str(v.get("out_h", str(qasm_abs.with_name("kernelshader.h"))))
    out_c_abs = Path(out_c) if out_c.startswith("/") else (ctx.repo / out_c if out_c.startswith(".vc4_auto/") else ctx.repo_path(out_c))
    out_h_abs = Path(out_h) if out_h.startswith("/") else (ctx.repo / out_h if out_h.startswith(".vc4_auto/") else ctx.repo_path(out_h))
    if not ctx.dry_run:
        out_c_abs.parent.mkdir(parents=True, exist_ok=True)
        out_h_abs.parent.mkdir(parents=True, exist_ok=True)
    try:
        exe = ctx.resolve_tool(str(v.get("tool", "vc4asm")))
    except VerificationError as e:
        return make_failure(ctx, slice_id, v, str(e), expected={"tool": "vc4asm"}, actual=e.details)
    cwd = Path(v.get("cwd", qasm_abs.parent))
    if not cwd.is_absolute():
        cwd = ctx.repo / cwd
    argv = [exe, "-c", str(out_c_abs), "-h", str(out_h_abs), str(qasm_abs)]
    log_path = ctx.command_log_path(slice_id, str(v.get("id", "vc4asm_assemble")))
    result = ctx.run_command(argv, cwd=cwd, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    missing = [str(p) for p in [out_c_abs, out_h_abs] if not p.exists()]
    if not result.ok or missing:
        return make_failure(ctx, slice_id, v, "vc4asm assembly failed", expected={"outputs": [str(out_c_abs), str(out_h_abs)]}, actual={"missing_outputs": missing, "exit_code": result.exit_code}, command_result=result)
    return make_success(ctx, slice_id, v, message="vc4asm assembled qasm", duration=result.duration_sec, details={"outputs": [ctx.rel(out_c_abs), ctx.rel(out_h_abs)]})


def mechanism_c_syntax(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    files = [str(x) for x in as_list(v.get("files"))]
    if not files:
        return make_failure(ctx, slice_id, v, "c_syntax requires files")
    cc = str(v.get("cc", os.environ.get("CC", "/usr/bin/cc")))
    include_dirs = [str(ctx.repo_path(str(x))) for x in as_list(v.get("include_dirs"))]
    flags = [str(x) for x in as_list(v.get("flags"))] or ["-std=c11", "-fsyntax-only"]
    argv = [cc] + flags + [f"-I{d}" for d in include_dirs] + [str(ctx.repo_path(p) if not p.startswith(".vc4_auto/") else ctx.repo / p) for p in files]
    log_path = ctx.command_log_path(slice_id, str(v.get("id", "c_syntax")))
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    if not result.ok:
        return make_failure(ctx, slice_id, v, "C syntax check failed", expected={"files": files}, actual={"exit_code": result.exit_code}, command_result=result)
    return make_success(ctx, slice_id, v, message="C syntax check passed", duration=result.duration_sec, details={"log_path": str(log_path)})


def mechanism_candidate_phase(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    name = str(v.get("name", ""))
    phase = str(v.get("phase", ""))
    if not name or not phase:
        return make_failure(ctx, slice_id, v, "candidate_phase requires name and phase")
    script = ctx.repo_path(str(v.get("script", "compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh")))
    if not script.exists():
        return make_failure(ctx, slice_id, v, "candidate runner script missing", expected=str(script), actual={"exists": False})
    argv = ["bash", str(script), name, phase]
    log_path = ctx.command_log_path(slice_id, str(v.get("id", f"candidate_{phase}_{name}")))
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    if not result.ok:
        return make_failure(ctx, slice_id, v, f"candidate phase failed: {name} {phase}", expected={"name": name, "phase": phase}, actual={"exit_code": result.exit_code}, command_result=result)
    return make_success(ctx, slice_id, v, message=f"candidate phase passed: {name} {phase}", duration=result.duration_sec, details={"log_path": str(log_path)})


def mechanism_hardware_run(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    if ctx.no_hardware or bool(v.get("skip_when_no_hardware", False)) and ctx.no_hardware:
        return skip_result(ctx, slice_id, v, "hardware verification skipped by --no-hardware")
    fixture = str(v.get("fixture", ""))
    side = str(v.get("side", "reference"))
    if not fixture:
        return make_failure(ctx, slice_id, v, "hardware_run requires fixture")
    if side == "reference":
        run_sh = ctx.repo_path(f"compiler/test/CodeGen/VC4/Hardware/Run/{fixture}/run.sh")
        argv = ["bash", "run.sh"] if v.get("use_fixture_run_sh", True) else ["bash", str(ctx.repo_path("compiler/test/CodeGen/VC4/Support/run_hardware_test.sh")), str(run_sh.parent), "reference"]
        cwd = run_sh.parent
    elif side == "candidate":
        script = ctx.repo_path(str(v.get("script", "compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh")))
        argv = ["bash", str(script), fixture, "run"]
        cwd = ctx.repo
    else:
        return make_failure(ctx, slice_id, v, "hardware_run side must be reference or candidate", expected=["reference", "candidate"], actual=side)
    log_path = ctx.command_log_path(slice_id, str(v.get("id", f"hardware_{side}_{fixture}")))
    result = ctx.run_command(argv, cwd=cwd, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    if not result.ok:
        return make_failure(ctx, slice_id, v, f"hardware run failed: {fixture} {side}", expected={"fixture": fixture, "side": side}, actual={"exit_code": result.exit_code}, command_result=result)
    return make_success(ctx, slice_id, v, message=f"hardware run passed: {fixture} {side}", duration=result.duration_sec, details={"log_path": str(log_path)})


def mechanism_expected_json_result(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    expected = str(v.get("expected", ""))
    log = str(v.get("log", ""))
    if not expected or not log:
        return make_failure(ctx, slice_id, v, "expected_json_result requires expected and log")
    script = ctx.repo_path(str(v.get("script", "compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py")))
    expected_abs = ctx.repo_path(expected)
    log_abs = ctx.repo / log if log.startswith(".vc4_auto/") else ctx.repo_path(log)
    if not expected_abs.exists():
        return make_failure(ctx, slice_id, v, "expected.json missing", expected=expected, actual={"exists": False})
    if not log_abs.exists():
        return make_failure(ctx, slice_id, v, "hardware log missing", expected=log, actual={"exists": False})
    argv = [sys.executable, str(script), str(expected_abs), str(log_abs)]
    log_path = ctx.command_log_path(slice_id, str(v.get("id", "expected_json_result")))
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    if not result.ok:
        return make_failure(ctx, slice_id, v, "expected.json result check failed", expected={"expected_json": expected, "run_log": log}, actual={"exit_code": result.exit_code}, command_result=result)
    return make_success(ctx, slice_id, v, message="expected.json result check passed", duration=result.duration_sec, details={"log_path": str(log_path)})


def mechanism_reference_immutable(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    globs = [str(x) for x in as_list(v.get("globs"))]
    if not globs:
        return make_failure(ctx, slice_id, v, "reference_immutable requires globs")
    # Use git status so committed reference files don't fail, but local mutations do.
    argv = ["git", "status", "--short", "--"] + globs
    log_path = ctx.command_log_path(slice_id, str(v.get("id", "reference_immutable")))
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=int(v.get("timeout_sec", 300)), log_path=log_path)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not result.ok or lines:
        return make_failure(ctx, slice_id, v, "immutable reference/oracle files have local changes", expected={"clean_globs": globs}, actual={"status_lines": lines, "exit_code": result.exit_code}, command_result=result)
    return make_success(ctx, slice_id, v, message="immutable reference/oracle files clean", duration=result.duration_sec)


def mechanism_launch_abi_public_api(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    input_mlir = ctx.repo_path(str(v.get("input_mlir", "")))
    header = ctx.repo / str(v.get("header", "")) if str(v.get("header", "")).startswith(".vc4_auto/") else ctx.repo_path(str(v.get("header", "")))
    source = ctx.repo / str(v.get("source", "")) if str(v.get("source", "")).startswith(".vc4_auto/") else ctx.repo_path(str(v.get("source", ""))) if v.get("source") else None
    manifest = ctx.repo / str(v.get("manifest", "")) if str(v.get("manifest", "")).startswith(".vc4_auto/") else ctx.repo_path(str(v.get("manifest", ""))) if v.get("manifest") else None
    for required_path, label in [(input_mlir, "input_mlir"), (header, "header")]:
        if not required_path.exists():
            return make_failure(ctx, slice_id, v, f"{label} missing", expected=str(required_path), actual={"exists": False})
    abi = extract_launch_abi(input_mlir)
    public_name = str(v.get("public_name") or abi.get("public_name") or "")
    if not public_name:
        return make_failure(ctx, slice_id, v, "launch ABI public_name missing", expected="non-empty public_name", actual=abi)
    header_text = read_text(header)
    source_text = read_text(source) if source and source.exists() else ""
    manifest_data: Any = None
    if manifest and manifest.exists():
        try:
            manifest_data = json.loads(read_text(manifest))
        except json.JSONDecodeError:
            manifest_data = None
    prototypes = find_c_prototypes(header_text + "\n" + source_text, public_name)
    params = []
    if prototypes:
        params = prototype_params(prototypes[0])
    names = [n for n in (param_name(p) for p in params) if n]
    forbidden_public_names = [str(x) for x in as_list(v.get("forbid_public_param_names"))] or ["qpu_id", "num_qpus", "uniforms", "uniform_words", "uniform_ptrs"]
    forbidden_mentions = [name for name in forbidden_public_names if name in names]
    required_arg_names = [entry.get("name") for entry in abi.get("args", []) if entry.get("name")]
    missing_args = [name for name in required_arg_names if name not in names and not re.search(rf"\b{re.escape(name)}\b", "\n".join(prototypes))]
    raw_uniform_api = []
    if bool(v.get("forbid_raw_uniform_public_api", True)):
        for proto in prototypes:
            if re.search(r"\buniform\w*\b", proto) and public_name in proto:
                raw_uniform_api.append(proto)
    manifest_mismatch = None
    if isinstance(manifest_data, dict) and manifest_data.get("public_name") not in (None, public_name):
        manifest_mismatch = {"expected": public_name, "actual": manifest_data.get("public_name")}
    if not prototypes or forbidden_mentions or missing_args or raw_uniform_api or manifest_mismatch:
        return make_failure(
            ctx,
            slice_id,
            v,
            "launch ABI public API contract failed",
            expected={"public_name": public_name, "required_arg_names": required_arg_names, "forbidden_public_names": forbidden_public_names},
            actual={"prototypes": prototypes, "param_names": names, "forbidden_mentions": forbidden_mentions, "missing_args": missing_args, "raw_uniform_api": raw_uniform_api, "manifest_mismatch": manifest_mismatch},
            duration=time.time() - started,
        )
    return make_success(ctx, slice_id, v, message="launch ABI public API contract passed", duration=time.time() - started, details={"public_name": public_name, "prototype": prototypes[0], "launch_abi": abi})


def mechanism_launch_abi_uniform_layout(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    input_mlir = ctx.repo_path(str(v.get("input_mlir", "")))
    source = ctx.repo / str(v.get("source", "")) if str(v.get("source", "")).startswith(".vc4_auto/") else ctx.repo_path(str(v.get("source", "")))
    if not input_mlir.exists():
        return make_failure(ctx, slice_id, v, "input MLIR missing", expected=str(input_mlir), actual={"exists": False})
    if not source.exists():
        return make_failure(ctx, slice_id, v, "launcher source missing", expected=str(source), actual={"exists": False})
    abi = extract_launch_abi(input_mlir)
    text = read_text(source)
    words = abi.get("uniform_words_per_qpu")
    indices = sorted(set(int(x) for x in abi.get("uniform_indices", []) if isinstance(x, int)))
    expected_dense = list(range(int(words))) if isinstance(words, int) else []
    dense_ok = indices == expected_dense
    missing_index_writes: List[int] = []
    if bool(v.get("require_index_writes", True)):
        for idx in indices:
            # Accept array writes, helper calls, or manifest-ish generated constants.
            patterns = [
                rf"\[[\s\(]*{idx}[\s\)]*\]",
                rf"uniform_index[^0-9-]*{idx}\b",
                rf"slot[^0-9-]*{idx}\b",
            ]
            if not any(re.search(p, text) for p in patterns):
                missing_index_writes.append(idx)
    forbidden_public_names = [str(x) for x in as_list(v.get("forbid_public_param_names"))] or ["qpu_id", "num_qpus", "uniforms", "uniform_words", "uniform_ptrs"]
    public_name = abi.get("public_name")
    prototypes = find_c_prototypes(text, str(public_name)) if public_name else []
    params = prototype_params(prototypes[0]) if prototypes else []
    names = [n for n in (param_name(p) for p in params) if n]
    forbidden_public = [name for name in forbidden_public_names if name in names]
    if not dense_ok or missing_index_writes or forbidden_public:
        return make_failure(
            ctx,
            slice_id,
            v,
            "launch ABI uniform layout contract failed",
            expected={"dense_uniform_indices": expected_dense, "forbidden_public_param_names": forbidden_public_names},
            actual={"indices": indices, "missing_index_writes": missing_index_writes, "public_param_names": names, "forbidden_public": forbidden_public},
            duration=time.time() - started,
        )
    return make_success(ctx, slice_id, v, message="launch ABI uniform layout contract passed", duration=time.time() - started, details={"launch_abi": abi, "public_param_names": names})



# ---------------------------------------------------------------------------
# Milestone 2 program-bundle verifier mechanisms
# ---------------------------------------------------------------------------


def resolve_repo_or_auto_path(ctx: VerifierContext, value: str) -> Path:
    value = str(value)
    if value.startswith(".vc4_auto/"):
        return ctx.repo / value
    return ctx.repo_path(value)


def read_json_file_checked(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise VerificationError(f"JSON file missing: {path}", details={"path": str(path), "exists": False})
    except json.JSONDecodeError as exc:
        raise VerificationError(f"invalid JSON in {path}: {exc}", details={"path": str(path), "error": str(exc)})


def bundle_path(ctx: VerifierContext, v: Mapping[str, Any]) -> Path:
    raw = str(v.get("bundle", ""))
    if not raw:
        raise VerificationError("verification requires bundle")
    return resolve_repo_or_auto_path(ctx, raw)


def manifest_from_bundle(ctx: VerifierContext, v: Mapping[str, Any]) -> tuple[Path, Dict[str, Any]]:
    b = bundle_path(ctx, v)
    manifest_rel = str(v.get("manifest", "manifest.json"))
    manifest_path = b / manifest_rel
    data = read_json_file_checked(manifest_path)
    if not isinstance(data, dict):
        raise VerificationError("manifest must be a JSON object", details={"manifest": str(manifest_path), "type": type(data).__name__})
    return manifest_path, data


def layout_from_bundle(ctx: VerifierContext, v: Mapping[str, Any]) -> tuple[Path, Dict[str, Any]]:
    b = bundle_path(ctx, v)
    layout_rel = str(v.get("layout", "layout.json"))
    layout_path = b / layout_rel
    data = read_json_file_checked(layout_path)
    if not isinstance(data, dict):
        raise VerificationError("layout must be a JSON object", details={"layout": str(layout_path), "type": type(data).__name__})
    return layout_path, data


def manifest_kernels(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    kernels = manifest.get("kernels")
    if isinstance(kernels, list):
        return [k for k in kernels if isinstance(k, dict)]
    # Legacy M1 compatibility: expose the old single-kernel manifest as a one
    # element logical kernel list so docs/explanations can still inspect it.
    if any(key in manifest for key in ("kernel", "public_name", "qasm_path")):
        public_name = str(manifest.get("public_name") or manifest.get("c_entry_point") or manifest.get("kernel") or "kernel")
        return [{
            "kernel_id": int(manifest.get("kernel_id", 0) or 0),
            "symbol_name": str(manifest.get("kernel") or public_name),
            "public_name": public_name,
            "qasm_path": str(manifest.get("qasm_path") or "kernel.qasm"),
            "code_symbol": str(manifest.get("code_symbol") or "kernelshader"),
            "scheduled_sink_ops": manifest.get("scheduled_sink_ops"),
            "uniform_words_per_request": manifest.get("uniform_words_per_request") or manifest.get("uniform_words_per_qpu"),
            "max_requests_per_wave": manifest.get("max_requests_per_wave") or 12,
            "tail_policy": manifest.get("tail_policy"),
            "schedule_mode": manifest.get("schedule_mode"),
            "args": manifest.get("args") or [],
            "builtins": manifest.get("builtins") or [],
            "resources": manifest.get("resources") or {},
        }]
    return []


def bundle_relative_path(bundle: Path, rel_value: str) -> Path:
    rel = normalize_repo_relpath(str(rel_value))
    if rel.startswith("/") or rel.startswith("../") or "/../" in rel:
        raise VerificationError("bundle-relative path is unsafe", details={"path": rel_value})
    return bundle / rel


def unique_field_errors(kernels: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Dict[str, List[Any]]:
    errors: Dict[str, List[Any]] = {}
    for field in fields:
        values = [k.get(field) for k in kernels]
        seen = set()
        dupes = []
        for value in values:
            key = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            if key in seen and value not in dupes:
                dupes.append(value)
            seen.add(key)
        if dupes:
            errors[field] = dupes
    return errors


def regex_list(value: Any) -> List[str]:
    return [str(x) for x in as_list(value)]


def function_body_for_name(text: str, name: str) -> str:
    pattern = re.compile(r"(?:^|\n)\s*(?:static\s+)?(?:inline\s+)?(?:int|void|uint32_t|unsigned|long|struct\s+\w+\s*\*|[A-Za-z_][A-Za-z0-9_\s\*]+)\s+" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", re.S)
    m = pattern.search(text)
    if not m:
        return ""
    start = m.end() - 1
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def public_launch_names_from_manifest(manifest: Mapping[str, Any]) -> List[str]:
    out = []
    for kernel in manifest_kernels(manifest):
        public = str(kernel.get("public_name") or "")
        if public:
            out.append(public + "_launch" if not public.endswith("_launch") else public)
    return out


def mechanism_manifest_schema(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    try:
        manifest_path, manifest = manifest_from_bundle(ctx, v)
    except VerificationError as exc:
        return make_failure(ctx, slice_id, v, str(exc), actual=exc.details, duration=time.time() - started)
    kernels = manifest_kernels(manifest)
    expected_schema = v.get("schema_version")
    expected_count = v.get("expected_kernel_count")
    min_count = int(v.get("min_kernel_count", 0) or 0)
    max_count = v.get("max_kernel_count")
    required_top = regex_list(v.get("required_top_level_keys")) or ["schema_version", "kind", "program_name", "target", "kernels"]
    required_kernel = regex_list(v.get("required_kernel_keys"))
    unique_fields = regex_list(v.get("unique_kernel_fields")) or ["kernel_id", "public_name", "code_symbol", "qasm_path"]
    forbid_top = regex_list(v.get("forbid_top_level_keys"))
    if manifest.get("schema_version") == 2:
        for legacy_key in ("kernel", "public_name", "qasm_path", "launch_abi"):
            if legacy_key not in forbid_top:
                forbid_top.append(legacy_key)
    missing_top = [k for k in required_top if k not in manifest]
    schema_mismatch = expected_schema is not None and manifest.get("schema_version") != expected_schema
    count_mismatch = expected_count is not None and len(kernels) != int(expected_count)
    min_mismatch = min_count and len(kernels) < min_count
    max_mismatch = max_count is not None and len(kernels) > int(max_count)
    missing_kernel: Dict[int, List[str]] = {}
    for i, kernel in enumerate(kernels):
        miss = [k for k in required_kernel if k not in kernel]
        if miss:
            missing_kernel[i] = miss
    dupes = unique_field_errors(kernels, unique_fields)
    forbidden_present = [k for k in forbid_top if k in manifest]
    unsafe_paths = []
    b = manifest_path.parent
    for kernel in kernels:
        qasm_path = kernel.get("qasm_path")
        if isinstance(qasm_path, str):
            try:
                p = bundle_relative_path(b, qasm_path).resolve()
                if not str(p).startswith(str(b.resolve())):
                    unsafe_paths.append(qasm_path)
            except VerificationError:
                unsafe_paths.append(qasm_path)
    if missing_top or schema_mismatch or count_mismatch or min_mismatch or max_mismatch or missing_kernel or dupes or forbidden_present or unsafe_paths:
        return make_failure(
            ctx, slice_id, v, "manifest schema contract failed",
            expected={"schema_version": expected_schema, "expected_kernel_count": expected_count, "required_top_level_keys": required_top, "required_kernel_keys": required_kernel, "unique_kernel_fields": unique_fields},
            actual={"manifest": ctx.rel(manifest_path), "kernel_count": len(kernels), "missing_top_level_keys": missing_top, "schema_mismatch": schema_mismatch, "count_mismatch": count_mismatch, "missing_kernel_keys": missing_kernel, "duplicate_fields": dupes, "forbidden_top_level_keys_present": forbidden_present, "unsafe_paths": unsafe_paths},
            duration=time.time() - started,
        )
    return make_success(ctx, slice_id, v, message="manifest schema contract passed", details={"manifest": ctx.rel(manifest_path), "kernel_count": len(kernels)}, duration=time.time() - started)


def mechanism_program_artifact_bundle(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    try:
        manifest_path, manifest = manifest_from_bundle(ctx, v)
    except VerificationError as exc:
        return make_failure(ctx, slice_id, v, str(exc), actual=exc.details, duration=time.time() - started)
    b = manifest_path.parent
    kernels = manifest_kernels(manifest)
    expected_count = v.get("expected_kernel_count")
    required_common = regex_list(v.get("required_common_files")) or ["kernel_launch.c", "kernel_launch.h", "manifest.json"]
    if bool(v.get("require_layout", "layout.json" in required_common)) and "layout.json" not in required_common:
        required_common.append("layout.json")
    missing_common = [rel for rel in required_common if not (b / rel).exists()]
    forbidden_common = regex_list(v.get("forbidden_common_files"))
    if manifest.get("schema_version") == 2 and "kernel.qasm" not in forbidden_common:
        forbidden_common.append("kernel.qasm")
    present_forbidden_common = [rel for rel in forbidden_common if (b / rel).exists()]
    qasm_missing = []
    qasm_forbidden: Dict[str, List[str]] = {}
    not_contains = regex_list(v.get("not_contains_in_qasm"))
    seen_qasm = set()
    duplicate_qasm = []
    if bool(v.get("require_qasm_files", True)):
        for kernel in kernels:
            qasm_rel = str(kernel.get("qasm_path", ""))
            if not qasm_rel:
                qasm_missing.append("<empty>")
                continue
            try:
                qasm_abs = bundle_relative_path(b, qasm_rel)
            except VerificationError:
                qasm_missing.append(qasm_rel)
                continue
            if qasm_rel in seen_qasm and qasm_rel not in duplicate_qasm:
                duplicate_qasm.append(qasm_rel)
            seen_qasm.add(qasm_rel)
            if not qasm_abs.exists():
                qasm_missing.append(qasm_rel)
                continue
            text = read_text(qasm_abs)
            bad = [token for token in not_contains if token in text]
            if bad:
                qasm_forbidden[qasm_rel] = bad
    if expected_count is not None and len(kernels) != int(expected_count):
        count_bad = True
    else:
        count_bad = False
    if missing_common or present_forbidden_common or qasm_missing or qasm_forbidden or duplicate_qasm or count_bad:
        return make_failure(ctx, slice_id, v, "program artifact bundle contract failed", expected={"required_common_files": required_common, "forbidden_common_files": forbidden_common, "expected_kernel_count": expected_count}, actual={"bundle": ctx.rel(b), "kernel_count": len(kernels), "missing_common": missing_common, "forbidden_common_present": present_forbidden_common, "missing_qasm": qasm_missing, "duplicate_qasm": duplicate_qasm, "qasm_forbidden": qasm_forbidden}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="program artifact bundle contract passed", details={"bundle": ctx.rel(b), "kernel_count": len(kernels)}, duration=time.time() - started)


def mechanism_all_qasm_assemble(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    try:
        manifest_path, manifest = manifest_from_bundle(ctx, v)
        exe = ctx.resolve_tool(str(v.get("tool", "vc4asm")))
    except VerificationError as exc:
        return make_failure(ctx, slice_id, v, str(exc), actual=exc.details, duration=time.time() - started)
    b = manifest_path.parent
    if v.get("out_dir"):
        out_dir = resolve_repo_or_auto_path(ctx, str(v.get("out_dir")))
    else:
        out_dir = b / "assembled"
    if not ctx.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    outputs = []
    for kernel in manifest_kernels(manifest):
        qasm_rel = str(kernel.get(str(v.get("qasm_field", "qasm_path")), ""))
        symbol = str(kernel.get(str(v.get("symbol_field", "code_symbol")), "")) or Path(qasm_rel).stem
        try:
            qasm_abs = bundle_relative_path(b, qasm_rel)
        except VerificationError as exc:
            failures.append({"kernel": kernel.get("public_name"), "error": str(exc), "qasm_path": qasm_rel})
            continue
        out_c = out_dir / f"{symbol}.c"
        out_h = out_dir / f"{symbol}.h"
        log_path = ctx.command_log_path(slice_id, f"{v.get('id', 'all_qasm_assemble')}_{symbol}")
        result = ctx.run_command([exe, "-c", str(out_c), "-h", str(out_h), str(qasm_abs)], cwd=qasm_abs.parent, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
        missing = [str(p) for p in (out_c, out_h) if not p.exists()]
        if not result.ok or missing:
            failures.append({"kernel": kernel.get("public_name"), "symbol": symbol, "exit_code": result.exit_code, "missing": missing, "log_path": str(log_path), "stdout_tail": tail(result.stdout), "stderr_tail": tail(result.stderr)})
        else:
            outputs.append({"kernel": kernel.get("public_name"), "c": ctx.rel(out_c), "h": ctx.rel(out_h)})
    if failures:
        return make_failure(ctx, slice_id, v, "one or more kernel qasm files failed to assemble", actual={"failures": failures}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="all qasm files assembled", details={"outputs": outputs}, duration=time.time() - started)


def mechanism_program_layout_contract(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    try:
        layout_path, layout = layout_from_bundle(ctx, v)
        _manifest_path, manifest = manifest_from_bundle(ctx, v)
    except VerificationError as exc:
        return make_failure(ctx, slice_id, v, str(exc), actual=exc.details, duration=time.time() - started)
    alignment = int(v.get("alignment_bytes", layout.get("alignment_bytes", 8)) or 8)
    regions = layout.get("regions")
    if not isinstance(regions, list):
        return make_failure(ctx, slice_id, v, "layout regions must be an array", actual={"regions_type": type(regions).__name__}, duration=time.time() - started)
    region_errors = []
    intervals = []
    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            region_errors.append({"index": idx, "error": "region is not object"})
            continue
        name = region.get("name", f"region_{idx}")
        offset = region.get("offset")
        size = region.get("size")
        if not isinstance(offset, int) or not isinstance(size, int) or offset < 0 or size < 0:
            region_errors.append({"name": name, "error": "offset/size must be non-negative integers", "offset": offset, "size": size})
            continue
        if alignment and offset % alignment != 0:
            region_errors.append({"name": name, "error": "offset misaligned", "offset": offset, "alignment": alignment})
        intervals.append((offset, offset + size, str(name)))
    overlaps = []
    if bool(v.get("require_no_overlaps", True)):
        for i, (a0, a1, an) in enumerate(intervals):
            for b0, b1, bn in intervals[i+1:]:
                if a0 < b1 and b0 < a1:
                    overlaps.append({"lhs": an, "rhs": bn, "lhs_range": [a0, a1], "rhs_range": [b0, b1]})
    heap_errors = []
    require_heap = bool(v.get("require_heap", False))
    heap_size = layout.get("heap_size_bytes", 0)
    if require_heap and (not isinstance(heap_size, int) or heap_size <= 0):
        heap_errors.append("heap_size_bytes must be positive")
    min_heap = v.get("min_heap_bytes")
    if min_heap is not None and (not isinstance(heap_size, int) or heap_size < int(min_heap)):
        heap_errors.append(f"heap_size_bytes must be at least {min_heap}")
    kernels = layout.get("kernels")
    if not isinstance(kernels, list):
        kernels = []
    expected_count = v.get("expected_kernel_count")
    manifest_kernel_count = len(manifest_kernels(manifest))
    count_bad = (expected_count is not None and len(kernels) != int(expected_count)) or (manifest_kernel_count and len(kernels) and len(kernels) != manifest_kernel_count)
    if region_errors or overlaps or heap_errors or count_bad:
        return make_failure(ctx, slice_id, v, "program layout contract failed", expected={"alignment_bytes": alignment, "require_no_overlaps": v.get("require_no_overlaps", True), "require_heap": require_heap, "expected_kernel_count": expected_count}, actual={"layout": ctx.rel(layout_path), "region_errors": region_errors, "overlaps": overlaps, "heap_errors": heap_errors, "layout_kernel_count": len(kernels), "manifest_kernel_count": manifest_kernel_count}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="program layout contract passed", details={"layout": ctx.rel(layout_path), "regions": len(regions), "heap_size_bytes": heap_size}, duration=time.time() - started)


def mechanism_generated_runtime_contract(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    try:
        manifest_path, manifest = manifest_from_bundle(ctx, v)
    except VerificationError as exc:
        manifest_path, manifest = (None, {})
    b = bundle_path(ctx, v)
    header_path = b / str(v.get("header", "kernel_launch.h"))
    source_path = b / str(v.get("source", "kernel_launch.c"))
    missing_files = [ctx.rel(p) for p in (header_path, source_path) if not p.exists()]
    if missing_files:
        return make_failure(ctx, slice_id, v, "generated runtime files missing", actual={"missing": missing_files}, duration=time.time() - started)
    header = read_text(header_path)
    source = read_text(source_path)
    combined = header + "\n" + source
    required_functions = regex_list(v.get("required_functions"))
    missing_functions = [fn for fn in required_functions if not re.search(rf"\b{re.escape(fn)}\s*\(", combined)]
    required_patterns = regex_list(v.get("required_patterns"))
    missing_patterns = [pat for pat in required_patterns if not re.search(pat, combined, flags=re.S)]
    forbidden_patterns = regex_list(v.get("forbidden_patterns"))
    forbidden_present = [pat for pat in forbidden_patterns if re.search(pat, combined, flags=re.S)]
    launch_forbidden = regex_list(v.get("forbidden_patterns_in_launch_functions"))
    launch_required = regex_list(v.get("required_patterns_in_launch_functions"))
    launch_violations = []
    for launch in public_launch_names_from_manifest(manifest):
        body = function_body_for_name(source, launch)
        if not body:
            launch_violations.append({"launch_function": launch, "missing_body": True})
            continue
        bad = [pat for pat in launch_forbidden if re.search(pat, body, flags=re.S)]
        miss = [pat for pat in launch_required if not re.search(pat, body, flags=re.S)]
        if bad or miss:
            launch_violations.append({"launch_function": launch, "forbidden_present": bad, "required_missing": miss})
    # Lightweight allocation policy checks by regex.  These deliberately fail
    # loudly until M2 runtime code grows stable event names/helpers.
    allocation_policy = v.get("allocation_policy") if isinstance(v.get("allocation_policy"), dict) else {}
    allocation_errors = []
    if allocation_policy.get("launch_reuses_resident_code"):
        for launch in public_launch_names_from_manifest(manifest):
            body = function_body_for_name(source, launch)
            if re.search(r"copy_.*code|code_upload|mem_alloc|mem_lock", body, flags=re.I|re.S):
                allocation_errors.append({"launch_function": launch, "error": "launch appears to allocate/copy/lock code"})
    if missing_functions or missing_patterns or forbidden_present or launch_violations or allocation_errors:
        return make_failure(ctx, slice_id, v, "generated runtime contract failed", expected={"required_functions": required_functions, "required_patterns": required_patterns, "forbidden_patterns": forbidden_patterns}, actual={"missing_functions": missing_functions, "missing_patterns": missing_patterns, "forbidden_present": forbidden_present, "launch_violations": launch_violations, "allocation_errors": allocation_errors}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="generated runtime contract passed", details={"header": ctx.rel(header_path), "source": ctx.rel(source_path)}, duration=time.time() - started)


def mechanism_heap_api_unit(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    sources = [resolve_repo_or_auto_path(ctx, str(x)) for x in as_list(v.get("source_files"))]
    stubs = [resolve_repo_or_auto_path(ctx, str(x)) for x in as_list(v.get("stub_files"))]
    missing = [ctx.rel(p) for p in sources + stubs if not p.exists()]
    if missing:
        return make_failure(ctx, slice_id, v, "heap API unit source files missing", actual={"missing": missing})
    cc = str(v.get("cc", os.environ.get("CC", "/usr/bin/cc")))
    include_dirs = [resolve_repo_or_auto_path(ctx, str(x)) for x in as_list(v.get("include_dirs"))]
    tmp_parent = ctx.state_root / "tmp"
    if not ctx.dry_run:
        tmp_parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="vc4_heap_unit_", dir=str(tmp_parent)))
    exe = tmpdir / "heap_unit"
    argv = [cc, "-std=c11", "-Wall", "-Wextra", "-O2"] + [f"-I{p}" for p in include_dirs] + [str(p) for p in sources + stubs] + ["-o", str(exe)]
    log_path = ctx.command_log_path(slice_id, str(v.get("id", "heap_api_unit_compile")))
    compile_result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    if not compile_result.ok:
        return make_failure(ctx, slice_id, v, "heap API unit compile failed", command_result=compile_result)
    run_log = ctx.command_log_path(slice_id, str(v.get("id", "heap_api_unit_run")) + "_run")
    run_result = ctx.run_command([str(exe)], cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=run_log)
    expected_exit = int(v.get("expect_exit_code", 0))
    contains = regex_list(v.get("stdout_contains"))
    missing_contains = [s for s in contains if s not in run_result.stdout]
    if run_result.exit_code != expected_exit or missing_contains:
        return make_failure(ctx, slice_id, v, "heap API unit run failed", expected={"exit_code": expected_exit, "stdout_contains": contains}, actual={"exit_code": run_result.exit_code, "missing_stdout": missing_contains}, command_result=run_result)
    return make_success(ctx, slice_id, v, message="heap API unit test passed", duration=compile_result.duration_sec + run_result.duration_sec, details={"compile_log": str(log_path), "run_log": str(run_log)})


def mechanism_launch_abi_contract(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    try:
        manifest_path, manifest = manifest_from_bundle(ctx, v)
    except VerificationError as exc:
        return make_failure(ctx, slice_id, v, str(exc), actual=exc.details, duration=time.time() - started)
    b = manifest_path.parent
    header_path = b / str(v.get("header", "kernel_launch.h"))
    source_path = b / str(v.get("source", "kernel_launch.c"))
    if not header_path.exists():
        return make_failure(ctx, slice_id, v, "launch ABI header missing", expected=str(header_path), actual={"exists": False}, duration=time.time() - started)
    header = read_text(header_path)
    source = read_text(source_path) if source_path.exists() else ""
    checks = v.get("kernels") if isinstance(v.get("kernels"), list) else []
    if not checks:
        checks = [{"public_name": k.get("public_name")} for k in manifest_kernels(manifest)]
    failures = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        public = str(check.get("public_name") or "")
        launch = str(check.get("launch_function") or (public + "_launch"))
        protos = find_c_prototypes(header + "\n" + source, launch)
        if not protos:
            failures.append({"public_name": public, "launch_function": launch, "error": "prototype not found"})
            continue
        params = prototype_params(protos[0])
        joined = "\n".join(params)
        if check.get("requires_program_handle") and not re.search(r"struct\s+vc4_program\s*\*\s*\w+", joined):
            failures.append({"launch_function": launch, "error": "missing struct vc4_program * parameter", "prototype": protos[0]})
        if check.get("requires_grid_block") and len(re.findall(r"\bvc4_dim3\b|struct\s+vc4_dim3", joined)) < 2:
            failures.append({"launch_function": launch, "error": "missing grid/block vc4_dim3 parameters", "prototype": protos[0]})
        if check.get("buffers_are_deviceptr"):
            kernel_meta = next((k for k in manifest_kernels(manifest) if k.get("public_name") == public), {})
            buffer_names = [a.get("name") for a in kernel_meta.get("args", []) if isinstance(a, dict) and a.get("kind") == "buffer"]
            for name in buffer_names:
                if name and re.search(rf"(?:\*\s*{re.escape(str(name))}\b|\b(?:float|uint32_t|int32_t|void)\s*\*\s*{re.escape(str(name))}\b)", joined):
                    failures.append({"launch_function": launch, "arg": name, "error": "buffer argument appears to be host pointer", "prototype": protos[0]})
                if name and not re.search(rf"\bvc4_deviceptr_t\s+{re.escape(str(name))}\b", joined):
                    failures.append({"launch_function": launch, "arg": name, "error": "buffer argument is not vc4_deviceptr_t", "prototype": protos[0]})
        if check.get("forbid_public_uniform_arrays") and re.search(r"\buniform\w*\b", protos[0]):
            failures.append({"launch_function": launch, "error": "public prototype exposes uniforms", "prototype": protos[0]})
        if check.get("forbid_implicit_host_copies"):
            body = function_body_for_name(source, launch)
            if re.search(r"vc4MemcpyHtoD|vc4MemcpyDtoH|memcpy\s*\(", body):
                failures.append({"launch_function": launch, "error": "launch body appears to perform host/device copy"})
    if failures:
        return make_failure(ctx, slice_id, v, "launch ABI contract failed", actual={"failures": failures}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="launch ABI contract passed", details={"manifest": ctx.rel(manifest_path)}, duration=time.time() - started)


def mechanism_runtime_event_log(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    log_raw = str(v.get("log_path", ""))
    fixture = str(v.get("fixture", ""))
    candidates = []
    if log_raw:
        candidates.append(resolve_repo_or_auto_path(ctx, log_raw))
    if fixture:
        candidates.append(ctx.repo_path(f"compiler/test/CodeGen/VC4/Hardware/Run/{fixture}/candidate/run.log"))
        candidates.extend(sorted(ctx.log_dir.glob(f"*hardware*candidate*{fixture}*.log")))
        candidates.extend(sorted(ctx.log_dir.glob(f"*{fixture}*hardware*candidate*.log")))
    log_path = next((p for p in candidates if p.exists()), None)
    if log_path is None:
        return make_failure(ctx, slice_id, v, "runtime event log not found", expected={"log_path": log_raw, "fixture": fixture}, actual={"candidates": [str(p) for p in candidates]}, duration=time.time() - started)
    text = read_text(log_path)
    contains = regex_list(v.get("contains"))
    not_contains = regex_list(v.get("not_contains"))
    missing = [s for s in contains if s not in text]
    forbidden = [s for s in not_contains if s in text]
    required_counters = v.get("required_counters") if isinstance(v.get("required_counters"), dict) else {}
    min_counters = v.get("min_counters") if isinstance(v.get("min_counters"), dict) else {}
    found_counters: Dict[str, int] = {}
    for name in set(required_counters) | set(min_counters):
        matches = re.findall(rf"\b{re.escape(str(name))}=(-?\d+)\b", text)
        if matches:
            found_counters[str(name)] = int(matches[-1])
    counter_errors = []
    for name, expected in required_counters.items():
        if found_counters.get(str(name)) != int(expected):
            counter_errors.append({"counter": name, "expected": int(expected), "actual": found_counters.get(str(name))})
    for name, minimum in min_counters.items():
        actual = found_counters.get(str(name))
        if actual is None or actual < int(minimum):
            counter_errors.append({"counter": name, "min": int(minimum), "actual": actual})
    if missing or forbidden or counter_errors:
        return make_failure(ctx, slice_id, v, "runtime event log contract failed", expected={"contains": contains, "not_contains": not_contains, "required_counters": required_counters, "min_counters": min_counters}, actual={"log_path": ctx.rel(log_path), "missing": missing, "forbidden_present": forbidden, "counter_errors": counter_errors, "found_counters": found_counters}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="runtime event log contract passed", details={"log_path": ctx.rel(log_path), "found_counters": found_counters}, duration=time.time() - started)


def expand_fixture_matrix(ctx: VerifierContext, v: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = v.get("fixtures")
    matrix_name = v.get("matrix")
    if raw is None and matrix_name:
        matrices = ctx.spec.get("fixture_matrices") or ctx.spec.get("matrices") or ctx.spec.get("defaults", {}).get("fixture_matrices", {})
        if isinstance(matrices, dict):
            raw = matrices.get(str(matrix_name), [])
    fixtures = []
    for item in as_list(raw):
        if isinstance(item, str):
            fixtures.append({"name": item})
        elif isinstance(item, dict):
            fixtures.append(dict(item))
    return fixtures


def mechanism_fixture_matrix(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    if ctx.no_hardware and bool(v.get("requires_hardware", False)):
        return skip_result(ctx, slice_id, v, "fixture matrix skipped by --no-hardware")
    started = time.time()
    fixtures = expand_fixture_matrix(ctx, v)
    phases = [str(x) for x in as_list(v.get("phases"))]
    if not fixtures:
        return make_failure(ctx, slice_id, v, "fixture_matrix requires fixtures or matrix")
    if not phases:
        return make_failure(ctx, slice_id, v, "fixture_matrix requires phases")
    keep_going = bool(v.get("keep_going", True))
    script = ctx.repo_path(str(v.get("script", "compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh")))
    check_script = ctx.repo_path(str(v.get("check_script", "compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py")))
    vc4_codegen = None
    per_fixture = []
    failures = []
    for fx in fixtures:
        name = str(fx.get("name", ""))
        if not name:
            continue
        root = ctx.repo_path(str(fx.get("root", f"compiler/test/CodeGen/VC4/Hardware/Run/{name}")))
        input_mlir = ctx.repo_path(str(fx.get("input", f"compiler/test/CodeGen/VC4/Hardware/Run/{name}/input.mlir")))
        expected_json = ctx.repo_path(str(fx.get("expected", f"compiler/test/CodeGen/VC4/Hardware/Run/{name}/expected.json")))
        bundle = resolve_repo_or_auto_path(ctx, str(fx.get("bundle", f"{ctx.env.get('VC4_CODEGEN_STATE_ROOT', '.vc4_auto/codegen_m1')}/candidates/{name}")))
        fx_results = []
        candidate_hardware_log: Optional[Path] = None
        for phase in phases:
            phase_id = f"{v.get('id', 'fixture_matrix')}_{name}_{phase}"
            log_path = ctx.command_log_path(slice_id, phase_id)
            if phase == "reference_hardware":
                if ctx.no_hardware:
                    fx_results.append({"phase": phase, "skipped": True})
                    continue
                result = ctx.run_command(["bash", "run.sh"], cwd=root, timeout_sec=verification_timeout_sec(ctx, {**v, "requires_hardware": True}), log_path=log_path)
            elif phase == "generate":
                if vc4_codegen is None:
                    try:
                        vc4_codegen = ctx.resolve_tool("vc4-codegen")
                    except VerificationError as exc:
                        failures.append({"fixture": name, "phase": phase, "error": str(exc), "details": exc.details})
                        break
                if bundle.exists() and not ctx.dry_run:
                    shutil.rmtree(bundle)
                if not ctx.dry_run:
                    bundle.mkdir(parents=True, exist_ok=True)
                result = ctx.run_command([vc4_codegen, str(input_mlir), "--emit-bundle", str(bundle)], cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
            elif phase in {"assemble", "build"}:
                result = ctx.run_command(["bash", str(script), name, phase], cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
            elif phase == "candidate_hardware":
                if ctx.no_hardware:
                    fx_results.append({"phase": phase, "skipped": True})
                    continue
                result = ctx.run_command(["bash", str(script), name, "run"], cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, {**v, "requires_hardware": True}), log_path=log_path)
                candidate_hardware_log = log_path
            elif phase == "expected_json":
                if candidate_hardware_log is None:
                    candidate_hardware_log = log_path.parent / f"{slice_id}_{v.get('id', 'fixture_matrix')}_{name}_candidate_hardware.log"
                result = ctx.run_command([sys.executable, str(check_script), str(expected_json), str(candidate_hardware_log)], cwd=ctx.repo, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
            else:
                failures.append({"fixture": name, "phase": phase, "error": "unknown phase"})
                break
            fx_results.append({"phase": phase, "ok": result.ok, "exit_code": result.exit_code, "timed_out": result.timed_out, "log_path": str(log_path)})
            if not result.ok:
                failures.append({"fixture": name, "phase": phase, "exit_code": result.exit_code, "timed_out": result.timed_out, "log_path": str(log_path), "stdout_tail": tail(result.stdout), "stderr_tail": tail(result.stderr)})
                if not keep_going:
                    break
        per_fixture.append({"name": name, "results": fx_results})
        if failures and not keep_going:
            break
    if failures:
        return make_failure(ctx, slice_id, v, "fixture matrix failed", expected={"fixtures": fixtures, "phases": phases}, actual={"failures": failures, "results": per_fixture}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="fixture matrix passed", details={"fixtures": per_fixture}, duration=time.time() - started)


def mechanism_resource_contract(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    try:
        manifest_path, manifest = manifest_from_bundle(ctx, v)
    except VerificationError as exc:
        return make_failure(ctx, slice_id, v, str(exc), actual=exc.details, duration=time.time() - started)
    target = v.get("target") if isinstance(v.get("target"), dict) else {}
    active_qpus = int(target.get("active_qpus", manifest.get("target", {}).get("max_active_qpus", 12)) or 12)
    vpm_bytes = int(target.get("vpm_bytes", manifest.get("target", {}).get("shared_vpm_bytes", 4096)) or 4096)
    semaphores = int(target.get("hardware_semaphores", manifest.get("target", {}).get("semaphores", 16)) or 16)
    failures = []
    manifest_by_public = {str(k.get("public_name")): k for k in manifest_kernels(manifest)}
    for req in as_list(v.get("kernels")):
        if not isinstance(req, dict):
            continue
        public = str(req.get("public_name", ""))
        kernel = manifest_by_public.get(public)
        if not kernel:
            failures.append({"public_name": public, "error": "kernel not found in manifest"})
            continue
        resources = kernel.get("resources") if isinstance(kernel.get("resources"), dict) else {}
        schedule_mode = kernel.get("schedule_mode")
        if req.get("schedule_mode") and schedule_mode != req.get("schedule_mode"):
            failures.append({"public_name": public, "error": "schedule_mode mismatch", "expected": req.get("schedule_mode"), "actual": schedule_mode})
        warps = int(resources.get("warps_per_block_max", kernel.get("warps_per_block_max", 1)) or 1)
        sem_per = int(resources.get("semaphores_per_block", 0) or 0)
        vpm_per = int(resources.get("vpm_bytes_per_block", 0) or 0)
        if req.get("warps_per_block_max") is not None and warps > int(req.get("warps_per_block_max")):
            failures.append({"public_name": public, "error": "warps_per_block_max exceeds expected", "actual": warps})
        if bool(req.get("require_full_block_residency")) and warps > active_qpus:
            failures.append({"public_name": public, "error": "warps_per_block_max exceeds active_qpus", "warps": warps, "active_qpus": active_qpus})
        if sem_per > semaphores:
            failures.append({"public_name": public, "error": "semaphores_per_block exceeds hardware semaphores", "semaphores_per_block": sem_per, "hardware_semaphores": semaphores})
        if vpm_per > vpm_bytes:
            failures.append({"public_name": public, "error": "vpm_bytes_per_block exceeds VPM bytes", "vpm_bytes_per_block": vpm_per, "vpm_bytes": vpm_bytes})
        for key in ["uses_barrier", "uses_shared_vpm"]:
            if key in req and bool(resources.get(key, False)) != bool(req.get(key)):
                failures.append({"public_name": public, "error": f"{key} mismatch", "expected": bool(req.get(key)), "actual": bool(resources.get(key, False))})
    expect = str(v.get("expect", "accept"))
    if expect == "accept" and failures:
        return make_failure(ctx, slice_id, v, "resource contract failed", actual={"failures": failures, "manifest": ctx.rel(manifest_path)}, duration=time.time() - started)
    if expect == "reject" and not failures:
        return make_failure(ctx, slice_id, v, "resource contract expected rejection but accepted", actual={"manifest": ctx.rel(manifest_path)}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="resource contract passed", details={"manifest": ctx.rel(manifest_path), "failures_observed": failures if expect == "reject" else []}, duration=time.time() - started)


def mechanism_support_script_contract(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    started = time.time()
    script = ctx.repo_path(str(v.get("script", "")))
    if not script.exists():
        return make_failure(ctx, slice_id, v, "support script missing", expected=str(script), actual={"exists": False}, duration=time.time() - started)
    text = read_text(script)
    required = regex_list(v.get("required_patterns"))
    forbidden = regex_list(v.get("forbidden_literals"))
    missing = [pat for pat in required if not re.search(pat, text, flags=re.S)]
    present_forbidden = [lit for lit in forbidden if lit in text]
    if missing or present_forbidden:
        return make_failure(ctx, slice_id, v, "support script contract failed", expected={"required_patterns": required, "forbidden_literals": forbidden}, actual={"missing_patterns": missing, "forbidden_present": present_forbidden}, duration=time.time() - started)
    return make_success(ctx, slice_id, v, message="support script contract passed", details={"script": ctx.rel(script)}, duration=time.time() - started)


def mechanism_negative_diagnostic(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    argv = v.get("argv") or v.get("command")
    if not isinstance(argv, list) or not argv:
        return make_failure(ctx, slice_id, v, "negative_diagnostic requires non-empty argv array")
    tmp_root = ctx.state_root / "tmp" / f"{slice_id}_{v.get('id', 'negative')}"
    if tmp_root.exists() and not ctx.dry_run:
        shutil.rmtree(tmp_root)
    if not ctx.dry_run:
        tmp_root.mkdir(parents=True, exist_ok=True)
    expanded = []
    for arg in argv:
        s = str(arg).replace("%t", str(tmp_root / "tmp"))
        expanded.append(s)
    # Resolve common repo-built tools while preserving explicit paths.
    if expanded and "/" not in expanded[0]:
        try:
            expanded[0] = ctx.resolve_tool(expanded[0])
        except VerificationError:
            pass
    cwd = ctx.repo_path(str(v.get("cwd", "."))) if v.get("cwd") else ctx.repo
    log_path = ctx.command_log_path(slice_id, str(v.get("id", "negative_diagnostic")))
    result = ctx.run_command(expanded, cwd=cwd, timeout_sec=verification_timeout_sec(ctx, v), log_path=log_path)
    expect_exit = v.get("expect_exit_code", "nonzero")
    if expect_exit == "nonzero":
        exit_ok = result.exit_code != 0
    else:
        exit_ok = result.exit_code == int(expect_exit)
    stdout_contains = regex_list(v.get("stdout_contains"))
    stderr_contains = regex_list(v.get("stderr_contains"))
    combined_contains = regex_list(v.get("contains"))
    missing_stdout = [s for s in stdout_contains if s not in result.stdout]
    missing_stderr = [s for s in stderr_contains if s not in result.stderr]
    combined = result.stdout + "\n" + result.stderr
    missing_combined = [s for s in combined_contains if s not in combined]
    bundle_created = False
    if bool(v.get("forbid_bundle_created", False)):
        bundle_created = any(p.exists() for p in tmp_root.glob("*.bundle")) or any(p.is_dir() and p.name.endswith(".bundle") for p in tmp_root.rglob("*"))
    if not exit_ok or missing_stdout or missing_stderr or missing_combined or bundle_created:
        return make_failure(ctx, slice_id, v, "negative diagnostic contract failed", expected={"expect_exit_code": expect_exit, "stdout_contains": stdout_contains, "stderr_contains": stderr_contains, "contains": combined_contains, "forbid_bundle_created": v.get("forbid_bundle_created", False)}, actual={"exit_code": result.exit_code, "missing_stdout": missing_stdout, "missing_stderr": missing_stderr, "missing_combined": missing_combined, "bundle_created": bundle_created}, command_result=result)
    return make_success(ctx, slice_id, v, message="negative diagnostic passed", duration=result.duration_sec, details={"log_path": str(log_path)})


MECHANISMS: Dict[str, Callable[[VerifierContext, str, Mapping[str, Any]], VerificationResult]] = {
    "source_products": mechanism_source_products,
    "forbidden_absent": mechanism_forbidden_absent,
    "tool_available": mechanism_tool_available,
    "command": mechanism_command,
    "build": mechanism_build,
    "lit": mechanism_lit,
    "vc4_codegen_generate": mechanism_vc4_codegen_generate,
    "artifact_bundle": mechanism_artifact_bundle,
    "qasm_text": mechanism_qasm_text,
    "vc4asm_assemble": mechanism_vc4asm_assemble,
    "c_syntax": mechanism_c_syntax,
    "candidate_phase": mechanism_candidate_phase,
    "hardware_run": mechanism_hardware_run,
    "expected_json_result": mechanism_expected_json_result,
    "reference_immutable": mechanism_reference_immutable,
    "launch_abi_public_api": mechanism_launch_abi_public_api,
    "launch_abi_uniform_layout": mechanism_launch_abi_uniform_layout,
    "manifest_schema": mechanism_manifest_schema,
    "program_artifact_bundle": mechanism_program_artifact_bundle,
    "all_qasm_assemble": mechanism_all_qasm_assemble,
    "program_layout_contract": mechanism_program_layout_contract,
    "generated_runtime_contract": mechanism_generated_runtime_contract,
    "heap_api_unit": mechanism_heap_api_unit,
    "launch_abi_contract": mechanism_launch_abi_contract,
    "runtime_event_log": mechanism_runtime_event_log,
    "fixture_matrix": mechanism_fixture_matrix,
    "resource_contract": mechanism_resource_contract,
    "support_script_contract": mechanism_support_script_contract,
    "negative_diagnostic": mechanism_negative_diagnostic,
}

MECHANISM_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "source_products": [],
    "forbidden_absent": ["globs"],
    "tool_available": [],
    "command": ["argv"],
    "build": ["target"],
    "lit": ["root"],
    "vc4_codegen_generate": ["input", "bundle"],
    "artifact_bundle": ["bundle"],
    "qasm_text": ["path"],
    "vc4asm_assemble": ["qasm"],
    "c_syntax": ["files"],
    "candidate_phase": ["name", "phase"],
    "hardware_run": ["fixture", "side"],
    "expected_json_result": ["expected", "log"],
    "reference_immutable": ["globs"],
    "launch_abi_public_api": ["input_mlir", "header"],
    "launch_abi_uniform_layout": ["input_mlir", "source"],
    "manifest_schema": ["bundle"],
    "program_artifact_bundle": ["bundle"],
    "all_qasm_assemble": ["bundle"],
    "program_layout_contract": ["bundle"],
    "generated_runtime_contract": ["bundle"],
    "heap_api_unit": ["source_files"],
    "launch_abi_contract": ["bundle"],
    "runtime_event_log": [],
    "fixture_matrix": ["phases"],
    "resource_contract": ["bundle"],
    "support_script_contract": ["script"],
    "negative_diagnostic": ["argv"],
}

MECHANISM_DOCS: Dict[str, str] = {
    "source_products": "Check exact source files and repo globs exist. Fields: files, globs, min_glob_matches.",
    "forbidden_absent": "Check forbidden source globs are absent, or git-status clean. Fields: globs, mode=filesystem|git-status.",
    "tool_available": "Check tools resolve from compiler/build/bin or PATH. Fields: tools, any_tools.",
    "command": "Run a typed argv command. Fields: argv, cwd, expect_exit_code, stdout_contains, stderr_contains.",
    "build": "Run ninja -C compiler/build <target>. Fields: target.",
    "lit": "Run lit -sv on a test root. Fields: root, optional tool.",
    "vc4_codegen_generate": "Run vc4-codegen input.mlir --emit-bundle bundle and require bundle files.",
    "artifact_bundle": "Check emitted bundle files and manifest keys/subset values.",
    "qasm_text": "Check qasm text contains / omits / orders tokens without exact-file matching.",
    "vc4asm_assemble": "Assemble qasm with vc4asm into generated C/header outputs.",
    "c_syntax": "Run host C compiler with -fsyntax-only over generated C/header probes.",
    "candidate_phase": "Run Support/run_candidate_codegen_test.sh TEST_NAME PHASE.",
    "hardware_run": "Run hardware reference or candidate side for a fixture. Skippable by --no-hardware.",
    "expected_json_result": "Run Support/check_vc4_test_result.py expected.json hardware.log.",
    "reference_immutable": "Check git status is clean for immutable reference/oracle globs.",
    "launch_abi_public_api": "Parse input vc4.launch_abi and verify generated public C API is semantic and does not expose hidden builtins/raw uniforms.",
    "launch_abi_uniform_layout": "Parse input vc4.launch_abi and verify generated launcher source has dense uniform slots and does not expose builtin values publicly.",
    "manifest_schema": "Validate manifest.json schema v2, kernel count, required keys, uniqueness, and bundle-relative paths.",
    "program_artifact_bundle": "Validate general program bundle filesystem shape; single-kernel is just kernels.length == 1.",
    "all_qasm_assemble": "Assemble every manifest-listed kernel qasm_path with vc4asm using code_symbol outputs.",
    "program_layout_contract": "Validate layout.json region offsets, alignments, no-overlap, heap, and kernel count.",
    "generated_runtime_contract": "Regex-check generated kernel_launch.c/.h for runtime API, allocation/code-upload policy, and launch-body restrictions.",
    "heap_api_unit": "Compile and run a host-only heap allocator unit test with optional stubs.",
    "launch_abi_contract": "Validate CUDA-like public launch ABI: program handle, grid/block, device pointers, scalars, and no implicit copies.",
    "runtime_event_log": "Parse stable runtime event/counter lines from hardware logs.",
    "fixture_matrix": "Run a declarative fixture matrix through generate/assemble/build/hardware/expected-json phases.",
    "resource_contract": "Validate independent/cooperative scheduler resource metadata against target QPU/VPM/semaphore limits.",
    "support_script_contract": "Scan support scripts for required program-bundle patterns and forbidden single-kernel literals.",
    "negative_diagnostic": "Run a command expected to fail and check deterministic diagnostics.",
}


def verify_one(ctx: VerifierContext, slice_id: str, v: Mapping[str, Any]) -> VerificationResult:
    mechanism = str(v.get("mechanism", ""))
    if not mechanism:
        return make_failure(ctx, slice_id, v, "verification is missing mechanism", expected="mechanism", actual=v)
    if bool(v.get("requires_hardware", False)) and ctx.no_hardware:
        return skip_result(ctx, slice_id, v, "hardware verification skipped by --no-hardware")
    if bool(v.get("skip", False)):
        return skip_result(ctx, slice_id, v, str(v.get("skip_reason", "marked skip in spec")))
    fn = MECHANISMS.get(mechanism)
    if fn is None:
        return make_failure(ctx, slice_id, v, f"unknown verification mechanism: {mechanism}", expected=sorted(MECHANISMS), actual=mechanism)
    try:
        return fn(ctx, slice_id, v)
    except subprocess.TimeoutExpired as e:
        return make_failure(ctx, slice_id, v, "verification command timed out", actual={"timeout_sec": e.timeout, "cmd": e.cmd, "exit_code": 124}, details={"stdout_tail": tail(coerce_process_text(e.stdout)), "stderr_tail": tail(coerce_process_text(e.stderr))})
    except VerificationError as e:
        return make_failure(ctx, slice_id, v, str(e), details=e.details)
    except Exception as e:
        return make_failure(ctx, slice_id, v, f"verifier internal error: {type(e).__name__}: {e}", details={"exception_type": type(e).__name__})


def run_verify(ctx: VerifierContext, slice_ids: List[str]) -> Dict[str, Any]:
    slices = get_slices(ctx.spec)
    all_results: List[VerificationResult] = []
    started = time.time()
    for slice_id in slice_ids:
        spec = slices.get(slice_id)
        if not spec:
            all_results.append(VerificationResult(False, slice_id, "<slice>", "slice", category="unknown_slice", message=f"slice not found in spec: {slice_id}"))
            if not ctx.keep_going:
                break
            continue
        verifications = spec.get("verifications") or []
        if not isinstance(verifications, list) or not verifications:
            all_results.append(VerificationResult(False, slice_id, "<slice>", "slice", category="missing_verifications", message=f"slice has no verifications: {slice_id}"))
            if not ctx.keep_going:
                break
            continue
        for v in verifications:
            if not isinstance(v, dict):
                all_results.append(VerificationResult(False, slice_id, "<non-object>", "schema", category="invalid_spec", message="verification entry is not an object", actual=v))
                if not ctx.keep_going:
                    break
                continue
            if ctx.only_mechanisms and str(v.get("mechanism", "")) not in ctx.only_mechanisms:
                continue
            result = verify_one(ctx, slice_id, v)
            all_results.append(result)
            if ctx.verbose:
                status = "SKIP" if result.skipped else "OK" if result.ok else "FAIL"
                print(f"[{status}] {slice_id}:{result.verification_id} {result.mechanism} {result.message}", file=sys.stderr)
            if not result.ok and result.required and not ctx.keep_going:
                break
        if any((not r.ok and r.required) for r in all_results if r.slice_id == slice_id) and not ctx.keep_going:
            break
    ok = all(r.ok or not r.required for r in all_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "duration_sec": round(time.time() - started, 3),
        "repo": str(ctx.repo),
        "spec": str(Path(ctx.args.spec).resolve()) if ctx.args.spec else DEFAULT_SPEC,
        "slice_ids": slice_ids,
        "failures": [r.to_packet() for r in all_results if not r.ok and r.required],
        "results": [r.to_packet() for r in all_results],
    }


SOURCE_PRODUCT_FILE_KEYS = (
    "required_source_files",
    "required_existing_files",
    "support_files",
    "semantic_oracle_files",
)
SOURCE_PRODUCT_MAPPING_KEYS = (
    "candidate_input_files",
    "candidate_fixture_inputs",
    "hardware_fixture_inputs",
)
SOURCE_PRODUCT_GLOB_KEYS = (
    "required_source_globs",
    "lit_test_globs",
)


def _extend_unique(dst: List[str], values: Iterable[str]) -> None:
    seen = set(dst)
    for value in values:
        if value not in seen:
            dst.append(value)
            seen.add(value)


def worklist_source_products(wslice: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    """Return exact source-product files/globs declared by a worklist slice.

    The worklist is the human-facing slice contract and the verifier spec is the
    executable mechanism list.  Keep them connected by deriving source-product
    checks from worklist ``products`` instead of relying on every spec author to
    duplicate the same filenames perfectly.
    """
    products = wslice.get("products") if isinstance(wslice.get("products"), Mapping) else {}
    files: List[str] = []
    globs: List[str] = []
    for key in SOURCE_PRODUCT_FILE_KEYS:
        value = products.get(key)
        if isinstance(value, str):
            _extend_unique(files, [value])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            _extend_unique(files, [str(x) for x in value])
    for key in SOURCE_PRODUCT_MAPPING_KEYS:
        value = products.get(key)
        if isinstance(value, Mapping):
            _extend_unique(files, [str(x) for x in value.values() if isinstance(x, str)])
    for key in SOURCE_PRODUCT_GLOB_KEYS:
        value = products.get(key)
        if isinstance(value, str):
            _extend_unique(globs, [value])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            _extend_unique(globs, [str(x) for x in value])
    return files, globs


def augment_spec_with_worklist_source_products(spec: Dict[str, Any], worklist: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a spec copy whose source_products checks include worklist products."""
    if not worklist:
        return spec
    augmented = copy.deepcopy(spec)
    spec_slices = get_slices(augmented)
    work_slices = get_slices(worklist)
    for sid, wslice in work_slices.items():
        if sid not in spec_slices:
            continue
        files, globs = worklist_source_products(wslice)
        if not files and not globs:
            continue
        sspec = spec_slices[sid]
        verifications = sspec.setdefault("verifications", [])
        if not isinstance(verifications, list):
            continue
        source_v = None
        for v in verifications:
            if isinstance(v, dict) and v.get("mechanism") == "source_products":
                source_v = v
                break
        if source_v is None:
            source_v = {
                "id": "worklist-source-products",
                "mechanism": "source_products",
                "description": "Exact source products declared by the worklist products contract.",
                "category": "source_product_missing",
                "route_hint": "gpt_pro",
                "files": [],
                "globs": [],
            }
            verifications.insert(0, source_v)
        cur_files = [str(x) for x in as_list(source_v.get("files"))]
        cur_globs = [str(x) for x in as_list(source_v.get("globs"))]
        _extend_unique(cur_files, files)
        _extend_unique(cur_globs, globs)
        if cur_files:
            source_v["files"] = cur_files
        if cur_globs:
            source_v["globs"] = cur_globs
    return augmented


def audit_contract(spec: Dict[str, Any], *, worklist: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    if spec.get("schema_version") != SCHEMA_VERSION:
        errors.append({"category": "schema_version", "message": f"expected schema_version {SCHEMA_VERSION}", "actual": spec.get("schema_version")})
    slices = get_slices(spec)
    if not slices:
        errors.append({"category": "missing_slices", "message": "spec has no slices"})
    for sid, sspec in slices.items():
        verifications = sspec.get("verifications")
        if not isinstance(verifications, list) or not verifications:
            errors.append({"slice_id": sid, "category": "missing_verifications", "message": "slice has no verification list"})
            continue
        seen_ids: set[str] = set()
        for idx, v in enumerate(verifications):
            if not isinstance(v, dict):
                errors.append({"slice_id": sid, "category": "invalid_verification", "index": idx, "message": "verification is not an object"})
                continue
            vid = str(v.get("id", ""))
            if not vid:
                errors.append({"slice_id": sid, "category": "missing_id", "index": idx, "message": "verification is missing id"})
            elif vid in seen_ids:
                errors.append({"slice_id": sid, "category": "duplicate_id", "id": vid})
            seen_ids.add(vid)
            mechanism = str(v.get("mechanism", ""))
            if mechanism not in MECHANISMS:
                errors.append({"slice_id": sid, "verification_id": vid, "category": "unknown_mechanism", "mechanism": mechanism})
                continue
            missing = [field for field in MECHANISM_REQUIRED_FIELDS.get(mechanism, []) if field not in v]
            if missing:
                errors.append({"slice_id": sid, "verification_id": vid, "category": "missing_required_fields", "mechanism": mechanism, "missing": missing})
    if worklist:
        work_slices = get_slices(worklist)
        missing_specs = [sid for sid in work_slices if sid not in slices]
        extra_specs = [sid for sid in slices if sid not in work_slices]
        if missing_specs:
            errors.append({"category": "worklist_specs_missing", "slice_ids": missing_specs})
        if extra_specs:
            warnings.append({"category": "extra_specs_not_in_worklist", "slice_ids": extra_specs})
        for sid, wspec in work_slices.items():
            if sid not in slices:
                continue
            products = wspec.get("products") if isinstance(wspec.get("products"), dict) else {}
            declared_source_files: List[str] = []
            for key in ["required_source_files", "required_existing_files", "candidate_input_files", "candidate_fixture_inputs", "hardware_fixture_inputs", "support_files", "semantic_oracle_files"]:
                val = products.get(key)
                if isinstance(val, list):
                    declared_source_files.extend(str(x) for x in val)
                elif isinstance(val, dict):
                    declared_source_files.extend(str(x) for x in val.values())
            declared_globs: List[str] = []
            for key in ["required_source_globs", "lit_test_globs"]:
                val = products.get(key)
                if isinstance(val, list):
                    declared_globs.extend(str(x) for x in val)
            spec_source_files: List[str] = []
            spec_source_globs: List[str] = []
            for v in slices[sid].get("verifications", []):
                if isinstance(v, dict) and v.get("mechanism") == "source_products":
                    spec_source_files.extend(str(x) for x in as_list(v.get("files")))
                    spec_source_globs.extend(str(x) for x in as_list(v.get("globs")))
            missing_declared_files = sorted(set(declared_source_files) - set(spec_source_files))
            if missing_declared_files:
                errors.append({"slice_id": sid, "category": "declared_source_files_not_in_source_products_verification", "paths": missing_declared_files})
            missing_declared_globs = sorted(set(declared_globs) - set(spec_source_globs))
            if missing_declared_globs:
                errors.append({"slice_id": sid, "category": "declared_source_globs_not_in_source_products_verification", "globs": missing_declared_globs})
    return {"schema_version": SCHEMA_VERSION, "ok": not errors, "errors": errors, "warnings": warnings, "mechanisms": sorted(MECHANISMS)}


def explain_slice(spec: Dict[str, Any], slice_id: str) -> Dict[str, Any]:
    slices = get_slices(spec)
    if slice_id not in slices:
        raise SystemExit(f"slice not found in spec: {slice_id}")
    sspec = slices[slice_id]
    verifications = []
    for v in sspec.get("verifications", []):
        if not isinstance(v, dict):
            continue
        mechanism = str(v.get("mechanism", ""))
        verifications.append({
            "id": v.get("id"),
            "mechanism": mechanism,
            "description": v.get("description", ""),
            "requires_hardware": bool(v.get("requires_hardware", False)),
            "required_fields": MECHANISM_REQUIRED_FIELDS.get(mechanism, []),
            "mechanism_doc": MECHANISM_DOCS.get(mechanism, ""),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "slice_id": slice_id,
        "title": sspec.get("title"),
        "expected_end_state": sspec.get("expected_end_state"),
        "verifications": verifications,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "mechanisms", "explain", "audit-contract", "verify"])
    parser.add_argument("--repo", default=".", help="repo root (default: cwd)")
    parser.add_argument("--spec", default=DEFAULT_SPEC, help=f"verification spec JSON (default: {DEFAULT_SPEC})")
    parser.add_argument("--worklist", default=DEFAULT_WORKLIST, help=f"worklist JSON for audit-contract (default: {DEFAULT_WORKLIST})")
    parser.add_argument("--slice", dest="slice_id", action="append", help="slice id; repeatable; use 'all' for all slices")
    parser.add_argument("--out", help="write JSON report to this path")
    parser.add_argument("--state-root", help="override generated verifier state root")
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--hardware-timeout-sec", type=int, default=0, help="default timeout for hardware-like verifications (default: spec defaults.hardware_timeout_sec, env VC4_HARDWARE_TIMEOUT_SEC, or 120)")
    parser.add_argument("--keep-going", action="store_true", help="run all requested verifications even after failures")
    parser.add_argument("--no-hardware", action="store_true", help="skip verifications marked requires_hardware")
    parser.add_argument("--dry-run", action="store_true", help="validate command construction without running external commands")
    parser.add_argument("--only-mechanism", action="append", default=[], help="verify only entries with this mechanism; repeatable")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="print JSON output (default for verify/audit)")
    return parser.parse_args(argv)


def selected_slices(spec: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    slices = get_slices(spec)
    requested = args.slice_id or []
    if not requested:
        if args.command in ("explain", "verify"):
            raise SystemExit("--slice is required for explain/verify (or use --slice all)")
        return list(slices.keys())
    out: List[str] = []
    for sid in requested:
        if sid == "all":
            out.extend(list(slices.keys()))
        else:
            out.append(sid)
    # preserve order, de-duplicate
    seen: set[str] = set()
    deduped: List[str] = []
    for sid in out:
        if sid not in seen:
            deduped.append(sid)
            seen.add(sid)
    return deduped



def _json_safe_for_report(value):
    """Recursively coerce verifier reports to JSON-serializable values.

    TimeoutExpired and some subprocess paths can carry raw bytes in stdout/stderr.
    The verifier must never crash while reporting a failure; it should surface the
    failure packet deterministically.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, bytearray):
        return bytes(value).decode("utf-8", errors="replace")

    if isinstance(value, tuple):
        return [_json_safe_for_report(v) for v in value]

    if isinstance(value, list):
        return [_json_safe_for_report(v) for v in value]

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if isinstance(k, bytes):
                kk = k.decode("utf-8", errors="replace")
            else:
                kk = str(k)
            out[kk] = _json_safe_for_report(v)
        return out

    # pathlib.Path, enums, exceptions, and any accidental custom objects.
    return str(value)

def emit_report(report: Dict[str, Any], args: argparse.Namespace) -> None:
    text = json.dumps(_json_safe_for_report(report), indent=2, sort_keys=False) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).resolve()
    spec_path = repo / args.spec if not Path(args.spec).is_absolute() else Path(args.spec)
    spec = load_json(spec_path)
    args.spec = str(spec_path)
    worklist_path = repo / args.worklist if not Path(args.worklist).is_absolute() else Path(args.worklist)
    worklist = load_json(worklist_path) if worklist_path.exists() else None
    if args.command in {"explain", "verify", "audit-contract"}:
        spec = augment_spec_with_worklist_source_products(spec, worklist)

    if args.command == "list":
        report = {"schema_version": SCHEMA_VERSION, "slices": [{"id": sid, "title": s.get("title", "")} for sid, s in get_slices(spec).items()]}
        emit_report(report, args)
        return 0

    if args.command == "mechanisms":
        report = {"schema_version": SCHEMA_VERSION, "mechanisms": {name: {"required_fields": MECHANISM_REQUIRED_FIELDS.get(name, []), "doc": MECHANISM_DOCS.get(name, "")} for name in sorted(MECHANISMS)}}
        emit_report(report, args)
        return 0

    if args.command == "audit-contract":
        report = audit_contract(spec, worklist=worklist)
        emit_report(report, args)
        return 0 if report.get("ok") else 1

    if args.command == "explain":
        ids = selected_slices(spec, args)
        report = {"schema_version": SCHEMA_VERSION, "slices": [explain_slice(spec, sid) for sid in ids]}
        emit_report(report, args)
        return 0

    if args.command == "verify":
        ctx = VerifierContext(repo, spec, args)
        report = run_verify(ctx, selected_slices(spec, args))
        emit_report(report, args)
        return 0 if report.get("ok") else 1

    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
