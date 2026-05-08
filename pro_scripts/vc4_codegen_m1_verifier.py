#!/usr/bin/env python3
"""Typed deterministic verifier for VC4 Codegen Milestone 1 slices.

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
        self.timeout_sec = int(args.timeout_sec or defaults.get("timeout_sec", 7200))
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
        proc = subprocess.run(
            cmd,
            cwd=str(real_cwd),
            env=self.env,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec or self.timeout_sec,
        )
        duration = time.time() - started
        if log_path:
            with log_path.open("w", encoding="utf-8") as f:
                f.write(f"# cwd: {real_cwd}\n")
                f.write("# command: " + " ".join(shell_quote(x) for x in cmd) + "\n")
                f.write(f"# exit_code: {proc.returncode}\n")
                f.write(f"# duration_sec: {duration:.3f}\n")
                f.write("\n## stdout\n")
                f.write(proc.stdout)
                f.write("\n## stderr\n")
                f.write(proc.stderr)
        return CommandResult(
            ok=proc.returncode == 0,
            argv=cmd,
            cwd=real_cwd,
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            log_path=log_path,
            duration_sec=duration,
        )


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
    return VerificationResult(
        ok=False,
        slice_id=slice_id,
        verification_id=str(v.get("id", "<unnamed>")),
        mechanism=str(v.get("mechanism", "<unknown>")),
        description=str(v.get("description", "")),
        category=str(v.get("category", v.get("mechanism", "verification_failed"))),
        route_hint=str(v.get("route_hint", "gpt_pro")),
        required=bool(v.get("required", True)),
        duration_sec=duration,
        message=message,
        expected=expected,
        actual=actual,
        details=details,
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
    result = ctx.run_command(cmd, cwd=cwd, timeout_sec=int(v.get("timeout_sec", ctx.timeout_sec)), log_path=log_path)
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
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=int(v.get("timeout_sec", ctx.timeout_sec)), log_path=log_path)
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
    result = ctx.run_command(argv, cwd=ctx.repo / "compiler/build/test" if (ctx.repo / "compiler/build/test").exists() else ctx.repo, timeout_sec=int(v.get("timeout_sec", ctx.timeout_sec)), log_path=log_path)
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
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=int(v.get("timeout_sec", ctx.timeout_sec)), log_path=log_path)
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
    result = ctx.run_command(argv, cwd=cwd, timeout_sec=int(v.get("timeout_sec", ctx.timeout_sec)), log_path=log_path)
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
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=int(v.get("timeout_sec", ctx.timeout_sec)), log_path=log_path)
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
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=int(v.get("timeout_sec", ctx.timeout_sec)), log_path=log_path)
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
    result = ctx.run_command(argv, cwd=cwd, timeout_sec=int(v.get("timeout_sec", ctx.timeout_sec)), log_path=log_path)
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
    result = ctx.run_command(argv, cwd=ctx.repo, timeout_sec=int(v.get("timeout_sec", ctx.timeout_sec)), log_path=log_path)
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
        return make_failure(ctx, slice_id, v, "verification command timed out", actual={"timeout_sec": e.timeout, "cmd": e.cmd}, details={"stdout_tail": tail(e.stdout or ""), "stderr_tail": tail(e.stderr or "")})
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


def emit_report(report: Dict[str, Any], args: argparse.Namespace) -> None:
    text = json.dumps(report, indent=2, sort_keys=False) + "\n"
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
