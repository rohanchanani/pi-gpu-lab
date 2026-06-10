#!/usr/bin/env python3
import argparse
import atexit
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_PHASE_DIRS = [
    "~/Downloads/vc4_phase11_strided_ranked_memory_skeletons_package",
    "~/Downloads/vc4_phase12_reductions_package",
    "~/Downloads/vc4_phase13_gemv_rowwise_dot_package",
    "~/Downloads/vc4_phase14_ml_storage_numeric_conversion_package",
]

DEFAULT_EXCLUDE_RE = (
    r"(?i)(^README(?:\.txt|\.md)?$|"
    r"failure|triage|_X_|Phase.*X_|context|"
    r"\.zip$|\.DS_Store$)"
)

SUCCESS_LINE_RE = re.compile(r"^SUCCESS\b")
FAILURE_LINE_RE = re.compile(r"^(FAILURE|BLOCKED)\b")
RESULT_FAILURE_RE = re.compile(
    r"(?m)^(?:[A-Z0-9_]*RESULT|PHASE[^=\n]*_RESULT|TOP_HALF[^=\n]*_RESULT|.*_RESULT)=FAILURE\b"
)
RESULT_BLOCKED_RE = re.compile(
    r"(?m)^(?:[A-Z0-9_]*RESULT|PHASE[^=\n]*_RESULT|TOP_HALF[^=\n]*_RESULT|.*_RESULT)=BLOCKED\b"
)
READY_NO_RE = re.compile(r"(?m)^READY_FOR_(?:NEXT_PHASE|PHASE[0-9A-Z_]+|FEATURE_LADDER_PHASE[0-9]+)=NO\b")

def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]

def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)

def first_nonempty_line(text: str) -> str:
    for line in strip_ansi(text).splitlines():
        if line.strip():
            return line.strip()
    return ""

def classify_final_message(text: str, allow_locked: bool) -> Tuple[str, str]:
    clean = strip_ansi(text)
    first = first_nonempty_line(clean)

    if SUCCESS_LINE_RE.match(first):
        return "success", "first non-empty line is SUCCESS"

    if FAILURE_LINE_RE.match(first):
        return "failure", f"first non-empty line is {first.split()[0]}"

    if RESULT_FAILURE_RE.search(clean):
        return "failure", "contains RESULT=FAILURE"

    if RESULT_BLOCKED_RE.search(clean):
        return "failure", "contains RESULT=BLOCKED"

    if READY_NO_RE.search(clean):
        return "failure", "contains READY_FOR_*=NO gate"

    if allow_locked and re.search(r"(?m)^[A-Z0-9_]*RESULT=LOCKED\b", clean):
        return "success", "contains RESULT=LOCKED and no failure marker"

    return "unknown", "final message did not begin with SUCCESS/FAILURE and no accepted success marker was found"

def resolve_phase_dir(raw: str, allow_glob: bool) -> Path:
    expanded = Path(raw).expanduser()
    if expanded.is_dir():
        return expanded.resolve()

    if not allow_glob:
        raise FileNotFoundError(f"phase directory not found: {expanded}")

    parent = expanded.parent
    name = expanded.name
    candidates = []
    if parent.is_dir():
        candidates.extend([p for p in parent.glob(name + "*") if p.is_dir()])
        phase_match = re.search(r"(phase\d+)", name, re.IGNORECASE)
        if phase_match:
            candidates.extend([p for p in parent.glob(f"*{phase_match.group(1)}*") if p.is_dir()])

    candidates = sorted(set(candidates), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(f"phase directory not found and no glob candidate found: {expanded}")

def collect_prompt_files(folder: Path, include_re: Optional[str], exclude_re: Optional[str]) -> List[Path]:
    include = re.compile(include_re) if include_re else None
    exclude = re.compile(exclude_re if exclude_re is not None else DEFAULT_EXCLUDE_RE)
    files = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".md", ".prompt"}:
            continue
        if include and not include.search(p.name):
            continue
        if exclude and exclude.search(p.name):
            continue
        files.append(p)
    return sorted(files, key=natural_key)

def run_capture(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, errors="replace")
    return proc.returncode, proc.stdout, proc.stderr

def git_head(repo: Path) -> str:
    rc, out, err = run_capture(["git", "rev-parse", "HEAD"], repo)
    return out.strip() if rc == 0 else f"<git rev-parse failed: {err.strip()}>"

def git_status(repo: Path) -> str:
    rc, out, err = run_capture(["git", "status", "--short", "--branch"], repo)
    return out if rc == 0 else err

def has_dirty_tracked_changes(repo: Path) -> bool:
    rc, out, _ = run_capture(["git", "status", "--porcelain"], repo)
    if rc != 0:
        return True
    for line in out.splitlines():
        if not line.startswith("?? "):
            return True
    return False

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def append_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(text)

def stream_process(cmd: List[str], cwd: Path, stdin_text: str, log_path: Path, env: dict) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND\t" + " ".join(shlex.quote(x) for x in cmd) + "\n")
        log.write("CWD\t" + str(cwd) + "\n")
        log.write("START_UTC\t" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )

        assert proc.stdin is not None
        try:
            proc.stdin.write(stdin_text)
            proc.stdin.close()
        except BrokenPipeError:
            pass

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()

        rc = proc.wait()
        log.write("\nEND_UTC\t" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        log.write("RETURN_CODE\t" + str(rc) + "\n")
        log.flush()
        return rc

def extract_session_id_from_json_log(log_path: Path) -> Optional[str]:
    if not log_path.exists():
        return None
    uuid_re = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

    def walk(obj: Any) -> Iterable[Tuple[str, Any]]:
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k, v
                yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)

    found = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        for k, v in walk(obj):
            if isinstance(v, str) and uuid_re.match(v) and k.lower() in {
                "session_id", "sessionid", "conversation_id", "conversationid", "id"
            }:
                found = v
    return found

def build_codex_command(
    codex: str,
    prompt_index: int,
    final_path: Path,
    yolo: bool,
    extra_flags: List[str],
    json_events: bool,
    session_id: Optional[str],
    prefer_session_id: bool,
) -> List[str]:
    cmd = [codex, "exec"]
    if json_events:
        cmd.append("--json")
    cmd.extend(["--output-last-message", str(final_path)])
    if yolo:
        cmd.append("--dangerously-bypass-approvals-and-sandbox")
    cmd.extend(extra_flags)

    if prompt_index == 1:
        cmd.append("-")
        return cmd

    cmd.append("resume")
    if prefer_session_id and session_id:
        cmd.append(session_id)
    else:
        cmd.append("--last")
    cmd.append("-")
    return cmd

def preflight_codex(codex: str, repo: Path, logs: Path, require_login: bool) -> bool:
    ok = True
    for name, cmd in [
        ("codex_version", [codex, "--version"]),
        ("codex_exec_help", [codex, "exec", "--help"]),
        ("codex_exec_resume_help", [codex, "exec", "resume", "--help"]),
    ]:
        rc, out, err = run_capture(cmd, repo)
        write_text(logs / "preflight" / f"{name}.log", f"COMMAND\t{' '.join(cmd)}\nRC\t{rc}\n\nSTDOUT\n{out}\n\nSTDERR\n{err}\n")
        if rc != 0:
            ok = False
    rc, out, err = run_capture([codex, "login", "status"], repo)
    write_text(logs / "preflight" / "codex_login_status.log", f"COMMAND\t{codex} login status\nRC\t{rc}\n\nSTDOUT\n{out}\n\nSTDERR\n{err}\n")
    if require_login and rc != 0:
        ok = False
    return ok

def acquire_lock(repo: Path, logs: Path, force: bool) -> Path:
    lock_dir = repo / ".vc4_auto"
    lock_dir.mkdir(exist_ok=True)
    lock_path = lock_dir / "codex_ladder_runner.lock"
    if lock_path.exists() and not force:
        raise SystemExit(
            f"lock file exists: {lock_path}\n"
            f"Remove it only if no runner is active, or rerun with --force-lock."
        )
    lock_path.write_text(
        f"pid={os.getpid()}\n"
        f"started_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"logs={logs}\n",
        encoding="utf-8",
    )
    def cleanup():
        try:
            if lock_path.exists():
                text = lock_path.read_text(encoding="utf-8", errors="replace")
                if f"pid={os.getpid()}" in text:
                    lock_path.unlink()
        except Exception:
            pass
    atexit.register(cleanup)
    return lock_path

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run VC4 feature-ladder Codex phase packages, defaulting to Phase 11/12/13/14 from ~/Downloads with YOLO permissions."
    )
    parser.add_argument(
        "phase_dirs",
        nargs="*",
        help="Phase package folders. If omitted, defaults to ~/Downloads/vc4_phase11_strided_ranked_memory_skeletons_package, ~/Downloads/vc4_phase12_reductions_package, ~/Downloads/vc4_phase13_gemv_rowwise_dot_package, ~/Downloads/vc4_phase14_ml_storage_numeric_conversion_package.",
    )
    parser.add_argument("--repo", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument("--logs", default=None, help="Output log directory. Defaults to .vc4_auto/codex_ladder_runner_<timestamp>.")
    parser.add_argument("--codex", default="codex", help="Codex executable. Defaults to codex.")
    parser.add_argument("--codex-flags", default=os.environ.get("CODEX_LADDER_RUNNER_FLAGS", ""), help="Extra Codex flags appended to every run.")
    parser.add_argument("--no-yolo", action="store_true", help="Disable default --dangerously-bypass-approvals-and-sandbox.")
    parser.add_argument("--no-json", action="store_true", help="Do not pass --json to codex exec.")
    parser.add_argument("--include-regex", default=None, help="Only run prompt files whose basename matches this regex.")
    parser.add_argument("--exclude-regex", default=None, help="Exclude prompt files whose basename matches this regex. Default excludes README/failure/triage/context files.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved package directories and prompt order without running Codex.")
    parser.add_argument("--allow-unknown-success", action="store_true", help="Continue if final message is not clearly SUCCESS but process exited 0 and no failure marker is found.")
    parser.add_argument("--allow-result-locked-success", action="store_true", help="Treat RESULT=LOCKED as success when no explicit SUCCESS line is present.")
    parser.add_argument("--stop-on-dirty-start", action="store_true", help="Stop if tracked files are dirty at runner start.")
    parser.add_argument("--stop-on-dirty-between-prompts", action="store_true", help="Stop before each prompt if tracked files are dirty. Normally leave off because prompts commit incrementally.")
    parser.add_argument("--no-glob-defaults", action="store_true", help="Do not glob for default phase dirs if exact default names are absent.")
    parser.add_argument("--force-lock", action="store_true", help="Overwrite existing .vc4_auto/codex_ladder_runner.lock.")
    parser.add_argument("--no-login-check", action="store_true", help="Do not require `codex login status` to succeed in preflight.")
    parser.add_argument("--resume-with-last-only", action="store_true", help="Always resume with --last instead of trying a parsed session id.")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"repo does not look like a git repository: {repo}")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    logs = Path(args.logs).resolve() if args.logs else repo / ".vc4_auto" / f"codex_ladder_runner_{timestamp}"
    logs.mkdir(parents=True, exist_ok=True)

    lock_path = acquire_lock(repo, logs, args.force_lock)
    write_text(logs / "lock_path.txt", str(lock_path) + "\n")

    yolo = not args.no_yolo
    extra_flags = shlex.split(args.codex_flags)
    json_events = not args.no_json

    phase_inputs = args.phase_dirs if args.phase_dirs else DEFAULT_PHASE_DIRS
    phase_dirs = []
    for raw in phase_inputs:
        phase_dirs.append(resolve_phase_dir(raw, allow_glob=not args.no_glob_defaults))

    start_status = git_status(repo)
    write_text(logs / "start_git_status.txt", start_status)
    write_text(logs / "start_git_head.txt", git_head(repo) + "\n")

    if args.stop_on_dirty_start and has_dirty_tracked_changes(repo):
        print("Tracked files are dirty at start and --stop-on-dirty-start was set.")
        print(start_status)
        return 2

    if not preflight_codex(args.codex, repo, logs, require_login=not args.no_login_check):
        print("Codex preflight failed. See logs:")
        print(logs / "preflight")
        return 2

    summary: Dict[str, Any] = {
        "repo": str(repo),
        "started_utc": timestamp,
        "phase_dirs": [str(p) for p in phase_dirs],
        "yolo": yolo,
        "codex_flags": extra_flags,
        "json_events": json_events,
        "phases": [],
    }

    all_prompt_manifests = []
    for phase_index, phase_dir in enumerate(phase_dirs, start=1):
        prompts = collect_prompt_files(phase_dir, args.include_regex, args.exclude_regex)
        if not prompts:
            raise SystemExit(f"no prompt files found in {phase_dir}")
        all_prompt_manifests.append((phase_dir, prompts))

    for phase_dir, prompts in all_prompt_manifests:
        print(f"\nPACKAGE\t{phase_dir}")
        for p in prompts:
            print(f"  {p.name}")

    if args.dry_run:
        write_text(logs / "dry_run_summary.json", json.dumps(summary, indent=2) + "\n")
        print(f"\nDRY_RUN_COMPLETE\nLogs: {logs}")
        return 0

    previous_signal_handlers = {}
    interrupted = {"value": False}
    def handle_signal(signum, frame):
        interrupted["value"] = True
        print(f"\nReceived signal {signum}; stopping after current prompt if possible.", file=sys.stderr)
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handle_signal)

    try:
        for phase_index, (phase_dir, prompts) in enumerate(all_prompt_manifests, start=1):
            phase_name = phase_dir.name
            phase_log_dir = logs / f"{phase_index:02d}_{phase_name}"
            phase_log_dir.mkdir(parents=True, exist_ok=True)
            write_text(phase_log_dir / "prompt_order.txt", "\n".join(str(p) for p in prompts) + "\n")

            print(f"\n=== PHASE {phase_index}/{len(all_prompt_manifests)}: {phase_name} ===")

            phase_record: Dict[str, Any] = {
                "phase_index": phase_index,
                "phase_name": phase_name,
                "phase_dir": str(phase_dir),
                "prompts": [],
            }
            summary["phases"].append(phase_record)

            session_id: Optional[str] = None

            for prompt_index, prompt_path in enumerate(prompts, start=1):
                if interrupted["value"]:
                    print("Stopping due to earlier signal.")
                    return 130

                if args.stop_on_dirty_between_prompts and has_dirty_tracked_changes(repo):
                    print("Tracked files are dirty before prompt and --stop-on-dirty-between-prompts was set.")
                    print(git_status(repo))
                    return 2

                prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace")
                final_path = phase_log_dir / f"{prompt_index:02d}_{prompt_path.stem}.final.txt"
                raw_log_path = phase_log_dir / f"{prompt_index:02d}_{prompt_path.stem}.raw.log"
                before_head = git_head(repo)

                cmd = build_codex_command(
                    codex=args.codex,
                    prompt_index=prompt_index,
                    final_path=final_path,
                    yolo=yolo,
                    extra_flags=extra_flags,
                    json_events=json_events,
                    session_id=session_id,
                    prefer_session_id=not args.resume_with_last_only,
                )

                print(f"\n=== RUN {phase_name} prompt {prompt_index}/{len(prompts)}: {prompt_path.name} ===")
                print("Git HEAD before:", before_head)
                print("Command:", " ".join(shlex.quote(x) for x in cmd))

                rc = stream_process(cmd, repo, prompt_text, raw_log_path, os.environ.copy())
                after_head = git_head(repo)
                print("Git HEAD after:", after_head)

                if prompt_index == 1 and not args.resume_with_last_only:
                    session_id = extract_session_id_from_json_log(raw_log_path)
                    if session_id:
                        write_text(phase_log_dir / "session_id.txt", session_id + "\n")

                final_text = ""
                if final_path.exists():
                    final_text = final_path.read_text(encoding="utf-8", errors="replace")
                else:
                    raw_text = raw_log_path.read_text(encoding="utf-8", errors="replace")
                    final_text = raw_text[-20000:]
                    write_text(final_path, final_text)

                classification, reason = classify_final_message(
                    final_text,
                    allow_locked=args.allow_result_locked_success,
                )

                prompt_record = {
                    "prompt_index": prompt_index,
                    "prompt": str(prompt_path),
                    "returncode": rc,
                    "classification": classification,
                    "reason": reason,
                    "final_message": str(final_path),
                    "raw_log": str(raw_log_path),
                    "git_head_before": before_head,
                    "git_head_after": after_head,
                    "session_id": session_id,
                }
                phase_record["prompts"].append(prompt_record)
                write_text(logs / "summary.json", json.dumps(summary, indent=2) + "\n")
                write_text(logs / "latest_git_status.txt", git_status(repo))

                if rc != 0:
                    print(f"\nSTOP: Codex process exited with {rc} for {prompt_path.name}.")
                    print(f"Final message: {final_path}")
                    print(f"Raw log: {raw_log_path}")
                    return rc if rc != 0 else 1

                if classification == "failure":
                    print(f"\nSTOP: prompt reported failure ({reason}).")
                    print(f"Final message: {final_path}")
                    print(f"Raw log: {raw_log_path}")
                    return 1

                if classification == "unknown" and not args.allow_unknown_success:
                    print(f"\nSTOP: could not prove prompt success ({reason}).")
                    print("First non-empty line:")
                    print(first_nonempty_line(final_text))
                    print(f"Final message: {final_path}")
                    print(f"Raw log: {raw_log_path}")
                    return 1

                print(f"Prompt classified as {classification}: {reason}")

            print(f"=== PHASE COMPLETE: {phase_name} ===")

        finished = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        summary["finished_utc"] = finished
        write_text(logs / "summary.json", json.dumps(summary, indent=2) + "\n")
        write_text(logs / "final_git_status.txt", git_status(repo))
        print("\nALL REQUESTED PHASE PACKAGES COMPLETED")
        print(f"Logs: {logs}")
        return 0
    finally:
        for sig, handler in previous_signal_handlers.items():
            signal.signal(sig, handler)

if __name__ == "__main__":
    raise SystemExit(main())
