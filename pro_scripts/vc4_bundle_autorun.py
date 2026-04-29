#!/usr/bin/env python3
"""
Deterministic VC4 hardware-test bundle autorunner.

This script drives the unattended protocol:

  1. Generate a test bundle from ChatGPT using pro_scripts/new_tab.js or
     pro_scripts/current_tab.js.
  2. Stage GPT_WEB_FILE output first.
  3. Copy staged files into the repo only after path validation.
  4. Run the generated Codex mechanical prompt.
  5. Run vc4-opt, check-vc4, and the hardware runner.
  6. On pass: write a small passed summary and git commit "passed <test>".
  7. On repeated failure: quarantine/remove the test, write an incomplete
     summary, and git commit "incomplete <test>".

The script intentionally does not update catalog.json. Catalog updates remain
manual and factual after successful hardware runs.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


VC4_OPT_FLAGS = [
    "--vc4-verify-emit-contract",
    "--vc4-verify-scheduled-hardware-rules",
    "--vc4-verify-scheduled-adjacent-hazards",
    "--vc4-verify-scheduled-io-spacing",
    "--vc4-verify-scheduled-peripheral-accesses",
    "-o",
    "/dev/null",
]

DEFAULT_CHAT_TIMEOUT_SEC = 35 * 60
DEFAULT_CODEX_TIMEOUT_SEC = 10 * 60
DEFAULT_STEP_TIMEOUT_SEC = 2 * 60
DEFAULT_CHAT_INFRA_RETRIES = 3
DEFAULT_BROWSER_INTERNAL_TIMEOUT_MS = 120000
# Buffer between the Python-side subprocess timeout and the JS-side
# response-settle timeout.  We want the JS driver to finish gracefully and
# write its in-flight marker / finalize the manifest before the Python
# wrapper kills it for being slow.  The buffer covers browser startup,
# paste, and post-response composer-restore steps.  With a 35-minute
# wrapper and a 5-minute buffer, the JS driver has 30 minutes of pure
# response-wait time -- consistent with "wait up to 30 min for ChatGPT".
CHAT_RESPONSE_BUFFER_SEC = 5 * 60

TEST_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class TestSpec:
    name: str
    spec_path: Path
    spec_text: str
    line_no: int


@dataclass(frozen=True)
class ExtraContext:
    label: str
    path: Path
    text: str


@dataclass
class CommandResult:
    stage: str
    cmd: list[str]
    display_cmd: str
    log_path: Path
    exit_code: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class Failure:
    stage: str
    message: str
    command: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False
    log_path: Optional[Path] = None


@dataclass
class AttemptOutcome:
    passed: bool
    failure: Optional[Failure] = None


class DriverError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tail_text(path: Path, max_lines: int = 500, max_chars: int = 60000) -> str:
    if not path.exists():
        return f"<missing file: {path}>"
    text = read_text(path)
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def last_vc4_result_line(run_log: Path) -> str:
    if not run_log.exists():
        return ""
    result = ""
    for line in read_text(run_log).splitlines():
        if "VC4_TEST_RESULT" in line:
            result = line.strip()
    return result


def run_command(
    *,
    repo: Path,
    stage: str,
    cmd: list[str],
    log_path: Path,
    timeout_sec: int,
    display_cmd: Optional[str] = None,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    shown = display_cmd or shlex.join(cmd)

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {shown}\n")
        log.write(f"# cwd: {repo}\n")
        log.write(f"# started_utc: {utc_now_iso()}\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
        )

        timed_out = False
        try:
            exit_code = proc.wait(timeout=timeout_sec)
        except KeyboardInterrupt:
            log.write(f"\n[INTERRUPT] stage={stage} received KeyboardInterrupt; terminating child process group\n")
            log.flush()
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                proc.wait()
            raise
        except subprocess.TimeoutExpired:
            timed_out = True
            log.write(f"\n[TIMEOUT] stage={stage} timeout_sec={timeout_sec}\n")
            log.flush()
            if os.name != "nt":
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                proc.terminate()
            try:
                exit_code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                exit_code = proc.wait()
        log.write(f"\n# finished_utc: {utc_now_iso()}\n")
        log.write(f"# exit_code: {exit_code}\n")
        log.write(f"# timed_out: {int(timed_out)}\n")

    if timed_out and exit_code == 0:
        exit_code = 124
    return CommandResult(
        stage=stage,
        cmd=cmd,
        display_cmd=shown,
        log_path=log_path,
        exit_code=exit_code,
        timed_out=timed_out,
    )


def script_invocation(repo: Path, script_rel: str) -> list[str]:
    script = (repo / script_rel).resolve()
    if not script.exists():
        raise DriverError(f"missing browser script: {script}")
    if os.access(script, os.X_OK):
        return [str(script)]
    return ["node", str(script)]


def git(
    repo: Path,
    args: list[str],
    *,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        text=True,
        capture_output=capture,
    )
    if check and proc.returncode != 0:
        raise DriverError(
            "git command failed: "
            + shlex.join(["git", *args])
            + "\nstdout:\n"
            + (proc.stdout or "")
            + "\nstderr:\n"
            + (proc.stderr or "")
        )
    return proc


def ensure_auto_excluded(repo: Path) -> None:
    exclude = repo / ".git" / "info" / "exclude"
    if not exclude.exists():
        return
    text = read_text(exclude)
    with exclude.open("a", encoding="utf-8") as f:
        if ".vc4_auto/" not in text:
            f.write("\n.vc4_auto/\n")


def git_status_paths(repo: Path) -> list[str]:
    proc = git(repo, ["status", "--porcelain=v1"], check=True, capture=True)
    paths: list[str] = []
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ")[-1]
        paths.append(path)
    return paths


def is_allowed_status_path(path: str, allowed_prefixes: Iterable[str]) -> bool:
    if path.startswith(".vc4_auto/"):
        return True
    for prefix in allowed_prefixes:
        clean = prefix.rstrip("/")
        if path == clean or path.startswith(clean + "/"):
            return True
    return False


def disallowed_status_paths(repo: Path, allowed_prefixes: Iterable[str]) -> list[str]:
    return [
        path
        for path in git_status_paths(repo)
        if not is_allowed_status_path(path, allowed_prefixes)
    ]


def cleanup_disallowed_paths(repo: Path, paths: Iterable[str]) -> None:
    for path in paths:
        if not path or path.startswith(".vc4_auto/"):
            continue
        tracked = git(
            repo,
            ["ls-files", "--error-unmatch", "--", path],
            check=False,
            capture=True,
        ).returncode == 0
        if tracked:
            git(repo, ["restore", "--staged", "--worktree", "--", path], check=False)
            continue

        abs_path = repo / path
        if abs_path.exists() or abs_path.is_symlink():
            if abs_path.is_dir() and not abs_path.is_symlink():
                shutil.rmtree(abs_path)
            else:
                abs_path.unlink(missing_ok=True)


def ensure_clean_repo(repo: Path, *, allow_dirty: bool) -> None:
    if allow_dirty:
        return
    dirty = [
        path for path in git_status_paths(repo)
        if not path.startswith(".vc4_auto/")
    ]
    if dirty:
        raise DriverError(
            "repo is dirty before autorun. Commit/stash changes first, "
            "or pass --allow-dirty if you know what you are doing.\n"
            + "\n".join(dirty[:120])
        )


def git_commit(repo: Path, message: str, add_paths: list[str]) -> None:
    if add_paths:
        existing_or_deleted = []
        for path in add_paths:
            # git add -A can stage deletions even if the path no longer exists.
            existing_or_deleted.append(path)
        git(repo, ["add", "-A", "--", *existing_or_deleted], check=True, capture=True)

    staged_clean = git(
        repo,
        ["diff", "--cached", "--quiet"],
        check=False,
        capture=True,
    ).returncode == 0

    if staged_clean:
        git(repo, ["commit", "--allow-empty", "-m", message], check=True, capture=True)
    else:
        git(repo, ["commit", "-m", message], check=True, capture=True)


def parse_manifest(repo: Path, manifest: Path) -> list[TestSpec]:
    if not manifest.exists():
        raise DriverError(f"missing manifest: {manifest}")

    specs: list[TestSpec] = []
    for line_no, raw in enumerate(read_text(manifest).splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = raw.rstrip("\n").split("\t")
        if len(parts) != 2:
            raise DriverError(
                f"{manifest}:{line_no}: expected exactly two tab-separated fields: "
                "TEST_NAME<TAB>SPEC_PATH"
            )

        name = parts[0].strip()
        spec_path_raw = parts[1].strip()
        if not TEST_NAME_RE.match(name):
            raise DriverError(
                f"{manifest}:{line_no}: invalid test name {name!r}; "
                "use letters, digits, underscore, or hyphen"
            )
        if not spec_path_raw:
            raise DriverError(f"{manifest}:{line_no}: empty spec path")

        spec_path = Path(spec_path_raw)
        if not spec_path.is_absolute():
            spec_path = repo / spec_path

        if not spec_path.exists():
            raise DriverError(f"{manifest}:{line_no}: missing spec file {spec_path}")

        spec_text = read_text(spec_path)
        expected = re.compile(rf"(?m)^TEST_NAME:\s*{re.escape(name)}\s*$")
        if not expected.search(spec_text):
            raise DriverError(
                f"{spec_path}: missing required line 'TEST_NAME: {name}'. "
                "The manifest name and spec TEST_NAME must match."
            )

        specs.append(TestSpec(name=name, spec_path=spec_path, spec_text=spec_text, line_no=line_no))

    return specs


def load_extra_contexts(repo: Path, paths: list[str]) -> list[ExtraContext]:
    contexts: list[ExtraContext] = []
    for raw in paths:
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = repo / path
        if not path.exists():
            raise DriverError(f"missing extra context file: {path}")
        try:
            label = str(path.resolve().relative_to(repo.resolve()))
        except ValueError:
            label = str(path)
        contexts.append(ExtraContext(label=label, path=path, text=read_text(path)))
    return contexts


def format_extra_contexts(contexts: list[ExtraContext]) -> str:
    if not contexts:
        return ""
    chunks = ["# Additional pasted context files\n"]
    for ctx in contexts:
        chunks.append(f"\nBEGIN_EXTRA_CONTEXT_FILE path={ctx.label}\n")
        chunks.append(ctx.text.rstrip())
        chunks.append(f"\nEND_EXTRA_CONTEXT_FILE path={ctx.label}\n")
    return "".join(chunks).rstrip() + "\n"


def state_paths(repo: Path, name: str) -> tuple[Path, Path]:
    base = repo / "vc4_test_specs" / "state"
    return base / "passed" / f"{name}.json", base / "incomplete" / f"{name}.json"


def in_flight_marker_path(repo: Path) -> Path:
    """Path to the JS driver's in-flight prompt marker.

    The marker is written by gpt_web_driver.js as soon as a prompt is
    successfully clicked into ChatGPT.  If a subsequent retry of the same
    prompt finds the marker, the JS driver resumes waiting on the existing
    tab instead of submitting a duplicate prompt.  See gpt_web_driver.js for
    the full lifecycle.
    """
    return repo / ".vc4_auto" / "in_flight_prompt.json"


def clear_in_flight_marker(repo: Path) -> None:
    """Remove a stale in-flight marker.

    Called at test boundaries and when the prompt text we are about to send
    changes.  The marker should never survive across logically distinct
    prompts because the JS driver matches markers by SHA-256 of the prompt
    text, not by a timestamp.
    """
    marker = in_flight_marker_path(repo)
    try:
        if marker.exists():
            marker.unlink()
    except OSError:
        pass


def is_marked_done(repo: Path, name: str) -> Optional[str]:
    passed, incomplete = state_paths(repo, name)
    if passed.exists():
        return "passed"
    if incomplete.exists():
        return "incomplete"
    return None


def reset_state_markers(repo: Path, names: list[str]) -> None:
    if not names:
        return
    ensure_clean_repo(repo, allow_dirty=False)
    changed = False
    for name in names:
        passed, incomplete = state_paths(repo, name)
        for path in (passed, incomplete):
            if path.exists():
                path.unlink()
                changed = True
    if changed:
        git_commit(repo, "reset " + " ".join(names), ["vc4_test_specs/state"])
    else:
        print("No state markers found for reset:", ", ".join(names))


def build_initial_prompt(god_prompt: str, extra_contexts: list[ExtraContext], spec: TestSpec) -> str:
    extra = format_extra_contexts(extra_contexts)
    return (
        god_prompt.rstrip()
        + "\n\n---\n\n"
        + extra
        + "\n---\n\n"
        + "# Automated VC4 hardware-run bundle request\n\n"
        + f"TEST_NAME: {spec.name}\n\n"
        + spec.spec_text.rstrip()
        + "\n"
    )


def build_next_prompt(spec: TestSpec) -> str:
    return (
        "# Automated VC4 hardware-run bundle request\n\n"
        + f"TEST_NAME: {spec.name}\n\n"
        + spec.spec_text.rstrip()
        + "\n"
    )


def stage_tree_listing(stage_dir: Path) -> str:
    if not stage_dir.exists():
        return "<staging directory does not exist>"
    entries = []
    for path in sorted(stage_dir.rglob("*")):
        if path.is_file() or path.is_symlink():
            entries.append(str(path.relative_to(stage_dir)))
    if not entries:
        return "<staging directory is empty>"
    return "\n".join(entries)


def build_failure_prompt(
    *,
    spec: TestSpec,
    attempt_index: int,
    failure: Failure,
    test_root: Path,
) -> str:
    run_log = test_root / "reference" / "run.log"
    failure_log = ""
    if failure.log_path is not None:
        failure_log = tail_text(failure.log_path, max_lines=500, max_chars=70000)

    run_log_tail = ""
    if run_log.exists():
        run_log_tail = tail_text(run_log, max_lines=700, max_chars=90000)

    return (
        f"The generated bundle for test {spec.name} failed.\n\n"
        "Use the same GPT_WEB_FILE format as before. Emit drop-in replacement files "
        "only for this test. Include a replacement "
        f"{spec.name}_codex_mechanical_prompt.md. If no mechanical changes are needed, "
        "emit a no-op mechanical prompt that says so.\n\n"
        "Do not touch catalog.json.\n"
        "Do not touch compiler implementation files.\n"
        "Do not touch CMake files.\n"
        "Do not touch dialect tests.\n"
        "Do not redesign broadly.\n"
        "Diagnose from the logs below and provide a surgical fix.\n\n"
        "Original test specification:\n"
        f"{spec.spec_text.rstrip()}\n\n"
        f"Attempt: {attempt_index}\n"
        f"Failed stage: {failure.stage}\n"
        f"Command: {failure.command or '<none>'}\n"
        f"Exit code: {failure.exit_code if failure.exit_code is not None else '<none>'}\n"
        f"Timed out: {int(failure.timed_out)}\n"
        f"Failure message: {failure.message}\n\n"
        "Relevant log:\n"
        f"{failure_log if failure_log else '<no command log>'}\n\n"
        "reference/run.log tail, if present:\n"
        f"{run_log_tail if run_log_tail else '<no reference/run.log>'}\n\n"
        "Required output:\n"
        "Emit GPT_WEB_FILE blocks for replacement files. Keep paths relative. "
        "Include final newlines. Include a replacement mechanical prompt. "
        "Then provide concise rerun commands.\n"
    )


def safe_stage_rel(path: Path, stage_dir: Path) -> Path:
    rel_path = path.relative_to(stage_dir)
    if rel_path.is_absolute():
        raise DriverError(f"absolute staged path is not allowed: {rel_path}")
    if any(part in ("..", ".git", "") for part in rel_path.parts):
        raise DriverError(f"unsafe staged path is not allowed: {rel_path}")
    return rel_path


def copy_staged_files_into_repo(stage_dir: Path, repo: Path) -> int:
    if not stage_dir.exists():
        raise DriverError(f"staging directory does not exist: {stage_dir}")

    copied = 0
    for src in sorted(stage_dir.rglob("*")):
        if src.is_dir():
            continue
        if src.is_symlink():
            raise DriverError(f"symlink in staged output is not allowed: {src}")
        rel_path = safe_stage_rel(src, stage_dir)
        if rel_path.parts and rel_path.parts[0] == ".gpt-web-run":
            continue
        dst = repo / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    if copied == 0:
        raise DriverError("no generated files were produced in staging output outside .gpt-web-run metadata")
    return copied


def required_material_paths(repo: Path, name: str) -> list[Path]:
    root = repo / "compiler/test/CodeGen/VC4/Hardware/Run" / name
    ref = root / "reference"
    return [
        root / "README.md",
        root / "input.mlir",
        root / "expected.json",
        root / "candidate" / "README.md",
        ref / ".gitignore",
        ref / f"{name}_harness.c",
        ref / f"{name}.qasm",
        ref / f"{name}_launch.c",
        ref / f"{name}_launch.h",
        repo / f"{name}_codex_mechanical_prompt.md",
    ]


def required_mechanical_paths(repo: Path, name: str) -> list[Path]:
    root = repo / "compiler/test/CodeGen/VC4/Hardware/Run" / name
    ref = root / "reference"
    return [
        ref / "Makefile",
        ref / "run.sh",
        ref / "mailbox.c",
        ref / "mailbox.h",
        root / "share" / "vc4tmpl" / "template.c",
        root / "share" / "vc4tmpl" / "template.h",
        root / "share" / "vc4inc" / "vc4.qinc",
    ]


def validate_paths_exist(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise DriverError("missing required generated files:\n" + "\n".join(missing))


def clean_test_transients(repo: Path, name: str) -> None:
    ref = repo / "compiler/test/CodeGen/VC4/Hardware/Run" / name / "reference"
    if not ref.exists():
        return

    shutil.rmtree(ref / "objs", ignore_errors=True)

    patterns = [
        "*.o",
        "*.d",
        "*.elf",
        "*.bin",
        "*.list",
        "*shader.c",
        "*shader.h",
        "run.log",
    ]
    for pattern in patterns:
        for path in ref.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


def write_auto_status(repo: Path, specs: list[TestSpec]) -> None:
    auto = repo / ".vc4_auto"
    passed_names = []
    incomplete_names = []
    pending_names = []
    for spec in specs:
        marker = is_marked_done(repo, spec.name)
        if marker == "passed":
            passed_names.append(spec.name)
        elif marker == "incomplete":
            incomplete_names.append(spec.name)
        else:
            pending_names.append(spec.name)

    status = {
        "updated_utc": utc_now_iso(),
        "passed": passed_names,
        "incomplete": incomplete_names,
        "pending": pending_names,
    }
    write_text(auto / "status.json", json.dumps(status, indent=2) + "\n")


def validation_failure(
    *,
    stage: str,
    message: str,
    log_path: Path,
    extra: str,
) -> Failure:
    write_text(log_path, message.rstrip() + "\n\n" + extra.rstrip() + "\n")
    return Failure(stage=stage, message=message, log_path=log_path)


def run_attempt(
    *,
    repo: Path,
    auto: Path,
    spec: TestSpec,
    attempt_index: int,
    prompt_text: str,
    use_new_tab: bool,
    new_tab_script: str,
    current_tab_script: str,
    chat_timeout_sec: int,
    codex_timeout_sec: int,
    step_timeout_sec: int,
    browser_internal_timeout_ms: int,
) -> AttemptOutcome:
    test_root = repo / "compiler/test/CodeGen/VC4/Hardware/Run" / spec.name
    attempt_name = f"attempt-{attempt_index:02d}"
    prompt_dir = auto / "prompts" / spec.name
    stage_dir = auto / "staging" / spec.name / attempt_name
    log_dir = auto / "logs" / spec.name / attempt_name

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = prompt_dir / f"{attempt_name}.md"
    write_text(prompt_path, prompt_text)

    browser_script = new_tab_script if use_new_tab else current_tab_script
    # Give the JS driver a slightly tighter response-wait deadline than the
    # Python-side subprocess timeout so it can finish gracefully (write its
    # in-flight marker, flush the manifest, etc.) before Python forcibly
    # kills the process.
    js_response_timeout_ms = max(
        60_000,
        (chat_timeout_sec - CHAT_RESPONSE_BUFFER_SEC) * 1000,
    )
    chat_cmd = script_invocation(repo, browser_script) + [
        "--prompt-file",
        str(prompt_path.resolve()),
        "--out",
        str(stage_dir.resolve()),
        "--repo-root",
        str(repo.resolve()),
        "--connect-timeout-ms",
        str(browser_internal_timeout_ms),
        "--page-timeout-ms",
        str(browser_internal_timeout_ms),
        "--prompt-timeout-ms",
        str(browser_internal_timeout_ms),
        "--response-timeout-ms",
        str(js_response_timeout_ms),
    ]

    chat_result = run_command(
        repo=repo,
        stage="chat",
        cmd=chat_cmd,
        log_path=log_dir / "chat.log",
        timeout_sec=chat_timeout_sec,
    )
    if not chat_result.ok:
        return AttemptOutcome(
            passed=False,
            failure=Failure(
                stage="chat",
                message="ChatGPT browser invocation failed",
                command=chat_result.display_cmd,
                exit_code=chat_result.exit_code,
                timed_out=chat_result.timed_out,
                log_path=chat_result.log_path,
            ),
        )

    try:
        copy_staged_files_into_repo(stage_dir, repo)
        validate_paths_exist(required_material_paths(repo, spec.name))
    except Exception as exc:
        log_path = log_dir / "chat_output_validation.log"
        failure = validation_failure(
            stage="chat-output-validation",
            message=str(exc),
            log_path=log_path,
            extra="Staged files:\n" + stage_tree_listing(stage_dir),
        )
        return AttemptOutcome(passed=False, failure=failure)

    codex_prompt = repo / f"{spec.name}_codex_mechanical_prompt.md"
    codex_text = read_text(codex_prompt)
    codex_cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        codex_text,
    ]
    codex_result = run_command(
        repo=repo,
        stage="codex",
        cmd=codex_cmd,
        display_cmd=(
            "codex exec --dangerously-bypass-approvals-and-sandbox "
            f"$(cat {spec.name}_codex_mechanical_prompt.md)"
        ),
        log_path=log_dir / "codex.log",
        timeout_sec=codex_timeout_sec,
    )
    if not codex_result.ok:
        return AttemptOutcome(
            passed=False,
            failure=Failure(
                stage="codex",
                message="Codex mechanical prompt failed",
                command=codex_result.display_cmd,
                exit_code=codex_result.exit_code,
                timed_out=codex_result.timed_out,
                log_path=codex_result.log_path,
            ),
        )

    codex_prompt.unlink(missing_ok=True)

    try:
        validate_paths_exist(required_mechanical_paths(repo, spec.name))
        run_sh = test_root / "reference" / "run.sh"
        run_sh.chmod(run_sh.stat().st_mode | 0o111)
    except Exception as exc:
        log_path = log_dir / "codex_output_validation.log"
        failure = validation_failure(
            stage="codex-output-validation",
            message=str(exc),
            log_path=log_path,
            extra="Current test tree:\n" + stage_tree_listing(test_root),
        )
        return AttemptOutcome(passed=False, failure=failure)

    clean_test_transients(repo, spec.name)

    allowed = [
        f"compiler/test/CodeGen/VC4/Hardware/Run/{spec.name}",
        "vc4_test_specs/state",
    ]
    bad = disallowed_status_paths(repo, allowed)
    if bad:
        cleanup_disallowed_paths(repo, bad)
        log_path = log_dir / "path_guard.log"
        failure = validation_failure(
            stage="path-guard",
            message="Generated/Codex step touched disallowed paths; they were restored/removed",
            log_path=log_path,
            extra="\n".join(bad),
        )
        return AttemptOutcome(passed=False, failure=failure)

    input_mlir = test_root / "input.mlir"
    vc4_opt_cmd = [
        "compiler/build/bin/vc4-opt",
        str(input_mlir.relative_to(repo)),
        *VC4_OPT_FLAGS,
    ]
    vc4_opt_result = run_command(
        repo=repo,
        stage="vc4-opt",
        cmd=vc4_opt_cmd,
        log_path=log_dir / "vc4_opt.log",
        timeout_sec=step_timeout_sec,
    )
    if not vc4_opt_result.ok:
        return AttemptOutcome(
            passed=False,
            failure=Failure(
                stage="vc4-opt",
                message="vc4-opt verifier failed",
                command=vc4_opt_result.display_cmd,
                exit_code=vc4_opt_result.exit_code,
                timed_out=vc4_opt_result.timed_out,
                log_path=vc4_opt_result.log_path,
            ),
        )

    check_cmd = ["cmake", "--build", "compiler/build", "--target", "check-vc4"]
    check_result = run_command(
        repo=repo,
        stage="check-vc4",
        cmd=check_cmd,
        log_path=log_dir / "check_vc4.log",
        timeout_sec=step_timeout_sec,
    )
    if not check_result.ok:
        return AttemptOutcome(
            passed=False,
            failure=Failure(
                stage="check-vc4",
                message="check-vc4 failed",
                command=check_result.display_cmd,
                exit_code=check_result.exit_code,
                timed_out=check_result.timed_out,
                log_path=check_result.log_path,
            ),
        )

    hardware_cmd = [
        "compiler/test/CodeGen/VC4/Support/run_hardware_test.sh",
        f"compiler/test/CodeGen/VC4/Hardware/Run/{spec.name}",
        "reference",
    ]
    hardware_result = run_command(
        repo=repo,
        stage="hardware",
        cmd=hardware_cmd,
        log_path=log_dir / "hardware.log",
        timeout_sec=step_timeout_sec,
    )
    if not hardware_result.ok:
        return AttemptOutcome(
            passed=False,
            failure=Failure(
                stage="hardware",
                message="hardware runner failed",
                command=hardware_result.display_cmd,
                exit_code=hardware_result.exit_code,
                timed_out=hardware_result.timed_out,
                log_path=hardware_result.log_path,
            ),
        )

    return AttemptOutcome(passed=True)


def mark_passed(
    *,
    repo: Path,
    auto: Path,
    spec: TestSpec,
    attempts_used: int,
    specs: list[TestSpec],
) -> None:
    test_root = repo / "compiler/test/CodeGen/VC4/Hardware/Run" / spec.name
    run_log = test_root / "reference" / "run.log"
    result_line = last_vc4_result_line(run_log)

    passed_auto = auto / "passed" / spec.name
    passed_auto.mkdir(parents=True, exist_ok=True)
    if run_log.exists():
        shutil.copy2(run_log, passed_auto / "reference_run.log")

    summary = {
        "name": spec.name,
        "status": "passed",
        "completed_utc": utc_now_iso(),
        "attempts_used": attempts_used,
        "test_dir": f"compiler/test/CodeGen/VC4/Hardware/Run/{spec.name}",
        "expected_json": f"compiler/test/CodeGen/VC4/Hardware/Run/{spec.name}/expected.json",
        "reference_run_log_archive": f".vc4_auto/passed/{spec.name}/reference_run.log",
        "result_line": result_line,
    }

    passed_state, _ = state_paths(repo, spec.name)
    write_text(passed_state, json.dumps(summary, indent=2) + "\n")

    clean_test_transients(repo, spec.name)

    allowed = [
        f"compiler/test/CodeGen/VC4/Hardware/Run/{spec.name}",
        "vc4_test_specs/state",
    ]
    bad = disallowed_status_paths(repo, allowed)
    if bad:
        cleanup_disallowed_paths(repo, bad)

    git_commit(
        repo,
        f"passed {spec.name}",
        [
            f"compiler/test/CodeGen/VC4/Hardware/Run/{spec.name}",
            f"vc4_test_specs/state/passed/{spec.name}.json",
        ],
    )
    write_auto_status(repo, specs)


def archive_and_mark_incomplete(
    *,
    repo: Path,
    auto: Path,
    spec: TestSpec,
    attempts_used: int,
    failure: Failure,
    final_failure_prompt: str,
    specs: list[TestSpec],
) -> None:
    test_root = repo / "compiler/test/CodeGen/VC4/Hardware/Run" / spec.name
    incomplete_auto = auto / "incomplete" / spec.name
    incomplete_auto.mkdir(parents=True, exist_ok=True)

    if test_root.exists():
        archive_dir = incomplete_auto / "archived_test_dir"
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        shutil.copytree(test_root, archive_dir)

    write_text(incomplete_auto / "final_failure_prompt.md", final_failure_prompt)

    summary = {
        "name": spec.name,
        "status": "incomplete",
        "completed_utc": utc_now_iso(),
        "attempts_used": attempts_used,
        "failed_stage": failure.stage,
        "message": failure.message,
        "command": failure.command,
        "exit_code": failure.exit_code,
        "timed_out": failure.timed_out,
        "failure_log_archive": f".vc4_auto/logs/{spec.name}",
        "archived_test_dir": f".vc4_auto/incomplete/{spec.name}/archived_test_dir",
        "final_failure_prompt": f".vc4_auto/incomplete/{spec.name}/final_failure_prompt.md",
    }

    _, incomplete_state = state_paths(repo, spec.name)
    write_text(incomplete_state, json.dumps(summary, indent=2) + "\n")

    if test_root.exists():
        shutil.rmtree(test_root)

    root_prompt = repo / f"{spec.name}_codex_mechanical_prompt.md"
    root_prompt.unlink(missing_ok=True)

    allowed = [
        f"compiler/test/CodeGen/VC4/Hardware/Run/{spec.name}",
        "vc4_test_specs/state",
    ]
    bad = disallowed_status_paths(repo, allowed)
    if bad:
        cleanup_disallowed_paths(repo, bad)

    git_commit(
        repo,
        f"incomplete {spec.name}",
        [
            f"compiler/test/CodeGen/VC4/Hardware/Run/{spec.name}",
            f"vc4_test_specs/state/incomplete/{spec.name}.json",
        ],
    )
    write_auto_status(repo, specs)


def run_one_test(
    *,
    repo: Path,
    auto: Path,
    spec: TestSpec,
    specs: list[TestSpec],
    god_prompt: str,
    extra_contexts: list[ExtraContext],
    have_context_tab: bool,
    new_tab_script: str,
    current_tab_script: str,
    max_fixes: int,
    chat_timeout_sec: int,
    codex_timeout_sec: int,
    step_timeout_sec: int,
    chat_infra_retries: int,
    browser_internal_timeout_ms: int,
) -> tuple[str, bool]:
    """
    Returns (status, have_context_tab_after), where status is passed/incomplete.

    Chat infrastructure failures are retried with the SAME prompt text.  This
    is critical for correctness: gpt_web_driver.js writes an in-flight marker
    keyed on the SHA-256 of the prompt as soon as the prompt is clicked into
    ChatGPT.  When the next attempt runs the same prompt, the JS driver sees
    the marker and resumes waiting on the existing tab instead of submitting
    the prompt a second time.  Submitting again while ChatGPT is still
    reasoning was the failure mode this design eliminates.

    Whenever the prompt text changes (initial -> failure-fix prompt), we
    proactively clear the marker so the JS driver does not try to resume on
    a stale tab.
    """
    # Starting a fresh logical test: any leftover marker from a previous
    # invocation (different test, crashed run, etc.) is stale.  Clear it.
    clear_in_flight_marker(repo)

    attempt_index = 0
    fixes_used = 0
    use_new_tab = not have_context_tab
    prompt_text = (
        build_initial_prompt(god_prompt, extra_contexts, spec)
        if use_new_tab
        else build_next_prompt(spec)
    )

    last_failure: Optional[Failure] = None
    chat_infra_failures = 0

    while True:
        marker_present = in_flight_marker_path(repo).exists()
        print(
            f"[vc4-auto] test={spec.name} attempt={attempt_index} "
            f"script={'new_tab' if use_new_tab else 'current_tab'}"
            + (" (resume)" if marker_present else "")
        )
        outcome = run_attempt(
            repo=repo,
            auto=auto,
            spec=spec,
            attempt_index=attempt_index,
            prompt_text=prompt_text,
            use_new_tab=use_new_tab,
            new_tab_script=new_tab_script,
            current_tab_script=current_tab_script,
            chat_timeout_sec=chat_timeout_sec,
            codex_timeout_sec=codex_timeout_sec,
            step_timeout_sec=step_timeout_sec,
            browser_internal_timeout_ms=browser_internal_timeout_ms,
        )

        if outcome.passed:
            print(f"[vc4-auto] PASS {spec.name}")
            clear_in_flight_marker(repo)
            mark_passed(
                repo=repo,
                auto=auto,
                spec=spec,
                attempts_used=attempt_index + 1,
                specs=specs,
            )
            return "passed", True

        assert outcome.failure is not None
        last_failure = outcome.failure

        if last_failure.stage == "chat":
            chat_infra_failures += 1
            marker_now = in_flight_marker_path(repo).exists()
            print(
                f"[vc4-auto] CHAT INFRA FAILURE {spec.name} "
                f"retry={chat_infra_failures}/{chat_infra_retries} "
                f"marker={'present (will resume)' if marker_now else 'absent (will resubmit)'}"
            )
            if chat_infra_failures <= chat_infra_retries:
                attempt_index += 1
                # Retry the same model prompt. If the marker is still on
                # disk, the JS driver will detect it (same prompt hash) and
                # resume waiting on the existing tab instead of submitting
                # the prompt a second time.  If the marker is absent, the
                # JS driver will treat this as a fresh submission.
                continue
            clear_in_flight_marker(repo)
            raise DriverError(
                f"ChatGPT browser automation failed {chat_infra_failures} times for {spec.name}; "
                "aborting without marking the test incomplete. Check Chrome remote debugging, login state, "
                "and .vc4_auto/logs for screenshots/answers."
            )

        chat_infra_failures = 0
        # Any non-chat failure means we have an answer (or at least a non-
        # infra problem) and we are about to switch prompts.  The previous
        # prompt's marker is now stale.
        clear_in_flight_marker(repo)
        print(
            f"[vc4-auto] FAIL {spec.name} stage={last_failure.stage} "
            f"fixes_used={fixes_used}/{max_fixes}"
        )

        if fixes_used < max_fixes:
            fixes_used += 1
            attempt_index += 1
            use_new_tab = False
            prompt_text = build_failure_prompt(
                spec=spec,
                attempt_index=attempt_index,
                failure=last_failure,
                test_root=repo / "compiler/test/CodeGen/VC4/Hardware/Run" / spec.name,
            )
            continue

        final_prompt = build_failure_prompt(
            spec=spec,
            attempt_index=attempt_index,
            failure=last_failure,
            test_root=repo / "compiler/test/CodeGen/VC4/Hardware/Run" / spec.name,
        )
        print(f"[vc4-auto] INCOMPLETE {spec.name}")
        clear_in_flight_marker(repo)
        archive_and_mark_incomplete(
            repo=repo,
            auto=auto,
            spec=spec,
            attempts_used=attempt_index + 1,
            failure=last_failure,
            final_failure_prompt=final_prompt,
            specs=specs,
        )
        return "incomplete", False


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repo root, default: current directory")
    parser.add_argument("--manifest", default="vc4_test_specs/manifest.tsv")
    parser.add_argument("--god-prompt", default="vc4_test_specs/god_prompt.md")
    parser.add_argument("--extra-context-file", action="append", default=[], help="extra file pasted into every fresh-tab prompt, e.g. compiler/dialect.txt")
    parser.add_argument("--new-tab", default="pro_scripts/new_tab.js")
    parser.add_argument("--current-tab", default="pro_scripts/current_tab.js")
    parser.add_argument("--max-fixes", type=int, default=3)
    parser.add_argument("--chat-timeout-sec", type=int, default=DEFAULT_CHAT_TIMEOUT_SEC)
    parser.add_argument("--codex-timeout-sec", type=int, default=DEFAULT_CODEX_TIMEOUT_SEC)
    parser.add_argument("--step-timeout-sec", type=int, default=DEFAULT_STEP_TIMEOUT_SEC)
    parser.add_argument("--chat-infra-retries", type=int, default=DEFAULT_CHAT_INFRA_RETRIES,
                        help="browser/CDP/UI retries before aborting without marking incomplete")
    parser.add_argument("--browser-internal-timeout-ms", type=int, default=DEFAULT_BROWSER_INTERNAL_TIMEOUT_MS,
                        help="timeout passed into new_tab/current_tab for CDP, page, and prompt operations")
    parser.add_argument("--only", default="", help="comma-separated test names to run")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--reset-test",
        action="append",
        default=[],
        help="remove passed/incomplete marker for a test and commit the reset, then exit unless --continue-after-reset is set",
    )
    parser.add_argument("--continue-after-reset", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    if not (repo / ".git").exists():
        raise DriverError(f"not a git repo root: {repo}")

    os.chdir(repo)
    ensure_auto_excluded(repo)

    if args.reset_test:
        reset_state_markers(repo, args.reset_test)
        if not args.continue_after_reset:
            return 0

    ensure_clean_repo(repo, allow_dirty=args.allow_dirty)

    manifest = (repo / args.manifest).resolve()
    god_prompt_path = (repo / args.god_prompt).resolve()
    if not god_prompt_path.exists():
        raise DriverError(f"missing god prompt file: {god_prompt_path}")

    god_prompt = read_text(god_prompt_path)
    if "REPLACE_WITH_THE_GOD_PROMPT" in god_prompt:
        raise DriverError(
            f"{god_prompt_path} still contains the placeholder marker. "
            "Paste the full god prompt into that file before running."
        )

    extra_contexts = load_extra_contexts(repo, args.extra_context_file)
    if extra_contexts:
        print("[vc4-auto] fresh-tab extra context files:")
        for ctx in extra_contexts:
            print("  -", ctx.label)

    specs = parse_manifest(repo, manifest)
    only = {name.strip() for name in args.only.split(",") if name.strip()}
    if only:
        specs = [spec for spec in specs if spec.name in only]

    auto = repo / ".vc4_auto"
    auto.mkdir(parents=True, exist_ok=True)
    write_auto_status(repo, specs)

    if not specs:
        print("[vc4-auto] manifest has no runnable tests")
        return 0

    have_context_tab = False

    for spec in specs:
        marker = is_marked_done(repo, spec.name)
        if marker:
            print(f"[vc4-auto] SKIP {spec.name}: already marked {marker}")
            continue

        status, have_context_tab = run_one_test(
            repo=repo,
            auto=auto,
            spec=spec,
            specs=specs,
            god_prompt=god_prompt,
            extra_contexts=extra_contexts,
            have_context_tab=have_context_tab,
            new_tab_script=args.new_tab,
            current_tab_script=args.current_tab,
            max_fixes=args.max_fixes,
            chat_timeout_sec=args.chat_timeout_sec,
            codex_timeout_sec=args.codex_timeout_sec,
            step_timeout_sec=args.step_timeout_sec,
            chat_infra_retries=args.chat_infra_retries,
            browser_internal_timeout_ms=args.browser_internal_timeout_ms,
        )
        print(f"[vc4-auto] DONE {spec.name}: {status}")

    write_auto_status(repo, specs)
    print("[vc4-auto] all manifest entries are processed or skipped")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DriverError as exc:
        print(f"[vc4-auto] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
