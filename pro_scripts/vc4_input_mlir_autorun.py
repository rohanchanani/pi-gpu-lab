#!/usr/bin/env python3
"""
Autorun generation of VC4 hardware-run input.mlir files.

This is the narrowed companion to the VC4 bundle autorunner:

  * input per test: spec + existing qasm/launch.c/launch.h/harness.c bundle
  * model output per test: exactly one replacement input.mlir
  * verification: vc4-opt verifier command, then check-vc4
  * no Codex step, no hardware runner, no catalog updates
  * default browser mode: current ChatGPT tab, with no god prompt
  * --new: open a new tab and send the god prompt only for the first runnable test

Manifest format:

    TEST_NAME<TAB>SPEC_PATH

The spec path may be absolute or relative to the repo root.  The script also
collects trusted files from:

    compiler/test/CodeGen/VC4/Hardware/Run/TEST_NAME/

including README.md, expected.json, current input.mlir, and reference qasm,
launcher .c/.h, and harness .c.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VC4_OPT_FLAGS = [
    "--vc4-verify-emit-contract",
    "--vc4-verify-scheduled-hardware-rules",
    "--vc4-verify-scheduled-adjacent-hazards",
    "--vc4-verify-scheduled-io-spacing",
    "--vc4-verify-scheduled-peripheral-accesses",
    "-o",
    "/dev/null",
]

RUN_ROOT = Path("compiler/test/CodeGen/VC4/Hardware/Run")
STATE_ROOT = Path("vc4_input_mlir_specs/state")
AUTO_ROOT = Path(".vc4_auto/input_mlir")
TEST_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")

DEFAULT_CHAT_TIMEOUT_SEC = 35 * 60
DEFAULT_RESPONSE_BUFFER_SEC = 5 * 60
DEFAULT_STEP_TIMEOUT_SEC = 5 * 60
DEFAULT_BROWSER_TIMEOUT_MS = 120000
DEFAULT_MAX_FILE_CHARS = 350_000


@dataclass(frozen=True)
class Spec:
    name: str
    path: Path
    text: str
    line_no: int


@dataclass(frozen=True)
class BundleFile:
    role: str
    path: Path
    rel_path: str
    text: str


@dataclass
class Failure:
    stage: str
    message: str
    command: str = ""
    exit_code: Optional[int] = None
    timed_out: bool = False
    log_path: Optional[Path] = None


class DriverError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def relpath(repo: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        return str(path)


def tail_text(path: Optional[Path], max_lines: int = 300, max_chars: int = 50000) -> str:
    if path is None:
        return "<no log path>"
    if not path.exists():
        return f"<missing: {path}>"
    text = read_text(path)
    text = "\n".join(text.splitlines()[-max_lines:])
    if len(text) > max_chars:
        text = text[-max_chars:]
    return text


def print_log_tail(path: Path, label: str, lines: int = 80) -> None:
    if not path.exists():
        print(f"[vc4-input-mlir-auto] {label}: no log at {path}")
        return
    data = read_text(path).splitlines()
    print(f"[vc4-input-mlir-auto] {label}: last {min(lines, len(data))} lines from {path}")
    for line in data[-lines:]:
        print(f"[vc4-input-mlir-auto] | {line}")


def terminate_process(proc: subprocess.Popen[str]) -> int:
    if os.name != "nt":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        proc.terminate()
    try:
        return proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            proc.kill()
        return proc.wait()


def run_logged(repo: Path, stage: str, cmd: list[str], log: Path, timeout_sec: int, shown: Optional[str] = None) -> Optional[Failure]:
    shown = shown or shlex.join(cmd)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[vc4-input-mlir-auto] {stage}: $ {shown}")
    print(f"[vc4-input-mlir-auto] {stage}: log -> {log}")
    start = time.time()
    timed_out = False
    with log.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"$ {shown}\n# cwd: {repo}\n# started_utc: {utc_now()}\n# timeout_sec: {timeout_sec}\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
        )
        try:
            code = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            f.write(f"\n[TIMEOUT] {stage}\n")
            f.flush()
            code = terminate_process(proc)
        elapsed = time.time() - start
        f.write(f"\n# finished_utc: {utc_now()}\n# exit_code: {code}\n# timed_out: {int(timed_out)}\n# elapsed_sec: {elapsed:.1f}\n")
    if timed_out and code == 0:
        code = 124
    print(f"[vc4-input-mlir-auto] {stage}: exit={code} timed_out={int(timed_out)} elapsed={elapsed:.1f}s")
    if code != 0 or timed_out:
        print_log_tail(log, stage)
        return Failure(stage=stage, message=f"{stage} failed", command=shown, exit_code=code, timed_out=timed_out, log_path=log)
    return None


def git(repo: Path, args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(["git", *args], cwd=str(repo), text=True, capture_output=True)
    if check and p.returncode:
        raise DriverError(f"git {' '.join(args)} failed\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}")
    return p


def ensure_git_repo(repo: Path) -> None:
    if not (repo / ".git").exists():
        raise DriverError(f"not a git repo root: {repo}")


def ensure_clean_repo(repo: Path, allow_dirty: bool) -> None:
    if allow_dirty:
        return
    status = git(repo, ["status", "--porcelain=v1"]).stdout.splitlines()
    dirty = []
    for line in status:
        path = line[3:]
        if path.startswith(".vc4_auto/"):
            continue
        dirty.append(line)
    if dirty:
        raise DriverError("repo is dirty; commit/stash first or pass --allow-dirty\n" + "\n".join(dirty[:120]))


def ensure_git_exclude(repo: Path) -> None:
    p = repo / ".git/info/exclude"
    if not p.exists():
        return
    text = read_text(p)
    if ".vc4_auto/" not in text:
        with p.open("a", encoding="utf-8") as f:
            f.write("\n.vc4_auto/\n")


def git_commit(repo: Path, message: str, paths: list[str], no_commit: bool) -> None:
    if no_commit:
        print(f"[vc4-input-mlir-auto] --no-commit: leaving changes unstaged for {message!r}")
        return
    stage_paths: list[str] = []
    for raw in paths:
        p = repo / raw
        tracked = git(repo, ["ls-files", "--", raw], check=False).stdout.strip()
        if p.exists() or tracked:
            stage_paths.append(raw)
    if stage_paths:
        git(repo, ["add", "-A", "--", *stage_paths])
    cached_clean = git(repo, ["diff", "--cached", "--quiet"], check=False).returncode == 0
    if cached_clean:
        git(repo, ["commit", "--allow-empty", "-m", message])
    else:
        git(repo, ["commit", "-m", message])


def parse_manifest(repo: Path, manifest: Path) -> list[Spec]:
    if not manifest.exists():
        raise DriverError(f"missing manifest: {manifest}")
    specs: list[Spec] = []
    for line_no, raw in enumerate(read_text(manifest).splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 2:
            raise DriverError(f"{manifest}:{line_no}: expected TEST_NAME<TAB>SPEC_PATH")
        name = parts[0].strip()
        spec_path = parts[1].strip()
        if not TEST_NAME_RE.match(name):
            raise DriverError(f"{manifest}:{line_no}: invalid test name {name!r}")
        path = Path(spec_path)
        if not path.is_absolute():
            path = repo / path
        if not path.exists():
            raise DriverError(f"{manifest}:{line_no}: missing spec file {path}")
        specs.append(Spec(name=name, path=path, text=read_text(path), line_no=line_no))
    return specs


def state_paths(repo: Path, name: str) -> tuple[Path, Path]:
    return repo / STATE_ROOT / "passed" / f"{name}.json", repo / STATE_ROOT / "incomplete" / f"{name}.json"


def state_status(repo: Path, name: str) -> Optional[str]:
    passed, incomplete = state_paths(repo, name)
    if passed.exists():
        return "passed"
    if incomplete.exists():
        return "incomplete"
    return None


def reset_state(repo: Path, names: list[str], no_commit: bool) -> None:
    changed = False
    for name in names:
        for p in state_paths(repo, name):
            if p.exists():
                p.unlink()
                changed = True
    if changed:
        git_commit(repo, "reset input-mlir " + " ".join(names), [str(STATE_ROOT)], no_commit)


def write_status(repo: Path, specs: list[Spec]) -> None:
    passed: list[str] = []
    incomplete: list[str] = []
    pending: list[str] = []
    for spec in specs:
        status = state_status(repo, spec.name)
        if status == "passed":
            passed.append(spec.name)
        elif status == "incomplete":
            incomplete.append(spec.name)
        else:
            pending.append(spec.name)
    write_text(
        repo / AUTO_ROOT / "status.json",
        json.dumps({"updated_utc": utc_now(), "passed": passed, "incomplete": incomplete, "pending": pending}, indent=2) + "\n",
    )


def choose_one(kind: str, root: Path, preferred: list[str], patterns: list[str]) -> Path:
    candidates: list[Path] = []
    for name in preferred:
        p = root / name
        if p.is_file() and p not in candidates:
            candidates.append(p)
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            if p.is_file() and p not in candidates:
                candidates.append(p)
    if not candidates:
        raise DriverError(f"missing {kind} under {root}")
    if len(candidates) != 1:
        raise DriverError(f"ambiguous {kind} under {root}:\n" + "\n".join(str(p) for p in candidates))
    return candidates[0]


def read_limited(path: Path, max_chars: int) -> str:
    text = read_text(path)
    if len(text) > max_chars:
        raise DriverError(f"context file too large ({len(text)} > {max_chars} chars): {path}; increase --max-file-chars")
    return text


def collect_bundle_files(repo: Path, name: str, max_chars: int) -> list[BundleFile]:
    root = repo / RUN_ROOT / name
    ref = root / "reference"
    if not ref.is_dir():
        raise DriverError(f"missing reference dir for {name}: {ref}")
    dashed = name.replace("_", "-")
    required = [
        ("reference_qasm", choose_one("qasm", ref, [f"{name}.qasm"], ["*.qasm"])),
        ("launcher_c", choose_one("launcher .c", ref, [f"{name}_launch.c", "_launch.c"], ["*_launch.c"])),
        ("launcher_h", choose_one("launcher .h", ref, [f"{name}_launch.h", "_launch.h"], ["*_launch.h"])),
        ("harness_c", choose_one("harness .c", ref, ["_harness.c", f"{name}_harness.c", f"3-test-{dashed}.c"], ["*harness.c", "3-test-*.c"])),
    ]
    optional = [
        ("bundle_readme", root / "README.md"),
        ("expected_json", root / "expected.json"),
        ("current_input_mlir", root / "input.mlir"),
    ]
    out: list[BundleFile] = []
    for role, path in optional + required:
        if path.exists():
            out.append(BundleFile(role=role, path=path, rel_path=relpath(repo, path), text=read_limited(path, max_chars)))
    return out


def load_extra_context(repo: Path, paths: list[str], max_chars: int) -> str:
    if not paths:
        return ""
    chunks = ["# Additional dialect/source/example context files\n"]
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = repo / p
        if not p.exists():
            raise DriverError(f"missing extra context file: {p}")
        rp = relpath(repo, p)
        chunks.append(f"\nBEGIN_EXTRA_CONTEXT_FILE path={rp}\n")
        chunks.append(read_limited(p, max_chars).rstrip())
        chunks.append(f"\nEND_EXTRA_CONTEXT_FILE path={rp}\n")
    return "".join(chunks).rstrip() + "\n"


def format_bundle_context(files: list[BundleFile]) -> str:
    chunks = ["# Existing trusted bundle files\n"]
    for f in files:
        chunks.append(f"\nBEGIN_VC4_BUNDLE_FILE role={f.role} path={f.rel_path}\n")
        chunks.append(f.text.rstrip())
        chunks.append(f"\nEND_VC4_BUNDLE_FILE role={f.role} path={f.rel_path}\n")
    return "".join(chunks).rstrip() + "\n"


def test_prompt(spec: Spec, files: list[BundleFile]) -> str:
    output = f"{RUN_ROOT}/{spec.name}/input.mlir"
    return (
        "# Automated VC4 input.mlir generation request\n\n"
        f"TEST_NAME: {spec.name}\n"
        f"OUTPUT_PATH: {output}\n\n"
        "Generate exactly one GPT_WEB_FILE block for OUTPUT_PATH.\n"
        "Generate only input.mlir: no qasm, C, headers, Codex prompt, catalog update, shell commands, or prose.\n"
        "The file must be idiomatic final-stage vc4 dialect, semantically equivalent to the trusted qasm/launcher/harness, and must pass vc4-opt plus check-vc4.\n\n"
        "# Test specification\n\n"
        + spec.text.rstrip()
        + "\n\n"
        + format_bundle_context(files)
    )


def initial_prompt(god_prompt: str, extra_context: str, spec: Spec, files: list[BundleFile]) -> str:
    parts = [god_prompt.rstrip(), "\n\n---\n\n"]
    if extra_context:
        parts.append(extra_context.rstrip())
        parts.append("\n\n---\n\n")
    parts.append(test_prompt(spec, files).rstrip())
    parts.append("\n")
    return "".join(parts)


def current_input_path(repo: Path, name: str) -> Path:
    return repo / RUN_ROOT / name / "input.mlir"


def failure_prompt(spec: Spec, files: list[BundleFile], failure: Failure, attempt_index: int, repo: Path) -> str:
    output = f"{RUN_ROOT}/{spec.name}/input.mlir"
    current = "<missing input.mlir>"
    inp = current_input_path(repo, spec.name)
    if inp.exists():
        current = read_text(inp)
        if len(current) > 120000:
            current = current[-120000:]
    log_tail = tail_text(failure.log_path, max_lines=700, max_chars=90000)
    return (
        f"The generated input.mlir for {spec.name} failed. Emit exactly one replacement GPT_WEB_FILE block for {output}.\n"
        "Do not emit any other files or explanations. Do not touch qasm/C/H/catalog/compiler/CMake. Do not use Codex.\n"
        "Diagnose the verifier/check failure and make a surgical replacement. Never output an empty module.\n\n"
        f"Attempt index: {attempt_index}\n"
        f"Failed stage: {failure.stage}\n"
        f"Command: {failure.command or '<none>'}\n"
        f"Exit code: {failure.exit_code}\n"
        f"Timed out: {int(failure.timed_out)}\n"
        f"Message: {failure.message}\n\n"
        "Relevant log:\n"
        + log_tail.rstrip()
        + "\n\nCurrent generated input.mlir:\nBEGIN_CURRENT_GENERATED_INPUT_MLIR\n"
        + current.rstrip()
        + "\nEND_CURRENT_GENERATED_INPUT_MLIR\n\nOriginal test specification:\n"
        + spec.text.rstrip()
        + "\n\n"
        + format_bundle_context(files)
    )


def script_command(repo: Path, script: str) -> list[str]:
    p = Path(script)
    if not p.is_absolute():
        p = repo / p
    if not p.exists():
        raise DriverError(f"missing browser script: {p}")
    if os.access(p, os.X_OK):
        return [str(p)]
    return ["node", str(p)]


def marker_path(repo: Path) -> Path:
    return repo / ".vc4_auto/in_flight_prompt.json"


def clear_marker(repo: Path) -> None:
    marker_path(repo).unlink(missing_ok=True)


def run_chat(repo: Path, prompt: str, use_new_tab: bool, args: argparse.Namespace, stage_dir: Path, log_dir: Path, attempt_index: int, name: str) -> Optional[Failure]:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = log_dir / f"prompt-{attempt_index:02d}.md"
    write_text(prompt_path, prompt)
    js_response_ms = max(60_000, (args.chat_timeout_sec - DEFAULT_RESPONSE_BUFFER_SEC) * 1000)
    script = args.new_tab if use_new_tab else args.current_tab
    cmd = script_command(repo, script) + [
        "--prompt-file", str(prompt_path.resolve()),
        "--out", str(stage_dir.resolve()),
        "--repo-root", str(repo.resolve()),
        "--connect-timeout-ms", str(args.browser_internal_timeout_ms),
        "--page-timeout-ms", str(args.browser_internal_timeout_ms),
        "--prompt-timeout-ms", str(args.browser_internal_timeout_ms),
        "--response-timeout-ms", str(js_response_ms),
    ]
    print(f"[vc4-input-mlir-auto] {name}: attempt={attempt_index} browser_mode={'new' if use_new_tab else 'current'} prompt={prompt_path}")
    failure = run_logged(repo, "chat", cmd, log_dir / "chat.log", args.chat_timeout_sec)
    if failure:
        debug = stage_dir / ".gpt-web-run/debug.log"
        if debug.exists():
            print_log_tail(debug, "JS debug", 60)
    return failure


def staged_files(stage_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(stage_dir.rglob("*")):
        if p.is_dir():
            continue
        r = p.relative_to(stage_dir)
        if r.parts and r.parts[0] == ".gpt-web-run":
            continue
        if p.is_symlink() or any(part in ("..", ".git", "") for part in r.parts):
            raise DriverError(f"unsafe staged path: {r}")
        out.append(p)
    return out


def validate_input_text(text: str) -> None:
    if not text.strip():
        raise DriverError("generated input.mlir is empty")
    if re.fullmatch(r"(?s)(//[^\n]*\n|\s)*module\s*\{\s*\}", text.strip()):
        raise DriverError("generated input.mlir is the known-bad empty module shape")
    if "// RUN: vc4-opt %s" not in text:
        raise DriverError("missing required // RUN: vc4-opt %s line")
    missing = [flag for flag in VC4_OPT_FLAGS[:-2] if flag not in text]
    if missing:
        raise DriverError("RUN line missing verifier flag(s): " + ", ".join(missing))
    if "vc4.module" not in text or "vc4.func" not in text:
        raise DriverError("input.mlir must use vc4.module and vc4.func")


def copy_staged_input(repo: Path, stage_dir: Path, name: str, log_dir: Path) -> Optional[Failure]:
    expected = f"{RUN_ROOT}/{name}/input.mlir"
    try:
        files = staged_files(stage_dir)
        rels = [str(p.relative_to(stage_dir)) for p in files]
        if rels != [expected]:
            answer = tail_text(stage_dir / ".gpt-web-run/answer.md", max_lines=500, max_chars=70000)
            raise DriverError("expected exactly one staged file at " + expected + ", got:\n" + "\n".join(rels) + "\n\nanswer tail:\n" + answer)
        text = read_text(files[0])
        if not text.endswith("\n"):
            text += "\n"
        validate_input_text(text)
        write_text(repo / expected, text)
        return None
    except Exception as exc:
        log = log_dir / "chat_output_validation.log"
        write_text(log, str(exc).rstrip() + "\n")
        print_log_tail(log, "chat-output-validation", 80)
        return Failure(stage="chat-output-validation", message=str(exc), log_path=log)


def verify(repo: Path, name: str, log_dir: Path, timeout_sec: int) -> Optional[Failure]:
    inp = current_input_path(repo, name)
    failure = run_logged(
        repo,
        "vc4-opt",
        ["compiler/build/bin/vc4-opt", str(inp.relative_to(repo)), *VC4_OPT_FLAGS],
        log_dir / "vc4_opt.log",
        timeout_sec,
    )
    if failure:
        return failure
    return run_logged(repo, "check-vc4", ["cmake", "--build", "compiler/build", "--target", "check-vc4"], log_dir / "check_vc4.log", timeout_sec)


def mark_passed(repo: Path, spec: Spec, attempts_used: int, all_specs: list[Spec], no_commit: bool) -> None:
    input_rel = f"{RUN_ROOT}/{spec.name}/input.mlir"
    passed, _ = state_paths(repo, spec.name)
    write_text(
        passed,
        json.dumps({
            "name": spec.name,
            "status": "passed",
            "completed_utc": utc_now(),
            "attempts_used": attempts_used,
            "input_path": input_rel,
        }, indent=2) + "\n",
    )
    git_commit(repo, f"passed input-mlir {spec.name}", [input_rel, relpath(repo, passed)], no_commit)
    write_status(repo, all_specs)


def mark_incomplete(repo: Path, spec: Spec, attempts_used: int, failure: Failure, original_text: Optional[str], final_prompt: str, all_specs: list[Spec], no_commit: bool) -> None:
    archive = repo / AUTO_ROOT / "incomplete" / spec.name
    archive.mkdir(parents=True, exist_ok=True)
    inp = current_input_path(repo, spec.name)
    if inp.exists():
        shutil.copy2(inp, archive / "last_generated_input.mlir")
    write_text(archive / "final_failure_prompt.md", final_prompt)
    if original_text is None:
        inp.unlink(missing_ok=True)
    else:
        write_text(inp, original_text)
    _, incomplete = state_paths(repo, spec.name)
    write_text(
        incomplete,
        json.dumps({
            "name": spec.name,
            "status": "incomplete",
            "completed_utc": utc_now(),
            "attempts_used": attempts_used,
            "failed_stage": failure.stage,
            "message": failure.message,
            "command": failure.command,
            "exit_code": failure.exit_code,
            "timed_out": failure.timed_out,
            "failure_log_archive": f"{AUTO_ROOT}/logs/{spec.name}",
            "last_generated_input_archive": f"{AUTO_ROOT}/incomplete/{spec.name}/last_generated_input.mlir",
        }, indent=2) + "\n",
    )
    git_commit(repo, f"incomplete input-mlir {spec.name}", [f"{RUN_ROOT}/{spec.name}/input.mlir", relpath(repo, incomplete)], no_commit)
    write_status(repo, all_specs)


def run_one(repo: Path, spec: Spec, all_specs: list[Spec], args: argparse.Namespace, have_context: bool, god_prompt: str, extra_context: str) -> bool:
    clear_marker(repo)
    inp = current_input_path(repo, spec.name)
    original_text = read_text(inp) if inp.exists() else None
    files = collect_bundle_files(repo, spec.name, args.max_file_chars)
    prompt = test_prompt(spec, files) if have_context else initial_prompt(god_prompt, extra_context, spec, files)
    use_new_tab = not have_context
    attempt_index = 0
    fix_count = 0
    chat_infra_failures = 0

    while True:
        stage_dir = repo / AUTO_ROOT / "staging" / spec.name / f"attempt-{attempt_index:02d}"
        log_dir = repo / AUTO_ROOT / "logs" / spec.name / f"attempt-{attempt_index:02d}"
        failure = run_chat(repo, prompt, use_new_tab, args, stage_dir, log_dir, attempt_index, spec.name)
        if failure and failure.stage == "chat":
            chat_infra_failures += 1
            if chat_infra_failures <= args.chat_infra_retries:
                print(f"[vc4-input-mlir-auto] chat infra retry {chat_infra_failures}/{args.chat_infra_retries} for {spec.name}")
                attempt_index += 1
                use_new_tab = False
                continue
            if original_text is None:
                inp.unlink(missing_ok=True)
            else:
                write_text(inp, original_text)
            raise DriverError(f"ChatGPT automation failed repeatedly for {spec.name}; aborting without marking incomplete")
        chat_infra_failures = 0

        if not failure:
            failure = copy_staged_input(repo, stage_dir, spec.name, log_dir)
        if not failure:
            failure = verify(repo, spec.name, log_dir, args.step_timeout_sec)
        if not failure:
            print(f"[vc4-input-mlir-auto] PASS {spec.name}")
            clear_marker(repo)
            mark_passed(repo, spec, attempt_index + 1, all_specs, args.no_commit)
            return True

        print(f"[vc4-input-mlir-auto] FAIL {spec.name}: stage={failure.stage} fixes={fix_count}/{args.max_fixes}")
        clear_marker(repo)
        if fix_count >= args.max_fixes:
            final = failure_prompt(spec, files, failure, attempt_index, repo)
            print(f"[vc4-input-mlir-auto] INCOMPLETE {spec.name}")
            mark_incomplete(repo, spec, attempt_index + 1, failure, original_text, final, all_specs, args.no_commit)
            return True

        fix_count += 1
        attempt_index += 1
        use_new_tab = False
        files = collect_bundle_files(repo, spec.name, args.max_file_chars)
        prompt = failure_prompt(spec, files, failure, attempt_index, repo)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="repo root; default: current directory")
    ap.add_argument("--manifest", default="vc4_input_mlir_specs/manifest.tsv", help="TEST_NAME<TAB>SPEC_PATH manifest")
    ap.add_argument("--new", action="store_true", help="open a fresh ChatGPT tab and send god prompt only for the first runnable test")
    ap.add_argument("--god-prompt", default="vc4_input_mlir_specs/god_prompt.md", help="god prompt used only with --new")
    ap.add_argument("--extra-context-file", action="append", default=[], help="dialect/source/example context file; repeatable; used only with --new")
    ap.add_argument("--current-tab", default="pro_scripts/current_tab.js", help="existing ChatGPT conversation wrapper")
    ap.add_argument("--new-tab", default="pro_scripts/new_tab.js", help="new ChatGPT tab wrapper")
    ap.add_argument("--max-fixes", type=int, default=3, help="number of fix retries after the initial attempt per test")
    ap.add_argument("--chat-timeout-sec", type=int, default=DEFAULT_CHAT_TIMEOUT_SEC)
    ap.add_argument("--step-timeout-sec", type=int, default=DEFAULT_STEP_TIMEOUT_SEC)
    ap.add_argument("--browser-internal-timeout-ms", type=int, default=DEFAULT_BROWSER_TIMEOUT_MS)
    ap.add_argument("--chat-infra-retries", type=int, default=3)
    ap.add_argument("--max-file-chars", type=int, default=DEFAULT_MAX_FILE_CHARS)
    ap.add_argument("--only", default="", help="comma-separated subset of tests from the manifest")
    ap.add_argument("--allow-dirty", action="store_true", help="allow starting with dirty repo state")
    ap.add_argument("--no-commit", action="store_true", help="do not create pass/incomplete/reset git commits")
    ap.add_argument("--reset-test", action="append", default=[], help="delete passed/incomplete state marker for a test; repeatable")
    ap.add_argument("--continue-after-reset", action="store_true")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    ensure_git_repo(repo)
    os.chdir(repo)
    ensure_git_exclude(repo)
    ensure_clean_repo(repo, args.allow_dirty)

    if args.reset_test:
        reset_state(repo, args.reset_test, args.no_commit)
        if not args.continue_after_reset:
            return 0

    specs = parse_manifest(repo, (repo / args.manifest).resolve())
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if only:
        known = {s.name for s in specs}
        missing = sorted(only - known)
        if missing:
            raise DriverError("--only requested names not in manifest: " + ", ".join(missing))
        specs = [s for s in specs if s.name in only]
    write_status(repo, specs)

    god_prompt = ""
    extra_context = ""
    if args.new:
        god_path = repo / args.god_prompt
        if not god_path.exists():
            raise DriverError(f"missing god prompt: {god_path}")
        god_prompt = read_text(god_path)
        extra_context = load_extra_context(repo, args.extra_context_file, args.max_file_chars)
        print(f"[vc4-input-mlir-auto] --new: first runnable test will receive god prompt {god_path}")
    else:
        print("[vc4-input-mlir-auto] current-tab mode: no god prompt or extra context will be sent; pass --new to bootstrap a fresh chat")

    have_context = not args.new
    ran_any = False
    for spec in specs:
        status = state_status(repo, spec.name)
        if status:
            print(f"[vc4-input-mlir-auto] skip {spec.name}: already {status}")
            continue
        ran_any = True
        have_context = run_one(repo, spec, specs, args, have_context, god_prompt, extra_context)

    if not ran_any:
        print("[vc4-input-mlir-auto] nothing to do")
    write_status(repo, specs)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DriverError as exc:
        print(f"[vc4-input-mlir-auto] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
