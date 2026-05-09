#!/usr/bin/env python3
"""Deterministic orchestrator for VC4 codegen milestones.

Stage 2 installs the core controller.  It can already run status/preflight/gates
for the Stage 1 worklist.  GPT prompt rendering is intentionally deferred to
Stage 3; when prompt rendering is installed, this same orchestrator will use it
to drive GPT Pro one slice at a time through gpt_web_driver.js.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from vc4_codegen_failure_classifier import classify_failure
    from vc4_codegen_gate_runner import CommandResult, GateRunner
    from vc4_codegen_patch_gate import guard_worktree, validate_and_apply
    from vc4_codegen_state import (
        DriverError,
        MilestoneConfig,
        StateStore,
        ensure_auto_excluded,
        ensure_clean_repo,
        find_repo_root,
        git_changed_paths,
        git_head,
        normalize_relpath,
        relpath,
        restore_paths,
        stage_and_commit,
        tail_file,
        write_failure_packet,
        write_json_file,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_failure_classifier import classify_failure  # type: ignore
    from vc4_codegen_gate_runner import CommandResult, GateRunner  # type: ignore
    from vc4_codegen_patch_gate import guard_worktree, validate_and_apply  # type: ignore
    from vc4_codegen_state import (  # type: ignore
        DriverError,
        MilestoneConfig,
        StateStore,
        ensure_auto_excluded,
        ensure_clean_repo,
        find_repo_root,
        git_changed_paths,
        git_head,
        normalize_relpath,
        relpath,
        restore_paths,
        stage_and_commit,
        tail_file,
        write_failure_packet,
        write_json_file,
    )


DEFAULT_CHAT_TIMEOUT_SEC = 45 * 60
DEFAULT_CODEX_TIMEOUT_SEC = 20 * 60
DEFAULT_GATE_TIMEOUT_SEC = 30 * 60
DEFAULT_BROWSER_INTERNAL_TIMEOUT_MS = 5 * 60 * 1000
DEFAULT_ARTIFACT_DOWNLOAD_TIMEOUT_MS = 3 * 60 * 1000


# ---------------------------------------------------------------------------
# Terminal display
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[vc4-m1] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[vc4-m1] WARN: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Prompt/GPT/Codex execution helpers
# ---------------------------------------------------------------------------


def run_subprocess(
    *,
    repo: Path,
    cmd: Sequence[str],
    log_path: Path,
    cwd: Path | None = None,
    timeout_sec: int,
    verbose: bool = False,
) -> CommandResult:
    runner = GateRunner(MilestoneConfig.load(repo), verbose=verbose, timeout_sec=timeout_sec)
    return runner.run_command(gate="subprocess", cmd=list(cmd), log_dir=log_path.parent, cwd=cwd or repo, timeout_sec=timeout_sec)


def prompt_renderer_path(repo: Path) -> Path:
    return repo / "pro_scripts/vc4_codegen_prompt_render.py"


def render_prompt(
    *,
    repo: Path,
    slice_id: str,
    attempt: int,
    mode: str,
    out_path: Path,
    failure_packet: Path | None = None,
    verbose: bool = False,
    worklist_path: Path | None = None,
    context_profiles_path: Path | None = None,
) -> None:
    renderer = prompt_renderer_path(repo)
    if not renderer.exists():
        raise DriverError(
            "Stage 3 prompt renderer is not installed yet: "
            f"{relpath(repo, renderer)}. Run only status/preflight/gates until Stage 3 is installed."
        )
    cmd = [sys.executable, str(renderer), "--repo", str(repo), "--slice", slice_id, "--attempt", str(attempt), "--mode", mode, "--out", str(out_path)]
    if worklist_path is not None:
        cmd += ["--worklist", str(worklist_path)]
    if context_profiles_path is not None:
        cmd += ["--context-profiles", str(context_profiles_path)]
    if failure_packet:
        cmd += ["--failure-packet", str(failure_packet)]
    log_path = out_path.with_suffix(out_path.suffix + ".render.log")
    result = run_subprocess(repo=repo, cmd=cmd, log_path=log_path, timeout_sec=300, verbose=verbose)
    if not result.ok:
        raise DriverError(f"prompt rendering failed; see {relpath(repo, result.log_path)}")
    if not out_path.exists():
        raise DriverError(f"prompt renderer succeeded but did not create {relpath(repo, out_path)}")


def gpt_driver_command(
    repo: Path,
    *,
    mode: str,
    prompt_path: Path,
    out_dir: Path,
    response_timeout_sec: int,
    browser_timeout_ms: int,
    artifact_download_timeout_ms: int = 180000,
) -> list[str]:
    driver = repo / "pro_scripts/gpt_web_driver.js"
    if not driver.exists():
        raise DriverError(f"missing GPT web driver: {relpath(repo, driver)}")
    # gpt_web_driver.js is a module in the supplied workflow.  This small node
    # wrapper calls its exported run(mode, argv) without modifying the driver.
    js = (
        "const path=require('path');"
        "const mod=require(path.resolve(process.argv[1]));"
        "const mode=process.argv[2];"
        "const argv=['node','gpt_web_driver.js',...process.argv.slice(3)];"
        "Promise.resolve(mod.run(mode, argv)).catch(err=>{console.error(err && err.stack || err); process.exit(1);});"
    )
    return [
        "node",
        "-e",
        js,
        str(driver),
        mode,
        "--prompt-file",
        str(prompt_path),
        "--out",
        str(out_dir),
        "--repo-root",
        str(repo),
        "--connect-timeout-ms",
        str(browser_timeout_ms),
        "--page-timeout-ms",
        str(browser_timeout_ms),
        "--prompt-timeout-ms",
        str(browser_timeout_ms),
        "--response-timeout-ms",
        str(max(60, response_timeout_sec - 60) * 1000),
        "--artifact-download-timeout-ms",
        str(artifact_download_timeout_ms),
    ]


def invoke_gpt(
    *,
    repo: Path,
    mode: str,
    prompt_path: Path,
    staging_dir: Path,
    log_dir: Path,
    chat_timeout_sec: int,
    browser_internal_timeout_ms: int,
    artifact_download_timeout_ms: int,
    verbose: bool,
) -> CommandResult:
    cmd = gpt_driver_command(
        repo,
        mode=mode,
        prompt_path=prompt_path,
        out_dir=staging_dir,
        response_timeout_sec=chat_timeout_sec,
        browser_timeout_ms=browser_internal_timeout_ms,
        artifact_download_timeout_ms=artifact_download_timeout_ms,
    )
    runner = GateRunner(MilestoneConfig.load(repo), verbose=verbose, timeout_sec=chat_timeout_sec)
    return runner.run_command(gate="chat:gpt-pro", cmd=cmd, log_dir=log_dir, timeout_sec=chat_timeout_sec)


def render_codex_prompt(
    *,
    repo: Path,
    slice_entry: Mapping[str, Any],
    failed_result: CommandResult,
    route: Mapping[str, Any],
    out_path: Path,
) -> None:
    allowed = "\n".join(f"- {p}" for p in slice_entry.get("allowed_paths", []))
    forbidden = "\n".join(f"- {p}" for p in slice_entry.get("forbidden_paths", []))
    text = f"""You are Codex acting as a narrow mechanical patcher for VC4 codegen Milestone 1.

Slice: {slice_entry.get('id')} — {slice_entry.get('title')}
Intent: {slice_entry.get('intent')}

Classification:
{json.dumps(dict(route), indent=2, sort_keys=True)}

Allowed paths:
{allowed}

Forbidden paths:
{forbidden}

Rules:
- Fix only the narrow mechanical issue shown below.
- Do not change qasm semantics.
- Do not change launch ABI semantics.
- Do not change tests unless the failure is purely mechanical test invocation plumbing and the path is allowlisted.
- Do not touch reference bundles, expected.json, catalog.json, pro_scripts/gpt_web_driver.js, or .vc4_auto.
- Do not refactor.
- Make the smallest local change needed.

Failed gate: {failed_result.gate}
Command: {shlex.join(failed_result.command)}
Exit code: {failed_result.exit_code}
Log path: {relpath(repo, failed_result.log_path)}

Log tail:
```
{tail_file(failed_result.log_path, max_lines=160)}
```
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def invoke_codex(
    *,
    repo: Path,
    prompt_path: Path,
    log_dir: Path,
    timeout_sec: int,
    verbose: bool,
) -> CommandResult:
    prompt_text = prompt_path.read_text(encoding="utf-8")
    cmd = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox", prompt_text]
    runner = GateRunner(MilestoneConfig.load(repo), verbose=verbose, timeout_sec=timeout_sec)
    return runner.run_command(gate="codex:mechanical", cmd=cmd, log_dir=log_dir, timeout_sec=timeout_sec)


# ---------------------------------------------------------------------------
# Slice execution
# ---------------------------------------------------------------------------


def dependencies_passed(state: StateStore, slice_entry: Mapping[str, Any]) -> tuple[bool, dict[str, str]]:
    statuses = state.dependency_status(slice_entry)
    return all(v == "passed" for v in statuses.values()), statuses


def run_slice_gates(
    *,
    config: MilestoneConfig,
    slice_entry: Mapping[str, Any],
    log_dir: Path,
    allow_dirty: bool,
    verbose: bool,
    timeout_sec: int,
    patch_preflight: bool = False,
) -> tuple[bool, list[CommandResult]]:
    results: list[CommandResult] = []
    if patch_preflight:
        preflight = run_patch_preflight_gate(
            config=config,
            slice_entry=slice_entry,
            log_dir=log_dir,
            verbose=verbose,
            timeout_sec=timeout_sec,
        )
        results.append(preflight)
        if not preflight.ok:
            write_json_file(log_dir / "gate_summary.json", [r.as_json(config.repo) for r in results])
            return False, results

    runner = GateRunner(config, verbose=verbose, timeout_sec=timeout_sec)
    source_products = runner.run_source_products_for_slice(slice_entry, log_dir=log_dir, gate="pre-gates")
    results.append(source_products)
    if not source_products.ok:
        write_json_file(log_dir / "gate_summary.json", [r.as_json(config.repo) for r in results])
        return False, results

    gates = [str(g) for g in slice_entry.get("gates", [])]
    gate_results = runner.run_gates(gates, log_dir=log_dir, allow_dirty=allow_dirty, stop_on_failure=True)
    results.extend(gate_results)

    # The typed verifier is the final deterministic source-of-truth gate for
    # every slice.  The legacy gates still run first because they provide useful
    # targeted logs and preserve existing failure routing, but a slice is not
    # complete unless its declarative verifier spec also passes.  This is what
    # prevents weak generic gates from marking slices such as m1-07/m1-08 as
    # already satisfied when their product tests are missing.
    if all(r.ok for r in results):
        verifier_result = runner.run_typed_verifier(str(slice_entry["id"]), log_dir=log_dir)
        results.append(verifier_result)

    write_json_file(log_dir / "gate_summary.json", [r.as_json(config.repo) for r in results])
    return all(r.ok for r in results), results


def first_failed(results: Sequence[CommandResult]) -> CommandResult | None:
    for r in results:
        if not r.ok:
            return r
    return None



def _git_capture(repo: Path, args: Sequence[str], *, max_chars: int = 60000) -> str:
    try:
        proc = subprocess.run(["git", *args], cwd=str(repo), text=True, capture_output=True)
    except Exception as exc:
        return f"<git {' '.join(args)} failed to start: {exc}>"
    text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if len(text) > max_chars:
        half = max_chars // 2
        text = text[:half] + "\n\n[... truncated ...]\n\n" + text[-half:]
    return text


def collect_candidate_change_report(repo: Path) -> dict[str, Any]:
    """Capture candidate diff/excerpts before failed attempts are cleaned."""
    changed = git_changed_paths(repo, include_untracked=True)
    report: dict[str, Any] = {
        "candidate_changed_paths": changed,
        "candidate_diff_stat": _git_capture(repo, ["diff", "--stat"]),
        "candidate_diff": _git_capture(repo, ["diff"]),
        "candidate_untracked_file_excerpts": {},
    }
    excerpts: dict[str, str] = {}
    for rel in changed:
        path = repo / rel
        if not path.exists() or not path.is_file():
            continue
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel], cwd=str(repo), text=True, capture_output=True)
        if tracked.returncode == 0:
            continue
        try:
            data = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        excerpts[rel] = data[:4000]
    report["candidate_untracked_file_excerpts"] = excerpts
    return report


def cleanup_failed_attempt_changes(repo: Path, *, baseline_paths: Sequence[str], log_dir: Path | None = None) -> list[str]:
    """Restore changes introduced by a failed attempt, preserving baseline dirt."""
    baseline = set(baseline_paths)
    current = set(git_changed_paths(repo, include_untracked=True))
    to_restore = sorted(current - baseline)
    if to_restore:
        restore_paths(repo, to_restore)
        if log_dir is not None:
            write_json_file(log_dir / "failed_attempt_cleanup.json", {"restored_paths": to_restore})
        log(f"cleaned failed candidate changes: {len(to_restore)} path(s)")
    return to_restore


def run_patch_preflight_gate(
    *,
    config: MilestoneConfig,
    slice_entry: Mapping[str, Any],
    log_dir: Path,
    verbose: bool,
    timeout_sec: int,
) -> CommandResult:
    preflight = config.repo / "pro_scripts/vc4_codegen_preflight.py"
    report = log_dir / "preflight_patch_invariants_report.json"
    cmd = [
        sys.executable,
        str(preflight),
        "patch-invariants",
        "--slice",
        str(slice_entry["id"]),
        "--report",
        str(report),
    ]
    runner = GateRunner(config, verbose=verbose, timeout_sec=min(timeout_sec, 300))
    return runner.run_command(gate="preflight:patch-invariants", cmd=cmd, log_dir=log_dir, timeout_sec=min(timeout_sec, 300))


def collect_staged_output_report(staging_dir: Path, *, max_chars: int = 60000) -> dict[str, Any]:
    report: dict[str, Any] = {"staged_files": []}
    if not staging_dir.exists():
        return report
    files = []
    for path in sorted(staging_dir.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(staging_dir).as_posix())
    report["staged_files"] = files
    response = staging_dir / "response.json"
    if response.exists():
        try:
            report["staged_response_json"] = json.loads(response.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            report["staged_response_json"] = response.read_text(encoding="utf-8", errors="replace")[:max_chars]
    patch = staging_dir / "changes.patch"
    if patch.exists():
        text = patch.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            half = max_chars // 2
            text = text[:half] + "\n\n[... truncated ...]\n\n" + text[-half:]
        report["staged_changes_patch"] = text
    artifact = staging_dir / "artifact_transport.json"
    if artifact.exists():
        try:
            report["artifact_transport_json"] = json.loads(artifact.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            report["artifact_transport_json"] = artifact.read_text(encoding="utf-8", errors="replace")[:max_chars]
    bundle = staging_dir / "bundle.zip"
    if bundle.exists():
        report["bundle_zip"] = {"path": str(bundle), "bytes": bundle.stat().st_size}
        try:
            import zipfile
            with zipfile.ZipFile(bundle) as zf:
                report["bundle_zip_members"] = sorted(zf.namelist())[:200]
                if "manifest.json" in zf.namelist():
                    report["bundle_manifest_json"] = json.loads(zf.read("manifest.json").decode("utf-8", "replace"))
        except Exception as exc:
            report["bundle_zip_error"] = str(exc)
    apply_script = staging_dir / "apply_bundle.sh"
    if apply_script.exists():
        report["apply_bundle_sh"] = apply_script.read_text(encoding="utf-8", errors="replace")[:4000]
    return report

def _decode_first_json_object(text: str) -> Mapping[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def collect_typed_verifier_report(failed: CommandResult) -> dict[str, Any]:
    if not str(failed.gate).startswith("typed-verifier:"):
        return {}
    if not failed.log_path.exists():
        return {"typed_verifier": {"error": "typed verifier log is missing"}}
    text = failed.log_path.read_text(encoding="utf-8", errors="replace")
    report = _decode_first_json_object(text)
    if not isinstance(report, Mapping):
        return {"typed_verifier": {"error": "could not parse typed verifier JSON report from log"}}
    failures = report.get("failures", [])
    results = report.get("results", [])
    first_failure = failures[0] if isinstance(failures, list) and failures else None
    return {
        "typed_verifier": {
            "ok": bool(report.get("ok")),
            "slice_ids": report.get("slice_ids"),
            "first_failure": first_failure,
            "failure_count": len(failures) if isinstance(failures, list) else None,
            "result_count": len(results) if isinstance(results, list) else None,
        }
    }


def write_gate_failure_packet(
    *,
    state: StateStore,
    paths: Any,
    slice_entry: Mapping[str, Any],
    failed: CommandResult,
) -> Path:
    extra = {"gate": failed.gate, **collect_candidate_change_report(state.config.repo)}
    extra.update(collect_typed_verifier_report(failed))
    packet = write_failure_packet(
        paths.failure_packet_path,
        slice_entry=slice_entry,
        stage="gate",
        message=f"Gate failed: {failed.gate}",
        command=failed.command,
        exit_code=failed.exit_code,
        timed_out=failed.timed_out,
        log_path=failed.log_path,
        extra=extra,
    )
    packet["gate"] = failed.gate
    write_json_file(paths.failure_packet_path, packet)
    return paths.failure_packet_path


def commit_slice_if_needed(config: MilestoneConfig, slice_entry: Mapping[str, Any], *, no_commit: bool) -> bool:
    if no_commit or not config.auto_commit_on_pass():
        return False

    # Final guard before any automatic commit.  Patch application already checks
    # GPT-produced paths, and Codex edits are path-guarded after each mechanical
    # attempt, but gates can still create files as side effects.  Never let a
    # passing slice auto-commit side-effect files outside the slice allowlist,
    # and never commit forbidden paths such as reference bundles, expected.json,
    # catalog.json, gpt_web_driver.js, or .vc4_auto.
    guard = guard_worktree(
        repo=config.repo,
        config=config,
        slice_entry=slice_entry,
        restore_disallowed=False,
    )
    changed = [str(p) for p in guard.get("changed_paths", [])]
    if not changed:
        log("auto-commit: no repo changes to commit")
        return False

    if not guard.get("ok"):
        report = {
            "slice_id": slice_entry.get("id"),
            "changed_paths": changed,
            "disallowed_paths": guard.get("disallowed_paths", []),
            "forbidden_paths": guard.get("forbidden_paths", []),
        }
        raise DriverError(
            "refusing to auto-commit: worktree contains paths outside the "
            "current slice policy. Inspect the report below, restore or move "
            "the side-effect files, then rerun the slice.\n"
            + json.dumps(report, indent=2, sort_keys=True)
        )

    message = str(slice_entry.get("commit_message") or f"vc4 codegen m1: {slice_entry.get('id')}")
    committed = stage_and_commit(config.repo, paths=changed, message=message, allow_empty=False)
    if committed:
        log(f"auto-commit: committed {len(changed)} allowlisted changed path(s) with message: {message}")
    return committed


def run_gate_only_slice(
    *,
    config: MilestoneConfig,
    state: StateStore,
    slice_entry: Mapping[str, Any],
    allow_dirty: bool,
    no_commit: bool,
    verbose: bool,
    gate_timeout_sec: int,
) -> int:
    attempt = state.next_attempt_index(str(slice_entry["id"]))
    paths = state.attempt_paths(str(slice_entry["id"]), attempt)
    ok, results = run_slice_gates(
        config=config,
        slice_entry=slice_entry,
        log_dir=paths.log_dir,
        allow_dirty=allow_dirty,
        verbose=verbose,
        timeout_sec=gate_timeout_sec,
    )
    if ok:
        commit_slice_if_needed(config, slice_entry, no_commit=no_commit)
        state.mark_slice_passed(slice_id=str(slice_entry["id"]), gate_results=[r.as_json(config.repo) for r in results], repo_head=git_head(config.repo, allow_missing=True))
        state.record_attempt(slice_id=str(slice_entry["id"]), attempt=attempt, status="passed", details={"gate_only": True})
        log(f"PASS {slice_entry['id']}")
        return 0
    failed = first_failed(results)
    if failed:
        fp = write_gate_failure_packet(state=state, paths=paths, slice_entry=slice_entry, failed=failed)
        state.mark_slice_failed(slice_id=str(slice_entry["id"]), failure_packet=relpath(config.repo, fp))
    state.record_attempt(slice_id=str(slice_entry["id"]), attempt=attempt, status="failed", details={"gate_only": True})
    log(f"FAIL {slice_entry['id']}; logs in {relpath(config.repo, paths.log_dir)}")
    return 1


def run_gpt_slice(
    *,
    config: MilestoneConfig,
    state: StateStore,
    slice_entry: Mapping[str, Any],
    allow_dirty: bool,
    no_commit: bool,
    verbose: bool,
    gpt_mode: str,
    chat_timeout_sec: int,
    codex_timeout_sec: int,
    gate_timeout_sec: int,
    browser_internal_timeout_ms: int,
    artifact_download_timeout_ms: int = DEFAULT_ARTIFACT_DOWNLOAD_TIMEOUT_MS,
) -> int:
    slice_id = str(slice_entry["id"])
    max_gpt = int(slice_entry.get("max_gpt_attempts", 1) or 1)
    codex_attempts_used = 0
    codex_attempts_by_category: dict[str, int] = {}
    failure_packet: Path | None = None

    for _ in range(max_gpt):
        attempt = state.next_attempt_index(slice_id)
        paths = state.attempt_paths(slice_id, attempt)
        mode = "failure" if failure_packet else "initial"
        log(f"=== {slice_id} {paths.attempt_name} ({mode}) ===")

        render_prompt(
            repo=config.repo,
            slice_id=slice_id,
            attempt=attempt,
            mode=mode,
            out_path=paths.prompt_path,
            failure_packet=failure_packet,
            verbose=verbose,
            worklist_path=config.worklist_path,
            context_profiles_path=config.context_profiles_path,
        )
        log(f"prompt: {relpath(config.repo, paths.prompt_path)}")

        chat = invoke_gpt(
            repo=config.repo,
            mode=gpt_mode,
            prompt_path=paths.prompt_path,
            staging_dir=paths.staging_dir,
            log_dir=paths.log_dir,
            chat_timeout_sec=chat_timeout_sec,
            browser_internal_timeout_ms=browser_internal_timeout_ms,
            artifact_download_timeout_ms=artifact_download_timeout_ms,
            verbose=verbose,
        )
        if not chat.ok:
            failure_packet = paths.failure_packet_path
            write_failure_packet(
                failure_packet,
                slice_entry=slice_entry,
                stage="chat",
                message="GPT web-driver invocation failed",
                command=chat.command,
                exit_code=chat.exit_code,
                timed_out=chat.timed_out,
                log_path=chat.log_path,
            )
            state.record_attempt(slice_id=slice_id, attempt=attempt, status="chat-failed")
            log(f"chat failed; retrying same slice with failure packet {relpath(config.repo, failure_packet)}")
            continue

        attempt_baseline_paths = git_changed_paths(config.repo, include_untracked=True)

        try:
            report = validate_and_apply(
                repo=config.repo,
                config=config,
                slice_entry=slice_entry,
                staging_dir=paths.staging_dir,
                apply_patch=True,
                write_report=paths.log_dir / "patch_gate_report.json",
            )
            log("patch applied: " + ", ".join(report.get("changed_paths", [])))
        except Exception as exc:
            failure_packet = paths.failure_packet_path
            write_failure_packet(
                failure_packet,
                slice_entry=slice_entry,
                stage="patch-guard",
                message=str(exc),
                log_path=paths.log_dir / "patch_gate_report.json" if (paths.log_dir / "patch_gate_report.json").exists() else None,
                extra=collect_staged_output_report(paths.staging_dir),
            )
            state.record_attempt(slice_id=slice_id, attempt=attempt, status="patch-rejected")
            log(f"patch rejected; failure packet {relpath(config.repo, failure_packet)}")
            continue

        ok, results = run_slice_gates(
            config=config,
            slice_entry=slice_entry,
            log_dir=paths.log_dir,
            allow_dirty=allow_dirty,
            verbose=verbose,
            timeout_sec=gate_timeout_sec,
            patch_preflight=True,
        )
        if ok:
            commit_slice_if_needed(config, slice_entry, no_commit=no_commit)
            state.mark_slice_passed(slice_id=slice_id, gate_results=[r.as_json(config.repo) for r in results], repo_head=git_head(config.repo, allow_missing=True))
            state.record_attempt(slice_id=slice_id, attempt=attempt, status="passed")
            log(f"PASS {slice_id}")
            return 0

        failed = first_failed(results)
        assert failed is not None
        failure_packet = write_gate_failure_packet(state=state, paths=paths, slice_entry=slice_entry, failed=failed)
        route = classify_failure(
            slice_entry=slice_entry,
            stage="gate",
            gate=failed.gate,
            log_text=tail_file(failed.log_path, max_lines=200),
            codex_attempts_used=codex_attempts_used,
            codex_attempts_by_category=codex_attempts_by_category,
        )
        write_json_file(paths.log_dir / "failure_route.json", route)
        log(f"failure route: {route['route']} ({route['category']})")

        if route.get("route") == "codex":
            codex_prompt = paths.log_dir / "codex_mechanical_prompt.md"
            render_codex_prompt(repo=config.repo, slice_entry=slice_entry, failed_result=failed, route=route, out_path=codex_prompt)
            codex = invoke_codex(repo=config.repo, prompt_path=codex_prompt, log_dir=paths.log_dir, timeout_sec=codex_timeout_sec, verbose=verbose)
            codex_attempts_used += 1
            category = str(route.get("category", "mechanical"))
            codex_attempts_by_category[category] = codex_attempts_by_category.get(category, 0) + 1
            if not codex.ok:
                state.record_attempt(slice_id=slice_id, attempt=attempt, status="codex-failed")
                failure_packet = paths.failure_packet_path
                write_failure_packet(
                    failure_packet,
                    slice_entry=slice_entry,
                    stage="codex",
                    message="Codex mechanical fix failed",
                    command=codex.command,
                    exit_code=codex.exit_code,
                    timed_out=codex.timed_out,
                    log_path=codex.log_path,
                )
                cleanup_failed_attempt_changes(config.repo, baseline_paths=attempt_baseline_paths, log_dir=paths.log_dir)
                continue
            guard = guard_worktree(repo=config.repo, config=config, slice_entry=slice_entry, restore_disallowed=True)
            write_json_file(paths.log_dir / "codex_path_guard.json", guard)
            if not guard.get("ok"):
                failure_packet = paths.failure_packet_path
                write_failure_packet(
                    failure_packet,
                    slice_entry=slice_entry,
                    stage="path-guard",
                    message="Codex changed paths outside the slice policy; disallowed paths were restored",
                    log_path=paths.log_dir / "codex_path_guard.json",
                )
                state.record_attempt(slice_id=slice_id, attempt=attempt, status="codex-path-guard-failed")
                cleanup_failed_attempt_changes(config.repo, baseline_paths=attempt_baseline_paths, log_dir=paths.log_dir)
                continue
            ok_after_codex, results_after_codex = run_slice_gates(
                config=config,
                slice_entry=slice_entry,
                log_dir=paths.log_dir / "after-codex",
                allow_dirty=allow_dirty,
                verbose=verbose,
                timeout_sec=gate_timeout_sec,
                patch_preflight=True,
            )
            if ok_after_codex:
                commit_slice_if_needed(config, slice_entry, no_commit=no_commit)
                state.mark_slice_passed(slice_id=slice_id, gate_results=[r.as_json(config.repo) for r in results_after_codex], repo_head=git_head(config.repo, allow_missing=True))
                state.record_attempt(slice_id=slice_id, attempt=attempt, status="passed-after-codex")
                log(f"PASS {slice_id} after Codex mechanical fix")
                return 0
            failed_after = first_failed(results_after_codex)
            if failed_after:
                failure_packet = write_gate_failure_packet(state=state, paths=paths, slice_entry=slice_entry, failed=failed_after)

        cleanup_failed_attempt_changes(config.repo, baseline_paths=attempt_baseline_paths, log_dir=paths.log_dir)
        state.record_attempt(slice_id=slice_id, attempt=attempt, status="needs-gpt-fix", details={"failure_packet": relpath(config.repo, failure_packet) if failure_packet else ""})
        log(f"attempt did not pass; next GPT attempt will receive {relpath(config.repo, failure_packet) if failure_packet else '<none>'}")

    state.mark_slice_failed(slice_id=slice_id, failure_packet=relpath(config.repo, failure_packet) if failure_packet else None)
    log(f"FAIL {slice_id}: exhausted {max_gpt} GPT attempt(s)")
    return 1



# ---------------------------------------------------------------------------
# Safe integration dry-run
# ---------------------------------------------------------------------------


def _dry_run_record(records: list[dict[str, Any]], name: str, ok: bool, **details: Any) -> None:
    rec = {"name": name, "ok": bool(ok), **details}
    records.append(rec)
    status = "OK" if ok else "FAIL"
    log(f"dry-run {status}: {name}")


def _write_stage5_valid_smoke_patch(staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "response.json").write_text(json.dumps({"slice_id": "m1-01-artifact-tool-skeleton", "summary": "stage5 valid patch-gate smoke", "changed_paths": ["compiler/test/CodeGen/VC4/Emit/stage5-dry-run-smoke.mlir"], "tests_to_run": []}, indent=2) + "\n", encoding="utf-8")
    (staging / "changes.patch").write_text(
        """diff --git a/compiler/test/CodeGen/VC4/Emit/stage5-dry-run-smoke.mlir b/compiler/test/CodeGen/VC4/Emit/stage5-dry-run-smoke.mlir
new file mode 100644
index 0000000..a9c1a15
--- /dev/null
+++ b/compiler/test/CodeGen/VC4/Emit/stage5-dry-run-smoke.mlir
@@ -0,0 +1 @@
+// stage5 dry-run patch gate smoke test
""",
        encoding="utf-8",
    )


def _write_stage5_forbidden_smoke_patch(staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "response.json").write_text(json.dumps({"slice_id": "m1-01-artifact-tool-skeleton", "summary": "stage5 forbidden patch-gate smoke", "changed_paths": ["compiler/test/CodeGen/VC4/catalog.json"], "tests_to_run": []}, indent=2) + "\n", encoding="utf-8")
    (staging / "changes.patch").write_text(
        """diff --git a/compiler/test/CodeGen/VC4/catalog.json b/compiler/test/CodeGen/VC4/catalog.json
--- a/compiler/test/CodeGen/VC4/catalog.json
+++ b/compiler/test/CodeGen/VC4/catalog.json
@@ -1 +1 @@
-{}
+{"forbidden": true}
""",
        encoding="utf-8",
    )


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Exercise the automation plumbing without contacting GPT Pro or Codex."""

    repo, config, state = load_config_and_state(args)
    state.ensure_dirs()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    root = state.root / "dry_run" / timestamp
    logs = root / "logs"
    staging = root / "staging"
    prompts = root / "prompts"
    root.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    log(f"dry-run root: {relpath(repo, root)}")
    log("dry-run will not invoke GPT Pro, Codex, git apply, or hardware candidate gates")

    if not args.skip_preflight:
        preflight_slice = config.get_slice("m1-00-preflight")
        ok, results = run_slice_gates(
            config=config,
            slice_entry=preflight_slice,
            log_dir=logs / "preflight",
            allow_dirty=args.allow_dirty,
            verbose=args.verbose,
            timeout_sec=args.gate_timeout_sec,
        )
        _dry_run_record(records, "preflight-gates", ok, results=[r.as_json(repo) for r in results])
        if not ok and not args.keep_going:
            write_json_file(root / "summary.json", {"ok": False, "records": records})
            return 1

    # Context pack generation.
    context_out = root / "m1-01.context.md"
    context_meta = root / "m1-01.context.metadata.json"
    context_cmd = [
        sys.executable,
        str(repo / "pro_scripts/vc4_codegen_context_pack.py"),
        "build",
        "--slice",
        "m1-01-artifact-tool-skeleton",
        "--out",
        str(context_out),
        "--metadata-out",
        str(context_meta),
    ]
    context_result = run_subprocess(repo=repo, cmd=context_cmd, log_path=logs / "context.log", timeout_sec=300, verbose=args.verbose)
    context_ok = context_result.ok and context_out.exists() and context_out.stat().st_size > 0 and context_meta.exists()
    _dry_run_record(
        records,
        "context-pack-m1-01",
        context_ok,
        command=context_result.command,
        log=relpath(repo, context_result.log_path),
        context=relpath(repo, context_out),
        metadata=relpath(repo, context_meta),
        bytes=context_out.stat().st_size if context_out.exists() else 0,
    )
    if not context_ok and not args.keep_going:
        write_json_file(root / "summary.json", {"ok": False, "records": records})
        return 1

    # Prompt rendering.
    prompt_out = prompts / "m1-01-attempt-01.md"
    prompt_meta = prompts / "m1-01-attempt-01.metadata.json"
    render_cmd = [
        sys.executable,
        str(repo / "pro_scripts/vc4_codegen_prompt_render.py"),
        "--slice",
        "m1-01-artifact-tool-skeleton",
        "--attempt",
        "1",
        "--mode",
        "initial",
        "--out",
        str(prompt_out),
        "--metadata-out",
        str(prompt_meta),
    ]
    render_result = run_subprocess(repo=repo, cmd=render_cmd, log_path=logs / "render_prompt.log", timeout_sec=300, verbose=args.verbose)
    prompt_text = prompt_out.read_text(encoding="utf-8") if prompt_out.exists() else ""
    required_prompt_needles = [
        "Slice: `m1-01-artifact-tool-skeleton`",
        "Allowed paths",
        "Forbidden paths",
        "response.json",
        "changes.patch",
        "Context pack",
        "BEGIN_GPTWEB_FILE",
    ]
    missing_needles = [needle for needle in required_prompt_needles if needle not in prompt_text]
    render_ok = render_result.ok and prompt_out.exists() and not missing_needles
    _dry_run_record(
        records,
        "prompt-render-m1-01",
        render_ok,
        command=render_result.command,
        log=relpath(repo, render_result.log_path),
        prompt=relpath(repo, prompt_out),
        metadata=relpath(repo, prompt_meta),
        bytes=len(prompt_text),
        missing=missing_needles,
    )
    if not render_ok and not args.keep_going:
        write_json_file(root / "summary.json", {"ok": False, "records": records})
        return 1

    # Patch gate check-only acceptance smoke.
    slice_entry = config.get_slice("m1-01-artifact-tool-skeleton")
    valid_staging = staging / "valid_patch"
    _write_stage5_valid_smoke_patch(valid_staging)
    try:
        valid_report = validate_and_apply(
            repo=repo,
            config=config,
            slice_entry=slice_entry,
            staging_dir=valid_staging,
            apply_patch=False,
            write_report=logs / "valid_patch_gate_report.json",
        )
        valid_ok = bool(valid_report.get("ok")) and not bool(valid_report.get("applied"))
    except Exception as exc:  # pragma: no cover - surfaced in dry-run output
        valid_report = {"error": str(exc)}
        valid_ok = False
    _dry_run_record(records, "patch-gate-valid-check-only", valid_ok, report=valid_report)
    if not valid_ok and not args.keep_going:
        write_json_file(root / "summary.json", {"ok": False, "records": records})
        return 1

    # Patch gate forbidden-path rejection smoke. This must fail before git apply.
    forbidden_staging = staging / "forbidden_patch"
    _write_stage5_forbidden_smoke_patch(forbidden_staging)
    try:
        forbidden_report = validate_and_apply(
            repo=repo,
            config=config,
            slice_entry=slice_entry,
            staging_dir=forbidden_staging,
            apply_patch=False,
            write_report=logs / "forbidden_patch_gate_report.json",
        )
        forbidden_ok = False
        forbidden_details: dict[str, Any] = {"unexpected_report": forbidden_report}
    except Exception as exc:
        msg = str(exc)
        forbidden_ok = "catalog.json" in msg or "forbidden" in msg.lower() or "not allowed" in msg.lower()
        forbidden_details = {"expected_error": msg}
    _dry_run_record(records, "patch-gate-forbidden-reject", forbidden_ok, **forbidden_details)
    if not forbidden_ok and not args.keep_going:
        write_json_file(root / "summary.json", {"ok": False, "records": records})
        return 1

    # Failure classifier smoke checks.
    route_compile = classify_failure(
        slice_entry=slice_entry,
        stage="gate",
        gate="build:vc4-codegen",
        log_text="compiler error: missing include",
        codex_attempts_used=0,
    )
    route_qasm = classify_failure(
        slice_entry=config.get_slice("m1-03-minimal-thrend-qasm"),
        stage="gate",
        gate="tool:vc4asm-candidate:minimal_thrend",
        log_text="vc4asm rejected qasm syntax",
        codex_attempts_used=0,
    )
    classifier_ok = route_compile.get("route") == "codex" and route_qasm.get("route") in {"gpt", "gpt_pro"}
    _dry_run_record(records, "failure-classifier-routes", classifier_ok, compile_route=route_compile, qasm_route=route_qasm)
    if not classifier_ok and not args.keep_going:
        write_json_file(root / "summary.json", {"ok": False, "records": records})
        return 1

    # Typed verifier contract audit.  This does not run slice tools/hardware; it
    # proves the central verifier and per-slice specs are syntactically usable.
    verifier_runner = GateRunner(config, verbose=args.verbose, timeout_sec=args.gate_timeout_sec)
    verifier_audit = verifier_runner.run_typed_audit(log_dir=logs)
    _dry_run_record(records, "typed-verifier-audit-contract", verifier_audit.ok, command=verifier_audit.command, log=relpath(repo, verifier_audit.log_path))
    if not verifier_audit.ok and not args.keep_going:
        write_json_file(root / "summary.json", {"ok": False, "records": records})
        return 1

    # Confirm candidate support scripts are present but do not run candidate generation yet.
    support_paths = [
        repo / "compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh",
        repo / "compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py",
        repo / "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/run.sh",
    ]
    support_ok = all(p.exists() for p in support_paths)
    _dry_run_record(records, "candidate-support-files-present", support_ok, paths=[relpath(repo, p) for p in support_paths])

    ok_all = all(bool(r.get("ok")) for r in records)
    summary = {
        "ok": ok_all,
        "repo": str(repo),
        "dry_run_root": relpath(repo, root),
        "next_runnable_slice": state.next_pending_slice().get("id") if state.next_pending_slice() else None,
        "records": records,
        "note": "No GPT Pro, Codex, git apply, or candidate hardware gates were invoked by this dry run.",
    }
    write_json_file(root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    log(f"dry-run summary: {relpath(repo, root / 'summary.json')}")
    return 0 if ok_all else 1

# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def load_config_and_state(args: argparse.Namespace) -> tuple[Path, MilestoneConfig, StateStore]:
    # These attributes may come from either the parent parser or the selected
    # subparser.  Subparser definitions use argparse.SUPPRESS defaults so a
    # value supplied before the subcommand is not overwritten when the same
    # switch is omitted after the subcommand.
    repo_arg = getattr(args, "repo", ".")
    worklist_arg = getattr(args, "worklist", "pro_scripts/vc4_codegen_m1_worklist.json")
    context_profiles_arg = getattr(
        args,
        "context_profiles",
        "pro_scripts/vc4_codegen_m1_context_profiles.json",
    )
    spec_arg = getattr(args, "spec", "")

    repo = find_repo_root(repo_arg)
    ensure_auto_excluded(repo)
    config = MilestoneConfig.load(
        repo,
        worklist_path=worklist_arg,
        context_profiles_path=context_profiles_arg,
    )

    # M2 worklists already carry defaults.verification_spec, but accepting
    # --spec lets callers override it and keeps the CLI symmetric with the
    # typed verifier entrypoint.
    if spec_arg:
        defaults = config.worklist.setdefault("defaults", {})
        if not isinstance(defaults, dict):
            raise DriverError("worklist.defaults must be an object before --spec can be applied")
        defaults["verification_spec"] = str(spec_arg)

    state = StateStore(config)
    state.ensure_dirs()
    return repo, config, state


def cmd_status(args: argparse.Namespace) -> int:
    repo, config, state = load_config_and_state(args)
    data = state.load()
    print(f"Milestone: {config.milestone}")
    print(f"Repo: {repo}")
    print(f"State: {relpath(repo, state.path)}")
    print()
    print(f"{'STATUS':<10} {'SLICE':<36} TITLE")
    print("-" * 96)
    for s in config.slices:
        sid = str(s["id"])
        status = state.slice_status(sid)
        deps = state.dependency_status(s)
        blocked = [dep for dep, dep_status in deps.items() if dep_status != "passed"]
        shown_status = status if not blocked or status == "passed" else "blocked"
        print(f"{shown_status:<10} {sid:<36} {s.get('title', '')}")
        if args.verbose and blocked:
            print(f"{'':<10} {'':<36} blocked by: {', '.join(blocked)}")
    next_slice = state.next_pending_slice()
    print()
    if next_slice:
        print(f"Next runnable slice: {next_slice['id']} — {next_slice['title']}")
    else:
        print("Next runnable slice: <none>")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    repo, config, state = load_config_and_state(args)
    slice_entry = config.get_slice("m1-00-preflight")
    log_dir = state.root / "logs" / "m1-00-preflight" / "manual-preflight"
    ok, _results = run_slice_gates(
        config=config,
        slice_entry=slice_entry,
        log_dir=log_dir,
        allow_dirty=args.allow_dirty,
        verbose=args.verbose,
        timeout_sec=args.gate_timeout_sec,
    )
    log(f"preflight logs: {relpath(repo, log_dir)}")
    return 0 if ok else 1


def cmd_gates(args: argparse.Namespace) -> int:
    repo, config, state = load_config_and_state(args)
    slice_entry = config.get_slice(args.slice)
    log_dir = Path(args.log_dir).resolve() if args.log_dir else state.root / "logs" / str(slice_entry["id"]) / "manual-gates"
    runner = GateRunner(config, verbose=args.verbose, timeout_sec=args.gate_timeout_sec)
    gates = [str(g) for g in slice_entry.get("gates", [])]
    if args.only_gate:
        selected = set(args.only_gate)
        typed_allowed = {f"typed-verifier:{slice_entry['id']}", "typed-verifier:audit-contract"}
        missing = selected - set(gates) - typed_allowed
        if missing:
            raise DriverError(f"requested --only-gate entries are not in slice: {sorted(missing)}")
        gates = [g for g in gates if g in selected]
        if f"typed-verifier:{slice_entry['id']}" in selected:
            gates.append(f"typed-verifier:{slice_entry['id']}")
        if "typed-verifier:audit-contract" in selected:
            gates.append("typed-verifier:audit-contract")
    if args.only_gate:
        results = runner.run_gates(gates, log_dir=log_dir, allow_dirty=args.allow_dirty, stop_on_failure=not args.keep_going)
        if all(r.ok for r in results) and any(g.startswith("typed-verifier:") for g in args.only_gate):
            pass
    else:
        ok, results = run_slice_gates(
            config=config,
            slice_entry=slice_entry,
            log_dir=log_dir,
            allow_dirty=args.allow_dirty,
            verbose=args.verbose,
            timeout_sec=args.gate_timeout_sec,
            patch_preflight=False,
        )
    print(json.dumps([r.as_json(repo) for r in results], indent=2, sort_keys=True))
    return 0 if all(r.ok for r in results) else 1


def cmd_context(args: argparse.Namespace) -> int:
    repo, config, state = load_config_and_state(args)
    context_script = repo / "pro_scripts/vc4_codegen_context_pack.py"
    if not context_script.exists():
        raise DriverError(f"Stage 3 context packer is not installed: {relpath(repo, context_script)}")
    out = Path(args.out).resolve() if args.out else state.root / "prompts" / args.slice / "manual-context.md"
    meta = Path(args.metadata_out).resolve() if args.metadata_out else out.with_suffix(out.suffix + ".metadata.json")
    cmd = [
        sys.executable,
        str(context_script),
        "build",
        "--slice",
        args.slice,
        "--mode",
        args.mode,
        "--out",
        str(out),
        "--metadata-out",
        str(meta),
    ]
    if args.failure_packet:
        cmd += ["--failure-packet", args.failure_packet]
    if args.max_chars:
        cmd += ["--max-chars", str(args.max_chars)]
    if args.allow_large_context:
        cmd += ["--allow-large-context"]
    log_path = out.with_suffix(out.suffix + ".context.log")
    result = run_subprocess(repo=repo, cmd=cmd, log_path=log_path, timeout_sec=300, verbose=args.verbose)
    if not result.ok:
        raise DriverError(f"context generation failed; see {relpath(repo, result.log_path)}")
    log(f"context: {relpath(repo, out)}")
    log(f"metadata: {relpath(repo, meta)}")
    return 0


def cmd_render_prompt(args: argparse.Namespace) -> int:
    repo, config, state = load_config_and_state(args)
    attempt = args.attempt if args.attempt else state.next_attempt_index(args.slice)
    out = Path(args.out).resolve() if args.out else state.root / "prompts" / args.slice / f"manual-attempt-{attempt:02d}.md"
    failure_packet = Path(args.failure_packet) if args.failure_packet else None
    render_prompt(
        repo=repo,
        slice_id=args.slice,
        attempt=attempt,
        mode=args.mode,
        out_path=out,
        failure_packet=failure_packet,
        verbose=args.verbose,
        worklist_path=config.worklist_path,
        context_profiles_path=config.context_profiles_path,
    )
    log(f"prompt: {relpath(repo, out)}")
    return 0



def probe_already_satisfied(
    *,
    config: MilestoneConfig,
    state: StateStore,
    slice_entry: Mapping[str, Any],
    allow_dirty: bool,
    no_commit: bool,
    verbose: bool,
    gate_timeout_sec: int,
) -> bool:
    """Run declared gates once before GPT to detect already-landed slices."""
    if bool(slice_entry.get("hardware_required", False)):
        return False
    slice_id = str(slice_entry["id"])
    log_dir = state.root / "logs" / slice_id / "already-satisfied-probe"
    ok, results = run_slice_gates(
        config=config,
        slice_entry=slice_entry,
        log_dir=log_dir,
        allow_dirty=allow_dirty,
        verbose=verbose,
        timeout_sec=gate_timeout_sec,
        patch_preflight=False,
    )
    if not ok:
        if verbose:
            log(f"already-satisfied probe did not pass for {slice_id}; continuing to GPT workflow")
        return False
    commit_slice_if_needed(config, slice_entry, no_commit=no_commit)
    state.mark_slice_passed(slice_id=slice_id, gate_results=[r.as_json(config.repo) for r in results], repo_head=git_head(config.repo, allow_missing=True))
    state.record_attempt(slice_id=slice_id, attempt=state.next_attempt_index(slice_id), status="already-satisfied", details={"probe_log_dir": relpath(config.repo, log_dir)})
    log(f"PASS {slice_id}: declared gates already pass; marked slice as already satisfied without GPT")
    return True

def cmd_run(args: argparse.Namespace) -> int:
    repo, config, state = load_config_and_state(args)
    if args.next:
        slice_entry = state.next_pending_slice()
        if not slice_entry:
            log("no pending runnable slice")
            return 0
    else:
        if not args.slice:
            raise DriverError("pass --slice <id> or --next")
        slice_entry = config.get_slice(args.slice)

    slice_id = str(slice_entry["id"])
    ok_deps, dep_status = dependencies_passed(state, slice_entry)
    if not ok_deps and not args.ignore_deps:
        raise DriverError(f"slice {slice_id} has unmet dependencies: {dep_status}. Pass --ignore-deps only for debugging.")

    ensure_clean_repo(repo, allow_dirty=args.allow_dirty)

    max_gpt = int(slice_entry.get("max_gpt_attempts", 0) or 0)
    if max_gpt > 0 and not args.skip_already_landed_probe:
        if probe_already_satisfied(
            config=config,
            state=state,
            slice_entry=slice_entry,
            allow_dirty=args.allow_dirty,
            no_commit=args.no_commit,
            verbose=args.verbose,
            gate_timeout_sec=args.gate_timeout_sec,
        ):
            return 0
    if max_gpt <= 0:
        return run_gate_only_slice(
            config=config,
            state=state,
            slice_entry=slice_entry,
            allow_dirty=args.allow_dirty,
            no_commit=args.no_commit,
            verbose=args.verbose,
            gate_timeout_sec=args.gate_timeout_sec,
        )
    return run_gpt_slice(
        config=config,
        state=state,
        slice_entry=slice_entry,
        allow_dirty=args.allow_dirty,
        no_commit=args.no_commit,
        verbose=args.verbose,
        gpt_mode=args.gpt_mode,
        chat_timeout_sec=args.chat_timeout_sec,
        codex_timeout_sec=args.codex_timeout_sec,
        gate_timeout_sec=args.gate_timeout_sec,
        browser_internal_timeout_ms=args.browser_internal_timeout_ms,
        artifact_download_timeout_ms=args.artifact_download_timeout_ms,
    )


def cmd_reset_slice(args: argparse.Namespace) -> int:
    repo, _config, state = load_config_and_state(args)
    state.reset_slice(args.slice)
    if args.purge_files:
        for child in ["prompts", "staging", "logs", "failure_packets"]:
            path = state.root / child / args.slice
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        log(f"purged stored attempts/logs for {args.slice}")
    log(f"reset state for {args.slice}")
    return 0


def cmd_list_gates(args: argparse.Namespace) -> int:
    repo, config, _state = load_config_and_state(args)
    for s in config.slices:
        print(f"{s['id']}: {s['title']}")
        for gate in s.get("gates", []):
            print(f"  - {gate}")
    return 0



def _add_common_config_args(parser: argparse.ArgumentParser, *, hidden: bool = False) -> None:
    """Allow milestone config options either before or after the subcommand.

    argparse normally requires parent-parser options to appear before the
    subcommand.  The M2 workflow commonly invokes:

      vc4_codegen_m1_autorun.py run --repo ... --worklist ... --spec ... --slice ...

    so every subparser also accepts the same configuration switches.  Suppressed
    defaults preserve values supplied before the subcommand.
    """
    help_text = argparse.SUPPRESS if hidden else None
    parser.add_argument(
        "--repo",
        default=argparse.SUPPRESS,
        help=help_text or "repo root, default: current directory",
    )
    parser.add_argument(
        "--worklist",
        default=argparse.SUPPRESS,
        help=help_text or "milestone worklist JSON",
    )
    parser.add_argument(
        "--context-profiles",
        default=argparse.SUPPRESS,
        help=help_text or "milestone context profiles JSON",
    )
    parser.add_argument(
        "--spec",
        default=argparse.SUPPRESS,
        help=help_text or "typed verifier spec JSON override",
    )



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="repo root, default: current directory")
    parser.add_argument("--worklist", default="pro_scripts/vc4_codegen_m1_worklist.json")
    parser.add_argument("--context-profiles", default="pro_scripts/vc4_codegen_m1_context_profiles.json")
    parser.add_argument("--spec", default="", help="typed verifier spec JSON override")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_status = sub.add_parser("status")
    _add_common_config_args(p_status, hidden=True)
    p_status.add_argument("--verbose", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_preflight = sub.add_parser("preflight", help="run m1-00 gates without marking state")
    _add_common_config_args(p_preflight, hidden=True)
    p_preflight.add_argument("--allow-dirty", action="store_true")
    p_preflight.add_argument("--verbose", action="store_true")
    p_preflight.add_argument("--gate-timeout-sec", type=int, default=DEFAULT_GATE_TIMEOUT_SEC)
    p_preflight.set_defaults(func=cmd_preflight)

    p_gates = sub.add_parser("gates", help="run gates for one slice without applying GPT patches")
    _add_common_config_args(p_gates, hidden=True)
    p_gates.add_argument("--slice", required=True)
    p_gates.add_argument("--only-gate", action="append", default=[])
    p_gates.add_argument("--allow-dirty", action="store_true")
    p_gates.add_argument("--keep-going", action="store_true")
    p_gates.add_argument("--verbose", action="store_true")
    p_gates.add_argument("--gate-timeout-sec", type=int, default=DEFAULT_GATE_TIMEOUT_SEC)
    p_gates.add_argument("--log-dir", default="")
    p_gates.set_defaults(func=cmd_gates)

    p_run = sub.add_parser("run", help="run one slice through the deterministic workflow")
    _add_common_config_args(p_run, hidden=True)
    target = p_run.add_mutually_exclusive_group(required=True)
    target.add_argument("--slice", default="")
    target.add_argument("--next", action="store_true")
    p_run.add_argument("--ignore-deps", action="store_true")
    p_run.add_argument("--allow-dirty", action="store_true")
    p_run.add_argument("--no-commit", action="store_true")
    p_run.add_argument("--verbose", action="store_true")
    p_run.add_argument("--gpt-mode", choices=["current_tab", "new"], default="current_tab")
    p_run.add_argument("--chat-timeout-sec", type=int, default=DEFAULT_CHAT_TIMEOUT_SEC)
    p_run.add_argument("--codex-timeout-sec", type=int, default=DEFAULT_CODEX_TIMEOUT_SEC)
    p_run.add_argument("--gate-timeout-sec", type=int, default=DEFAULT_GATE_TIMEOUT_SEC)
    p_run.add_argument("--browser-internal-timeout-ms", type=int, default=DEFAULT_BROWSER_INTERNAL_TIMEOUT_MS)
    p_run.add_argument("--artifact-download-timeout-ms", type=int, default=DEFAULT_ARTIFACT_DOWNLOAD_TIMEOUT_MS)
    p_run.add_argument("--skip-already-landed-probe", action="store_true", help="do not run declared gates before GPT to detect an already-landed slice")
    p_run.set_defaults(func=cmd_run)


    p_context = sub.add_parser("context", help="build a deterministic context pack for one slice")
    _add_common_config_args(p_context, hidden=True)
    p_context.add_argument("--slice", required=True)
    p_context.add_argument("--mode", choices=["initial", "failure", "diagnosis"], default="initial")
    p_context.add_argument("--failure-packet", default="")
    p_context.add_argument("--out", default="")
    p_context.add_argument("--metadata-out", default="")
    p_context.add_argument("--max-chars", type=int, default=0)
    p_context.add_argument("--allow-large-context", action="store_true")
    p_context.add_argument("--verbose", action="store_true")
    p_context.set_defaults(func=cmd_context)

    p_prompt = sub.add_parser("render-prompt", help="render a GPT Pro prompt for one slice without invoking GPT")
    _add_common_config_args(p_prompt, hidden=True)
    p_prompt.add_argument("--slice", required=True)
    p_prompt.add_argument("--attempt", type=int, default=0)
    p_prompt.add_argument("--mode", choices=["initial", "failure", "diagnosis"], default="initial")
    p_prompt.add_argument("--failure-packet", default="")
    p_prompt.add_argument("--out", default="")
    p_prompt.add_argument("--verbose", action="store_true")
    p_prompt.set_defaults(func=cmd_render_prompt)


    p_dry = sub.add_parser("dry-run", help="safe no-GPT integration dry run of Milestone 1 automation plumbing")
    _add_common_config_args(p_dry, hidden=True)
    p_dry.add_argument("--allow-dirty", action="store_true", help="allow dirty repo during preflight gate")
    p_dry.add_argument("--skip-preflight", action="store_true", help="skip build/check preflight gates during the dry run")
    p_dry.add_argument("--keep-going", action="store_true", help="continue dry-run checks after a failed check")
    p_dry.add_argument("--verbose", action="store_true")
    p_dry.add_argument("--gate-timeout-sec", type=int, default=DEFAULT_GATE_TIMEOUT_SEC)
    p_dry.set_defaults(func=cmd_dry_run)

    p_reset = sub.add_parser("reset-slice")
    _add_common_config_args(p_reset, hidden=True)
    p_reset.add_argument("--slice", required=True)
    p_reset.add_argument("--purge-files", action="store_true", help="also remove .vc4_auto prompts/staging/logs/failure_packets for the slice")
    p_reset.set_defaults(func=cmd_reset_slice)

    p_list = sub.add_parser("list-gates")
    _add_common_config_args(p_list, hidden=True)
    p_list.set_defaults(func=cmd_list_gates)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DriverError as exc:
        print(f"[vc4-m1] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
