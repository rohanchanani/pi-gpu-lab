#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from vc4_codegen_state import MilestoneConfig, StateStore, git_head
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_state import MilestoneConfig, StateStore, git_head  # type: ignore


DEFAULT_WORKLIST = "pro_scripts/vc4_codegen_m2_worklist.json"
DEFAULT_CONTEXT_PROFILES = "pro_scripts/vc4_codegen_m2_context_profiles.json"
DEFAULT_SPEC = "pro_scripts/vc4_codegen_m2_verifications.json"
DEFAULT_STATE_ROOT = ".vc4_auto/codegen_m2"
DEFAULT_FROM_SLICE = "m2-02-program-bundle-assembly"
DEFAULT_TIMEOUT_SEC = 1800
DEFAULT_VERIFIER_SCRIPT = "pro_scripts/vc4_milestone_verifier.py"
DEFAULT_AUTORUN_SCRIPT = "pro_scripts/vc4_milestone_autorun.py"
MILESTONE_CONFIG_REQUIRED_FIELDS = (
    "schema_version",
    "milestone",
    "title",
    "worklist",
    "verifications",
    "context_profiles",
    "prompt_template_dir",
    "state_root",
    "candidate_state_root",
    "default_from_slice",
    "default_timeout_sec",
    "hardware_required_by_default",
    "generic_scripts",
)


def q(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"\n$ {q(cmd)}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
    )
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def require_file(repo: Path, rel: str) -> Path:
    raw = Path(rel)
    path = raw if raw.is_absolute() else repo / raw
    if not path.exists():
        raise SystemExit(f"error: missing {rel}")
    return path


def resolve_repo_path(repo: Path, path: str) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else repo / raw


def cli_option_present(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def load_milestone_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise SystemExit(f"error: missing milestone config: {path}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: invalid milestone config JSON {path}: {e}") from e
    if not isinstance(data, dict):
        raise SystemExit("error: milestone config must be a JSON object")
    missing = [key for key in MILESTONE_CONFIG_REQUIRED_FIELDS if key not in data]
    if missing:
        raise SystemExit(f"error: milestone config missing required fields: {', '.join(missing)}")
    if data.get("schema_version") != 1:
        raise SystemExit(f"error: milestone config schema_version must be 1: {data.get('schema_version')}")
    for key in ("worklist", "context_profiles", "verifications", "state_root", "default_from_slice"):
        if not isinstance(data.get(key), str) or not data.get(key):
            raise SystemExit(f"error: milestone config field must be a non-empty string: {key}")
    try:
        data["default_timeout_sec"] = int(data["default_timeout_sec"])
    except (TypeError, ValueError) as e:
        raise SystemExit("error: milestone config default_timeout_sec must be an integer") from e
    generic_scripts = data.get("generic_scripts")
    if not isinstance(generic_scripts, dict):
        raise SystemExit("error: milestone config generic_scripts must be an object")
    return data


def log_prefix(milestone: str) -> str:
    safe = str(milestone or "vc4-milestone").strip() or "vc4-milestone"
    return f"[{safe}-resume]"


def git_status(repo: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=str(repo),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit(proc.returncode)
    return proc.stdout.strip()


def ensure_clean(repo: Path, env: dict[str, str], *, allow_dirty: bool) -> None:
    dirty = git_status(repo, env)
    if dirty and not allow_dirty:
        print("error: repo is dirty; refusing to start/continue automation:", file=sys.stderr)
        print(dirty, file=sys.stderr)
        print("Commit/stash/restore first, or pass --allow-dirty intentionally.", file=sys.stderr)
        raise SystemExit(2)


def load_worklist(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("slices"), list):
        raise SystemExit(f"error: {path} has no slices[]")
    return data


def build_resume_state_store(
    *,
    repo: Path,
    worklist: Path,
    context_profiles: Path,
    spec: Path,
    state_root: Path,
    milestone_config: dict[str, Any] | None,
) -> tuple[MilestoneConfig, StateStore]:
    """Create the same StateStore view used by vc4_milestone_autorun.

    The resume driver can prove a slice is already complete by running the
    typed verifier directly.  When that happens, the autorun StateStore must be
    synchronized too; otherwise the next slice may fail dependency checks even
    though resume just verified its predecessor.  Keep this helper descriptor-
    driven and behavior-preserving: it only mirrors the milestone defaults that
    autorun applies before constructing StateStore.
    """
    config = MilestoneConfig.load(
        repo,
        worklist_path=str(worklist),
        context_profiles_path=str(context_profiles),
    )

    defaults = config.worklist.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        raise SystemExit("error: worklist.defaults must be an object for resume state synchronization")

    defaults["state_root"] = str(state_root)
    defaults["verification_spec"] = str(spec)

    if milestone_config is not None:
        if milestone_config.get("candidate_state_root"):
            defaults["candidate_state_root"] = str(milestone_config["candidate_state_root"])
        if milestone_config.get("prompt_template_dir"):
            defaults["prompt_template_dir"] = str(milestone_config["prompt_template_dir"])
        generic_scripts = milestone_config.get("generic_scripts")
        if isinstance(generic_scripts, dict):
            if generic_scripts.get("verifier"):
                defaults["typed_verifier_script"] = str(generic_scripts["verifier"])

    state = StateStore(config)
    state.ensure_dirs()
    return config, state


def sync_verified_slice_state(
    *,
    config: MilestoneConfig,
    state: StateStore,
    slice_entry: dict[str, Any],
    prefix: str,
    reason: str,
) -> None:
    """Mark a verifier-passing slice as passed in autorun state.

    Resume skips GPT/autorun for slices whose typed verifier already passes.
    Those skips are correct, but dependency checks for later slices consult the
    autorun StateStore.  Synchronize that state exactly when the typed verifier
    has established the slice is complete.
    """
    slice_id = str(slice_entry["id"])
    data = state.load()
    slices = data.get("slices", {})
    if isinstance(slices, dict):
        entry = slices.get(slice_id)
        if isinstance(entry, dict) and entry.get("status") == "passed":
            return

    gate_results = [
        {
            "gate": "resume-typed-verifier",
            "ok": True,
            "source": "vc4_milestone_resume.py",
            "reason": reason,
        }
    ]
    state.mark_slice_passed(
        slice_id=slice_id,
        gate_results=gate_results,
        repo_head=git_head(config.repo, allow_missing=True),
    )
    attempt = state.next_attempt_index(slice_id)
    state.record_attempt(
        slice_id=slice_id,
        attempt=attempt,
        status="passed",
        details={"resume_state_sync": True, "reason": reason},
    )
    print(f"{prefix} state synchronized: marked {slice_id} passed ({reason})", flush=True)


def select_slices(
    slices: list[dict[str, Any]],
    *,
    from_slice: str | None,
    to_slice: str | None,
    only: set[str] | None,
    skip_hardware: bool,
) -> list[dict[str, Any]]:
    known = [str(s.get("id")) for s in slices]
    if only:
        missing = sorted(only - set(known))
        if missing:
            raise SystemExit(f"error: unknown --only-slice value(s): {', '.join(missing)}")
        selected = [s for s in slices if str(s.get("id")) in only]
    else:
        selected = list(slices)

        if from_slice:
            if from_slice not in known:
                raise SystemExit(f"error: unknown --from-slice {from_slice}")
            start = known.index(from_slice)
            selected = selected[start:]

        if to_slice:
            if to_slice not in known:
                raise SystemExit(f"error: unknown --to-slice {to_slice}")
            stop = [str(s.get("id")) for s in selected].index(to_slice)
            selected = selected[: stop + 1]

    if skip_hardware:
        selected = [s for s in selected if not bool(s.get("hardware_required", False))]

    return selected


def verifier_probe(
    *,
    repo: Path,
    python: str,
    verifier_script: Path,
    milestone_config: Path | None,
    spec: Path,
    worklist: Path,
    state_root: Path,
    slice_id: str,
    timeout_sec: int,
    env: dict[str, str],
    verbose: bool,
    prefix: str,
) -> bool:
    out_dir = repo / state_root / "resume_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slice_id}.json"

    cmd = [
        python,
        str(verifier_script),
        "verify",
        "--repo",
        str(repo),
        "--spec",
        str(spec),
        "--worklist",
        str(worklist),
        "--slice",
        slice_id,
        "--out",
        str(out),
        "--timeout-sec",
        str(timeout_sec),
        "--keep-going",
    ]
    if milestone_config is not None:
        cmd += ["--milestone-config", str(milestone_config)]

    if verbose:
        proc = run_cmd(cmd, cwd=repo, env=env)
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            print(f"{prefix} verifier says not yet passed: {slice_id}")
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
            if tail:
                print(tail)
    return proc.returncode == 0


def cumulative_prefix_probe(
    *,
    repo: Path,
    python: str,
    verifier_script: Path,
    milestone_config: Path | None,
    spec: Path,
    worklist: Path,
    state_root: Path,
    slice_ids: list[str],
    timeout_sec: int,
    env: dict[str, str],
    verbose: bool,
    prefix: str,
) -> bool:
    """Re-verify all selected prior slices after a new slice commit.

    This catches cross-slice regressions before the runner starts the next slice.
    Hardware flakiness is handled inside the typed verifier; this function does
    not skip or weaken redundant verification.
    """
    for sid in slice_ids:
        if not verifier_probe(
            repo=repo,
            python=python,
            verifier_script=verifier_script,
            milestone_config=milestone_config,
            spec=spec,
            worklist=worklist,
            state_root=state_root,
            slice_id=sid,
            timeout_sec=timeout_sec,
            env=env,
            verbose=verbose,
            prefix=prefix,
        ):
            print(f"{prefix} cumulative prefix check failed at {sid}", file=sys.stderr)
            return False
    return True


def run_slice(
    *,
    repo: Path,
    python: str,
    autorun_script: Path,
    milestone_config: Path | None,
    worklist: Path,
    context_profiles: Path,
    spec: Path,
    slice_id: str,
    gpt_mode: str,
    allow_dirty: bool,
    verbose: bool,
    env: dict[str, str],
) -> int:
    cmd = [
        python,
        str(autorun_script),
        "--repo",
        str(repo),
        "--worklist",
        str(worklist),
        "--context-profiles",
        str(context_profiles),
        "--spec",
        str(spec),
        "run",
        "--slice",
        slice_id,
        "--gpt-mode",
        gpt_mode,
    ]
    if milestone_config is not None:
        cmd += ["--milestone-config", str(milestone_config)]

    if verbose:
        cmd.append("--verbose")
    if allow_dirty:
        cmd.append("--allow-dirty")

    return run_cmd(cmd, cwd=repo, env=env).returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resumable one-slice-at-a-time runner for VC4 codegen milestones."
    )
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--milestone-config", default="")
    ap.add_argument("--worklist", default=DEFAULT_WORKLIST)
    ap.add_argument("--context-profiles", default=DEFAULT_CONTEXT_PROFILES)
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--state-root", default=DEFAULT_STATE_ROOT)
    ap.add_argument("--from-slice", default=DEFAULT_FROM_SLICE)
    ap.add_argument("--to-slice", default=None)
    ap.add_argument("--only-slice", action="append", default=None)
    ap.add_argument("--gpt-mode", default="current_tab", choices=["current_tab", "new"])
    ap.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-hardware", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    raw_args = sys.argv[1:]
    args = ap.parse_args(raw_args)

    repo = Path(args.repo).resolve()
    milestone_config: dict[str, Any] | None = None
    milestone_config_path: Path | None = None
    if args.milestone_config:
        milestone_config_path = resolve_repo_path(repo, args.milestone_config)
        milestone_config = load_milestone_config(milestone_config_path)

        if not cli_option_present(raw_args, "--worklist"):
            args.worklist = str(milestone_config["worklist"])
        if not cli_option_present(raw_args, "--context-profiles"):
            args.context_profiles = str(milestone_config["context_profiles"])
        if not cli_option_present(raw_args, "--spec"):
            args.spec = str(milestone_config["verifications"])
        if not cli_option_present(raw_args, "--state-root"):
            args.state_root = str(milestone_config["state_root"])
        if not cli_option_present(raw_args, "--from-slice"):
            args.from_slice = str(milestone_config["default_from_slice"])
        if not cli_option_present(raw_args, "--timeout-sec"):
            args.timeout_sec = int(milestone_config["default_timeout_sec"])

    generic_scripts = milestone_config.get("generic_scripts", {}) if milestone_config else {}
    verifier_script_arg = str(generic_scripts.get("verifier") or DEFAULT_VERIFIER_SCRIPT)
    autorun_script_arg = str(generic_scripts.get("autorun") or DEFAULT_AUTORUN_SCRIPT)

    worklist = require_file(repo, args.worklist)
    context_profiles = require_file(repo, args.context_profiles)
    spec = require_file(repo, args.spec)
    autorun_script = require_file(repo, autorun_script_arg)
    verifier_script = require_file(repo, verifier_script_arg)

    state_root = Path(args.state_root)
    env = os.environ.copy()
    env["PATH"] = f"{repo / 'compiler/build/bin'}{os.pathsep}{env.get('PATH', '')}"

    data = load_worklist(worklist)
    resume_state_config, resume_state = build_resume_state_store(
        repo=repo,
        worklist=worklist,
        context_profiles=context_profiles,
        spec=spec,
        state_root=state_root,
        milestone_config=milestone_config,
    )
    milestone = str((milestone_config or {}).get("milestone") or data.get("milestone") or "vc4-codegen")
    prefix = log_prefix(milestone)
    slices = select_slices(
        data["slices"],
        from_slice=args.from_slice,
        to_slice=args.to_slice,
        only=set(args.only_slice) if args.only_slice else None,
        skip_hardware=args.skip_hardware,
    )

    if not slices:
        print(f"{prefix} no slices selected")
        return 0

    print(f"{prefix} selected slices:")
    for s in slices:
        hw = " hardware" if s.get("hardware_required") else ""
        print(f"  - {s['id']}{hw}: {s.get('title', '')}")

    if args.dry_run:
        return 0

    ensure_clean(repo, env, allow_dirty=args.allow_dirty)

    python = sys.executable or "python3"

    for slice_index, s in enumerate(slices):
        slice_id = str(s["id"])
        print(f"\n{prefix} === {slice_id} ===", flush=True)

        ensure_clean(repo, env, allow_dirty=args.allow_dirty)

        if verifier_probe(
            repo=repo,
            python=python,
            verifier_script=verifier_script,
            milestone_config=milestone_config_path,
            spec=spec,
            worklist=worklist,
            state_root=state_root,
            slice_id=slice_id,
            timeout_sec=args.timeout_sec,
            env=env,
            verbose=args.verbose,
            prefix=prefix,
        ):
            sync_verified_slice_state(
                config=resume_state_config,
                state=resume_state,
                slice_entry=s,
                prefix=prefix,
                reason="already_satisfied",
            )
            print(f"{prefix} PASS already satisfied; skipping GPT for {slice_id}")
            continue

        rc = run_slice(
            repo=repo,
            python=python,
            autorun_script=autorun_script,
            milestone_config=milestone_config_path,
            worklist=worklist,
            context_profiles=context_profiles,
            spec=spec,
            slice_id=slice_id,
            gpt_mode=args.gpt_mode,
            allow_dirty=args.allow_dirty,
            verbose=args.verbose,
            env=env,
        )
        if rc != 0:
            print(f"{prefix} STOP: {slice_id} failed with exit code {rc}", file=sys.stderr)
            return rc

        ensure_clean(repo, env, allow_dirty=args.allow_dirty)

        if not verifier_probe(
            repo=repo,
            python=python,
            verifier_script=verifier_script,
            milestone_config=milestone_config_path,
            spec=spec,
            worklist=worklist,
            state_root=state_root,
            slice_id=slice_id,
            timeout_sec=args.timeout_sec,
            env=env,
            verbose=args.verbose,
            prefix=prefix,
        ):
            print(f"{prefix} STOP: {slice_id} did not verify after autorun", file=sys.stderr)
            return 1

        sync_verified_slice_state(
            config=resume_state_config,
            state=resume_state,
            slice_entry=s,
            prefix=prefix,
            reason="post_autorun_verifier",
        )
        print(f"{prefix} PASS {slice_id}")

        prefix_ids = [str(item["id"]) for item in slices[: slice_index + 1]]
        print(f"{prefix} cumulative prefix recheck after commit: " + ", ".join(prefix_ids), flush=True)
        if not cumulative_prefix_probe(
            repo=repo,
            python=python,
            verifier_script=verifier_script,
            milestone_config=milestone_config_path,
            spec=spec,
            worklist=worklist,
            state_root=state_root,
            slice_ids=prefix_ids,
            timeout_sec=args.timeout_sec,
            env=env,
            verbose=args.verbose,
            prefix=prefix,
        ):
            print(f"{prefix} STOP: cumulative prefix did not verify after {slice_id}", file=sys.stderr)
            return 1

    print(f"\n{prefix} all selected slices passed or were already satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
