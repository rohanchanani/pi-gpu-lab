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


DEFAULT_WORKLIST = "pro_scripts/vc4_codegen_m2_worklist.json"
DEFAULT_CONTEXT_PROFILES = "pro_scripts/vc4_codegen_m2_context_profiles.json"
DEFAULT_SPEC = "pro_scripts/vc4_codegen_m2_verifications.json"
DEFAULT_STATE_ROOT = ".vc4_auto/codegen_m2"
DEFAULT_FROM_SLICE = "m2-02-program-bundle-assembly"


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
    path = repo / rel
    if not path.exists():
        raise SystemExit(f"error: missing {rel}")
    return path


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
    spec: Path,
    worklist: Path,
    state_root: Path,
    slice_id: str,
    timeout_sec: int,
    env: dict[str, str],
    verbose: bool,
) -> bool:
    out_dir = repo / state_root / "resume_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slice_id}.json"

    cmd = [
        python,
        str(repo / "pro_scripts/vc4_codegen_m1_verifier.py"),
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
            print(f"[m2-resume] verifier says not yet passed: {slice_id}")
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-40:])
            if tail:
                print(tail)
    return proc.returncode == 0


def run_slice(
    *,
    repo: Path,
    python: str,
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
        str(repo / "pro_scripts/vc4_codegen_m1_autorun.py"),
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

    if verbose:
        cmd.append("--verbose")
    if allow_dirty:
        cmd.append("--allow-dirty")

    return run_cmd(cmd, cwd=repo, env=env).returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resumable one-slice-at-a-time runner for VC4 Codegen M2."
    )
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--worklist", default=DEFAULT_WORKLIST)
    ap.add_argument("--context-profiles", default=DEFAULT_CONTEXT_PROFILES)
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    ap.add_argument("--state-root", default=DEFAULT_STATE_ROOT)
    ap.add_argument("--from-slice", default=DEFAULT_FROM_SLICE)
    ap.add_argument("--to-slice", default=None)
    ap.add_argument("--only-slice", action="append", default=None)
    ap.add_argument("--gpt-mode", default="current_tab", choices=["current_tab", "new"])
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--skip-hardware", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    worklist = require_file(repo, args.worklist)
    context_profiles = require_file(repo, args.context_profiles)
    spec = require_file(repo, args.spec)
    require_file(repo, "pro_scripts/vc4_codegen_m1_autorun.py")
    require_file(repo, "pro_scripts/vc4_codegen_m1_verifier.py")

    state_root = Path(args.state_root)
    env = os.environ.copy()
    env["PATH"] = f"{repo / 'compiler/build/bin'}{os.pathsep}{env.get('PATH', '')}"

    data = load_worklist(worklist)
    slices = select_slices(
        data["slices"],
        from_slice=args.from_slice,
        to_slice=args.to_slice,
        only=set(args.only_slice) if args.only_slice else None,
        skip_hardware=args.skip_hardware,
    )

    if not slices:
        print("[m2-resume] no slices selected")
        return 0

    print("[m2-resume] selected slices:")
    for s in slices:
        hw = " hardware" if s.get("hardware_required") else ""
        print(f"  - {s['id']}{hw}: {s.get('title', '')}")

    if args.dry_run:
        return 0

    ensure_clean(repo, env, allow_dirty=args.allow_dirty)

    python = sys.executable or "python3"

    for s in slices:
        slice_id = str(s["id"])
        print(f"\n[m2-resume] === {slice_id} ===", flush=True)

        ensure_clean(repo, env, allow_dirty=args.allow_dirty)

        if verifier_probe(
            repo=repo,
            python=python,
            spec=spec,
            worklist=worklist,
            state_root=state_root,
            slice_id=slice_id,
            timeout_sec=args.timeout_sec,
            env=env,
            verbose=args.verbose,
        ):
            print(f"[m2-resume] PASS already satisfied; skipping GPT for {slice_id}")
            continue

        rc = run_slice(
            repo=repo,
            python=python,
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
            print(f"[m2-resume] STOP: {slice_id} failed with exit code {rc}", file=sys.stderr)
            return rc

        ensure_clean(repo, env, allow_dirty=args.allow_dirty)

        if not verifier_probe(
            repo=repo,
            python=python,
            spec=spec,
            worklist=worklist,
            state_root=state_root,
            slice_id=slice_id,
            timeout_sec=args.timeout_sec,
            env=env,
            verbose=args.verbose,
        ):
            print(f"[m2-resume] STOP: {slice_id} did not verify after autorun", file=sys.stderr)
            return 1

        print(f"[m2-resume] PASS {slice_id}")

    print("\n[m2-resume] all selected M2 slices passed or were already satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
