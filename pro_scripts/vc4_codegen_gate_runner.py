#!/usr/bin/env python3
"""Run deterministic gates for VC4 codegen Milestone 1 slices.

Gates are named in pro_scripts/vc4_codegen_m1_worklist.json.  This script maps
those names to explicit local commands/checks.  GPT Pro never supplies shell
commands directly; adding or changing a gate requires changing this script or
the static worklist.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from vc4_codegen_state import (
        DriverError,
        MilestoneConfig,
        StateStore,
        atomic_write_text,
        ensure_auto_excluded,
        ensure_clean_repo,
        executable_in_path,
        find_repo_root,
        git_status_paths,
        read_json_file,
        relpath,
        tail_file,
        write_json_file,
    )
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from vc4_codegen_state import (  # type: ignore
        DriverError,
        MilestoneConfig,
        StateStore,
        atomic_write_text,
        ensure_auto_excluded,
        ensure_clean_repo,
        executable_in_path,
        find_repo_root,
        git_status_paths,
        read_json_file,
        relpath,
        tail_file,
        write_json_file,
    )


@dataclasses.dataclass
class CommandResult:
    gate: str
    ok: bool
    command: list[str]
    cwd: Path
    log_path: Path
    exit_code: int | None
    timed_out: bool
    elapsed_sec: float
    message: str = ""

    def as_json(self, repo: Path) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "ok": self.ok,
            "command": self.command,
            "cwd": relpath(repo, self.cwd) if self.cwd.exists() else str(self.cwd),
            "log_path": relpath(repo, self.log_path) if self.log_path.exists() else str(self.log_path),
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "message": self.message,
        }


class GateRunner:
    def __init__(self, config: MilestoneConfig, *, verbose: bool = False, timeout_sec: int = 1800):
        self.config = config
        self.repo = config.repo
        self.verbose = verbose
        self.timeout_sec = timeout_sec
        self.state = StateStore(config)
        self.previous_results: dict[str, CommandResult] = {}

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    @property
    def build_dir(self) -> Path:
        return self.config.build_dir()

    def build_bin(self, name: str) -> Path:
        return self.build_dir / "bin" / name

    def candidate_dir(self, name: str) -> Path:
        return self.state.root / "candidates" / name

    def test_input(self, name: str) -> Path:
        # Hardware fixtures take precedence.
        hw = self.repo / "compiler/test/CodeGen/VC4/Hardware/Run" / name / "input.mlir"
        if hw.exists():
            return hw
        emit = self.repo / "compiler/test/CodeGen/VC4/Emit"
        mapping = {
            "minimal_thrend": emit / "emit-minimal-thrend.mlir",
            "qpu_bundle_basic": emit / "emit-qpu-bundle-basic.mlir",
            "qpu_ldi_sema": emit / "emit-qpu-ldi-sema.mlir",
            "qpu_branch": emit / "emit-qpu-branch-delay-slots.mlir",
            "simple_memory_output": emit / "emit-simple-memory-output.mlir",
        }
        if name in mapping and mapping[name].exists():
            return mapping[name]
        direct = emit / f"{name}.mlir"
        if direct.exists():
            return direct
        raise DriverError(f"cannot find input.mlir fixture for candidate/test name {name!r}")

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def log_path(self, log_dir: Path, gate: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", gate).strip("_") or "gate"
        return log_dir / f"{safe}.log"

    def run_command(
        self,
        *,
        gate: str,
        cmd: Sequence[str],
        log_dir: Path,
        cwd: Path | None = None,
        timeout_sec: int | None = None,
        env: Mapping[str, str] | None = None,
        allow_nonzero: bool = False,
    ) -> CommandResult:
        cwd = cwd or self.repo
        log_path = self.log_path(log_dir, gate)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        shown = shlex.join([str(x) for x in cmd])
        header = [
            f"# gate: {gate}",
            f"# cwd: {cwd}",
            f"# command: {shown}",
            f"# timeout_sec: {timeout_sec or self.timeout_sec}",
            "",
        ]
        if self.verbose:
            print(f"[vc4-gate] RUN {gate}: {shown}", flush=True)
        try:
            proc = subprocess.run(
                [str(x) for x in cmd],
                cwd=str(cwd),
                env={**os.environ, **dict(env or {})},
                text=True,
                capture_output=True,
                timeout=timeout_sec or self.timeout_sec,
            )
            elapsed = time.monotonic() - started
            body = proc.stdout + ("\n" if proc.stdout and proc.stderr else "") + proc.stderr
            atomic_write_text(log_path, "\n".join(header) + body)
            ok = proc.returncode == 0 or allow_nonzero
            result = CommandResult(
                gate=gate,
                ok=ok,
                command=[str(x) for x in cmd],
                cwd=cwd,
                log_path=log_path,
                exit_code=proc.returncode,
                timed_out=False,
                elapsed_sec=elapsed,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
            body = stdout + ("\n" if stdout and stderr else "") + stderr
            atomic_write_text(log_path, "\n".join(header) + body + f"\n# TIMEOUT after {elapsed:.1f}s\n")
            result = CommandResult(
                gate=gate,
                ok=False,
                command=[str(x) for x in cmd],
                cwd=cwd,
                log_path=log_path,
                exit_code=None,
                timed_out=True,
                elapsed_sec=elapsed,
                message=f"timeout after {elapsed:.1f}s",
            )
        if self.verbose or not result.ok:
            status = "OK" if result.ok else "FAIL"
            print(f"[vc4-gate] {status} {gate} ({result.elapsed_sec:.1f}s) log={relpath(self.repo, result.log_path)}", flush=True)
            if not result.ok:
                print(tail_file(result.log_path, max_lines=80), flush=True)
        self.previous_results[gate] = result
        return result

    def write_check_log(self, *, gate: str, log_dir: Path, ok: bool, message: str) -> CommandResult:
        path = self.log_path(log_dir, gate)
        atomic_write_text(path, message.rstrip() + "\n")
        result = CommandResult(
            gate=gate,
            ok=ok,
            command=[],
            cwd=self.repo,
            log_path=path,
            exit_code=0 if ok else 1,
            timed_out=False,
            elapsed_sec=0.0,
            message=message,
        )
        self.previous_results[gate] = result
        if self.verbose or not ok:
            print(f"[vc4-gate] {'OK' if ok else 'FAIL'} {gate}: {message}", flush=True)
        return result

    # ------------------------------------------------------------------
    # Gate dispatch
    # ------------------------------------------------------------------

    def run_gate(self, gate: str, *, log_dir: Path, allow_dirty: bool = False) -> CommandResult:
        if gate == "git:clean-or-confirm":
            dirty = git_status_paths(self.repo)
            if not dirty:
                return self.write_check_log(gate=gate, log_dir=log_dir, ok=True, message="repo has no non-.vc4_auto changes")
            msg = "repo is dirty:\n" + "\n".join(dirty[:200])
            if allow_dirty:
                return self.write_check_log(gate=gate, log_dir=log_dir, ok=True, message=msg + "\nallow_dirty=true; continuing")
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message=msg + "\npass --allow-dirty to continue")

        if gate == "json:worklist":
            return self._gate_json_file(gate, self.config.worklist_path, log_dir)
        if gate == "json:context-profiles":
            return self._gate_json_file(gate, self.config.context_profiles_path, log_dir)
        if gate == "tool:gpt-web-driver-exists":
            path = self.config.gpt_web_driver()
            ok = path.exists() and path.is_file()
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=ok, message=f"{relpath(self.repo, path)} exists={ok}")
        if gate == "fixture:minimal-thrend-exists":
            return self._gate_minimal_thrend_fixture(gate, log_dir)

        if gate == "build:vc4-opt":
            return self._gate_configured_command(gate, log_dir, ["ninja", "-C", "compiler/build", "vc4-opt"])
        if gate == "build:check-vc4":
            return self._gate_configured_command(gate, log_dir, ["ninja", "-C", "compiler/build", "check-vc4"], timeout_sec=max(self.timeout_sec, 3600))
        if gate == "build:vc4-codegen":
            return self._gate_configured_command(gate, log_dir, ["ninja", "-C", "compiler/build", "vc4-codegen"])

        if gate == "tool:vc4-codegen-help":
            return self.run_command(gate=gate, cmd=[self._vc4_codegen(), "--help"], log_dir=log_dir)
        if gate == "tool:vc4-codegen-skeleton-valid":
            return self._gate_vc4_codegen_generate(gate, log_dir, "minimal_thrend")
        if gate == "tool:vc4-codegen-skeleton-invalid":
            return self._gate_vc4_codegen_invalid(gate, log_dir)

        if gate.startswith("tool:vc4-codegen-generate:"):
            name = gate.split(":", 2)[2]
            return self._gate_vc4_codegen_generate(gate, log_dir, name)
        if gate.startswith("tool:vc4asm-candidate:"):
            name = gate.split(":", 2)[2]
            return self._gate_vc4asm_candidate(gate, log_dir, name)

        if gate.startswith("lit:"):
            return self._gate_lit(gate, log_dir)

        if gate.startswith("cc:generated-header"):
            return self._gate_cc_generated(gate, log_dir, compile_launcher=False)
        if gate.startswith("cc:generated-launcher"):
            return self._gate_cc_generated(gate, log_dir, compile_launcher=True)

        if gate.startswith("candidate:generate:"):
            name = gate.split(":", 2)[2]
            return self._gate_vc4_codegen_generate(gate, log_dir, name)
        if gate.startswith("candidate:assemble:"):
            name = gate.split(":", 2)[2]
            return self._gate_vc4asm_candidate(gate, log_dir, name)
        if gate.startswith("candidate:build:"):
            name = gate.split(":", 2)[2]
            return self._gate_candidate_support(gate, log_dir, name, "build")
        if gate.startswith("hardware:reference:"):
            name = gate.split(":", 2)[2]
            return self._gate_hardware_reference(gate, log_dir, name)
        if gate.startswith("hardware:candidate:"):
            name = gate.split(":", 2)[2]
            return self._gate_candidate_support(gate, log_dir, name, "run")
        if gate.startswith("result:expected-json:"):
            name = gate.split(":", 2)[2]
            return self._gate_expected_json(gate, log_dir, name)

        if gate == "dialect:verify-minimal-thrend-input":
            return self._gate_verify_minimal_thrend(log_dir)

        return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message=f"unknown gate name: {gate}")

    def run_gates(self, gates: Sequence[str], *, log_dir: Path, allow_dirty: bool = False, stop_on_failure: bool = True) -> list[CommandResult]:
        results: list[CommandResult] = []
        log_dir.mkdir(parents=True, exist_ok=True)
        for gate in gates:
            result = self.run_gate(gate, log_dir=log_dir, allow_dirty=allow_dirty)
            results.append(result)
            if stop_on_failure and not result.ok:
                break
        summary_path = log_dir / "gate_summary.json"
        write_json_file(summary_path, [r.as_json(self.repo) for r in results])
        return results

    # ------------------------------------------------------------------
    # Individual gates
    # ------------------------------------------------------------------

    def _gate_json_file(self, gate: str, path: Path, log_dir: Path) -> CommandResult:
        try:
            data = read_json_file(path)
            msg = f"valid JSON: {relpath(self.repo, path)}\ntop-level type: {type(data).__name__}"
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=True, message=msg)
        except Exception as exc:
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message=str(exc))

    def _gate_minimal_thrend_fixture(self, gate: str, log_dir: Path) -> CommandResult:
        root = self.repo / "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend"

        # Stage 2 is installed before Stage 4 normalizes hardware-run support
        # scripts, so this preflight gate should prove the minimal smoke fixture
        # is usable as a reference bundle without requiring the top-level
        # candidate/run wrapper yet. The top-level run.sh is intentionally
        # treated as an optional contract file here; Stage 4 is responsible for
        # adding or normalizing candidate-facing support around this fixture.
        required = [
            root / "input.mlir",
            root / "expected.json",
            root / "reference",
            root / "reference/run.sh",
            root / "reference/Makefile",
            root / "reference/minimal_thrend.qasm",
            root / "reference/minimal_thrend_launch.c",
            root / "reference/minimal_thrend_launch.h",
        ]
        optional = [
            root / "run.sh",
            root / "reference/3-test-minimal-thrend.c",
            root / "reference/mailbox.c",
            root / "reference/mailbox.h",
        ]

        missing_required = [relpath(self.repo, p) for p in required if not p.exists()]
        missing_optional = [relpath(self.repo, p) for p in optional if not p.exists()]
        ok = not missing_required
        msg = "minimal_thrend fixture check\n"
        msg += "root: " + relpath(self.repo, root) + "\n"
        if missing_required:
            msg += "missing required files/directories:\n" + "\n".join(missing_required)
        else:
            msg += "all required reference-smoke files/directories are present\n"
            if missing_optional:
                msg += "optional files not present before Stage 4 normalization:\n" + "\n".join(missing_optional)
            else:
                msg += "all optional contract/support files are also present"
        return self.write_check_log(gate=gate, log_dir=log_dir, ok=ok, message=msg)

    def _gate_configured_command(self, gate: str, log_dir: Path, fallback: list[str], timeout_sec: int | None = None) -> CommandResult:
        defaults = self.config.defaults.get("build_commands", {}) if isinstance(self.config.defaults, dict) else {}
        key_map = {
            "build:vc4-opt": "build_vc4_opt",
            "build:check-vc4": "check_vc4",
        }
        raw = defaults.get(key_map.get(gate, "")) if isinstance(defaults, dict) else None
        cmd = shlex.split(str(raw)) if raw else fallback
        return self.run_command(gate=gate, cmd=cmd, log_dir=log_dir, timeout_sec=timeout_sec)

    def _vc4_codegen(self) -> str:
        path = self.build_bin("vc4-codegen")
        if path.exists():
            return str(path)
        in_path = executable_in_path("vc4-codegen")
        if in_path:
            return in_path
        # Return the expected path so the failing log is actionable.
        return str(path)

    def _vc4_opt(self) -> str:
        path = self.build_bin("vc4-opt")
        if path.exists():
            return str(path)
        in_path = executable_in_path("vc4-opt")
        if in_path:
            return in_path
        return str(path)

    def _gate_vc4_codegen_generate(self, gate: str, log_dir: Path, name: str) -> CommandResult:
        out_dir = self.candidate_dir(name)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.test_input(name)
        result = self.run_command(
            gate=gate,
            cmd=[self._vc4_codegen(), str(input_path), "--emit-bundle", str(out_dir)],
            log_dir=log_dir,
        )
        if not result.ok:
            return result
        required = ["kernel.qasm", "kernel_launch.c", "kernel_launch.h", "manifest.json"]
        missing = [f for f in required if not (out_dir / f).exists()]
        if missing:
            return self.write_check_log(
                gate=gate + ":artifact-check",
                log_dir=log_dir,
                ok=False,
                message=f"vc4-codegen succeeded but candidate dir is missing files: {missing}\nout_dir={out_dir}",
            )
        return result

    def _gate_vc4_codegen_invalid(self, gate: str, log_dir: Path) -> CommandResult:
        tmp = self.state.root / "tmp" / "invalid_missing_launch_abi.mlir"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(
            """vc4.module @bad {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>} {
    vc4.qpu.bundle {sig = #vc4.qpu_signal<thrend>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 34 : i32, waddr_mul = 35 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 36 : i32, waddr_mul = 37 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}
""",
            encoding="utf-8",
        )
        out_dir = self.state.root / "tmp" / "invalid_out"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        result = self.run_command(
            gate=gate,
            cmd=[self._vc4_codegen(), str(tmp), "--emit-bundle", str(out_dir)],
            log_dir=log_dir,
            allow_nonzero=True,
        )
        if result.exit_code == 0:
            return self.write_check_log(gate=gate + ":expect-failure", log_dir=log_dir, ok=False, message="invalid input unexpectedly succeeded")
        return self.write_check_log(gate=gate + ":expect-failure", log_dir=log_dir, ok=True, message="invalid input was rejected as expected")

    def _gate_vc4asm_candidate(self, gate: str, log_dir: Path, name: str) -> CommandResult:
        support = self.repo / "compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh"
        if support.exists():
            return self.run_command(gate=gate, cmd=["bash", str(support), name, "assemble"], log_dir=log_dir)

        out_dir = self.candidate_dir(name)
        qasm = out_dir / "kernel.qasm"
        if not qasm.exists():
            gen = self._gate_vc4_codegen_generate("tool:vc4-codegen-generate:" + name, log_dir, name)
            if not gen.ok:
                return gen
        vc4asm = executable_in_path("vc4asm")
        if not vc4asm:
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message="vc4asm not found in PATH")
        return self.run_command(
            gate=gate,
            cmd=[vc4asm, "-c", str(out_dir / "kernelshader.c"), "-h", str(out_dir / "kernelshader.h"), str(qasm)],
            log_dir=log_dir,
        )

    def _gate_lit(self, gate: str, log_dir: Path) -> CommandResult:
        lit = self.build_bin("llvm-lit")
        if not lit.exists():
            found = executable_in_path("llvm-lit") or executable_in_path("lit")
            lit_cmd = found or str(lit)
        else:
            lit_cmd = str(lit)
        emit_dir = self.repo / "compiler/test/CodeGen/VC4/Emit"
        target = emit_dir
        if not target.exists():
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message=f"lit target does not exist yet: {relpath(self.repo, target)}")
        return self.run_command(gate=gate, cmd=[lit_cmd, "-v", str(target)], log_dir=log_dir)

    def _gate_cc_generated(self, gate: str, log_dir: Path, *, compile_launcher: bool) -> CommandResult:
        name = "minimal_thrend"
        out_dir = self.candidate_dir(name)
        if not (out_dir / "kernel_launch.h").exists():
            gen = self._gate_vc4_codegen_generate("tool:vc4-codegen-generate:" + name, log_dir, name)
            if not gen.ok:
                return gen
        cc = executable_in_path("cc") or executable_in_path("clang") or executable_in_path("gcc")
        if not cc:
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message="no C compiler found in PATH")
        if compile_launcher:
            src = out_dir / "kernel_launch.c"
            if not src.exists():
                return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message=f"missing generated launcher C: {src}")
            return self.run_command(gate=gate, cmd=[cc, "-std=c11", "-fsyntax-only", "-I", str(out_dir), str(src)], log_dir=log_dir)
        tmp = self.state.root / "tmp" / "generated_header_probe.c"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text('#include "kernel_launch.h"\nint main(void) { return 0; }\n', encoding="utf-8")
        return self.run_command(gate=gate, cmd=[cc, "-std=c11", "-fsyntax-only", "-I", str(out_dir), str(tmp)], log_dir=log_dir)

    def _gate_candidate_support(self, gate: str, log_dir: Path, name: str, phase: str) -> CommandResult:
        support = self.repo / "compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh"
        if support.exists():
            return self.run_command(gate=gate, cmd=["bash", str(support), name, phase], log_dir=log_dir)
        candidate_dir = self.repo / "compiler/test/CodeGen/VC4/Hardware/Run" / name / "candidate"
        run_sh = candidate_dir / "run.sh"
        if phase == "run" and run_sh.exists():
            return self.run_command(gate=gate, cmd=["bash", "run.sh"], log_dir=log_dir, cwd=candidate_dir, timeout_sec=max(self.timeout_sec, 3600))
        return self.write_check_log(
            gate=gate,
            log_dir=log_dir,
            ok=False,
            message=(
                "candidate support runner is not available yet. Expected "
                f"{relpath(self.repo, support)} or {relpath(self.repo, run_sh)}"
            ),
        )

    def _gate_hardware_reference(self, gate: str, log_dir: Path, name: str) -> CommandResult:
        ref_dir = self.repo / "compiler/test/CodeGen/VC4/Hardware/Run" / name / "reference"
        run_sh = ref_dir / "run.sh"
        if not run_sh.exists():
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message=f"missing reference run.sh: {relpath(self.repo, run_sh)}")
        return self.run_command(gate=gate, cmd=["bash", "run.sh"], log_dir=log_dir, cwd=ref_dir, timeout_sec=max(self.timeout_sec, 3600))

    def _gate_expected_json(self, gate: str, log_dir: Path, name: str) -> CommandResult:
        expected = self.repo / "compiler/test/CodeGen/VC4/Hardware/Run" / name / "expected.json"
        if not expected.exists():
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message=f"missing expected.json: {relpath(self.repo, expected)}")
        checker = self.repo / "compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py"
        previous = None
        for key in [f"hardware:candidate:{name}", f"candidate:run:{name}"]:
            if key in self.previous_results:
                previous = self.previous_results[key]
                break
        if previous is None:
            # Fall back to the newest log that looks like a candidate hardware run.
            candidates = sorted(log_dir.glob(f"*hardware_candidate_{name}*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                previous_log = candidates[0]
            else:
                return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message="no previous candidate hardware log is available for expected.json checking")
        else:
            previous_log = previous.log_path
        if checker.exists():
            return self.run_command(gate=gate, cmd=[sys.executable, str(checker), str(expected), str(previous_log)], log_dir=log_dir)
        return self._builtin_expected_json_check(gate, log_dir, expected, previous_log)

    def _builtin_expected_json_check(self, gate: str, log_dir: Path, expected: Path, log_path: Path) -> CommandResult:
        data = read_json_file(expected)
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        parsed = parse_vc4_test_result(text)
        if not parsed:
            return self.write_check_log(gate=gate, log_dir=log_dir, ok=False, message=f"no VC4_TEST_RESULT line found in {log_path}")
        errors: list[str] = []
        if str(parsed.get("status", "")) != str(data.get("status", "PASS")):
            errors.append(f"status mismatch: expected {data.get('status')} got {parsed.get('status')}")
        if "name" in data and str(parsed.get("name", data.get("name"))) != str(data.get("name")):
            errors.append(f"name mismatch: expected {data.get('name')} got {parsed.get('name')}")
        required = data.get("required", {}) if isinstance(data, dict) else {}
        if isinstance(required, dict):
            for key, expected_value in required.items():
                actual = parsed.get(key)
                if actual is None:
                    errors.append(f"missing required field {key}")
                    continue
                if str(actual) != str(expected_value):
                    errors.append(f"field {key} mismatch: expected {expected_value!r} got {actual!r}")
        return self.write_check_log(
            gate=gate,
            log_dir=log_dir,
            ok=not errors,
            message=("builtin expected.json check passed" if not errors else "builtin expected.json check failed\n" + "\n".join(errors)) + "\nparsed=" + json.dumps(parsed, sort_keys=True),
        )

    def _gate_verify_minimal_thrend(self, log_dir: Path) -> CommandResult:
        input_path = self.repo / "compiler/test/CodeGen/VC4/Hardware/Run/minimal_thrend/input.mlir"
        flags = self.config.defaults.get("required_verifier_pipeline", [])
        if not isinstance(flags, list):
            flags = []
        return self.run_command(gate="dialect:verify-minimal-thrend-input", cmd=[self._vc4_opt(), str(input_path), *[str(f) for f in flags], "-o", os.devnull], log_dir=log_dir)


def parse_vc4_test_result(log_text: str) -> dict[str, Any] | None:
    """Parse a VC4_TEST_RESULT line as JSON or key=value fields."""
    for line in reversed(log_text.splitlines()):
        if "VC4_TEST_RESULT" not in line:
            continue
        tail = line.split("VC4_TEST_RESULT", 1)[1].strip(" :")
        if not tail:
            continue
        if tail.startswith("{"):
            try:
                value = json.loads(tail)
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pass
        fields: dict[str, Any] = {}
        for part in re.split(r"[\s,]+", tail):
            if not part or "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')
            if not key:
                continue
            if re.fullmatch(r"[-+]?\d+", value):
                fields[key] = int(value)
            else:
                try:
                    fields[key] = float(value)
                except ValueError:
                    fields[key] = value
        if fields:
            return fields
    return None


def cmd_run(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    ensure_auto_excluded(repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    runner = GateRunner(config, verbose=args.verbose, timeout_sec=args.timeout_sec)
    slice_entry = config.get_slice(args.slice)
    gates = [str(g) for g in slice_entry.get("gates", [])]
    if args.only_gate:
        selected = set(args.only_gate)
        gates = [g for g in gates if g in selected]
        missing = selected - set(gates)
        if missing:
            raise DriverError(f"requested --only-gate entries are not in slice gate list: {sorted(missing)}")
    log_dir = Path(args.log_dir).resolve() if args.log_dir else runner.state.root / "logs" / str(slice_entry["id"]) / "manual-gates"
    results = runner.run_gates(gates, log_dir=log_dir, allow_dirty=args.allow_dirty, stop_on_failure=not args.keep_going)
    ok = all(r.ok for r in results)
    print(json.dumps([r.as_json(repo) for r in results], indent=2, sort_keys=True), flush=True)
    return 0 if ok else 1


def cmd_list(args: argparse.Namespace) -> int:
    repo = find_repo_root(args.repo)
    config = MilestoneConfig.load(repo, worklist_path=args.worklist, context_profiles_path=args.context_profiles)
    for s in config.slices:
        print(f"{s['id']}: {s['title']}")
        for gate in s.get("gates", []):
            print(f"  - {gate}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--worklist", default="pro_scripts/vc4_codegen_m1_worklist.json")
    parser.add_argument("--context-profiles", default="pro_scripts/vc4_codegen_m1_context_profiles.json")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run gates for one slice")
    p_run.add_argument("--slice", required=True)
    p_run.add_argument("--only-gate", action="append", default=[])
    p_run.add_argument("--allow-dirty", action="store_true")
    p_run.add_argument("--keep-going", action="store_true")
    p_run.add_argument("--verbose", action="store_true")
    p_run.add_argument("--timeout-sec", type=int, default=1800)
    p_run.add_argument("--log-dir", default="")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="list known slice gates")
    p_list.set_defaults(func=cmd_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DriverError as exc:
        print(f"[vc4-gate] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
