#!/usr/bin/env python3
"""Audit final VC4Kernel representative artifacts across the full pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SEMANTIC_RESOURCE_FIELDS = {
    "schedule_mode",
    "warps_per_block",
    "user_vpm_rows_per_block",
    "compiler_vpm_staging_rows_per_warp",
    "compiler_vpm_staging_rows_per_block",
    "spill_vpm_rows_per_block",
    "total_vpm_rows_per_block",
    "uses_tmu",
    "uses_vpm",
    "uses_vpm_qpu_read",
    "uses_vpm_qpu_write",
    "uses_vdr",
    "uses_vdw",
    "uses_barrier",
    "semaphore_count_per_block",
    "requires_vpm_base_row_builtin",
    "requires_semaphore_base_builtin",
    "max_resident_blocks",
}

OLD_RESOURCE_FIELDS = [
    "shared_" "vpm_bytes",
    "user_shared_" "vpm_rows_" "per_block",
    "vpm_" "rows_" "per_block",
    "vpm_" "bytes_" "per_block",
    "warps_" "per_block_max",
    "uses_" "shared_vpm",
    "require_" "full_block_residency",
    "semaphores_" "per_block",
]

PRODUCER_DIALECTS = {
    "affine",
    "gpu",
    "linalg",
    "math",
    "memref",
    "nvgpu",
    "nvvm",
    "rocdl",
    "scf",
    "spirv",
    "stablehlo",
    "tensor",
    "tosa",
    "triton",
    "tt",
    "ttg",
    "vector",
}


@dataclass(frozen=True)
class Fixture:
    name: str
    dialect: str
    group: str
    markers: tuple[str, ...] = ()


REPRESENTATIVE_FIXTURES = (
    Fixture(
        "mixed_surface_lock_elementwise_full_vc4kernel",
        "vc4kernel",
        "elementwise_tmu_vdw_sfu_rotate_pack_unpack_dynamic_selector",
        (
            "uses_tmu",
            "uses_vdw",
            "dynamic_rotate",
            "pack_unpack",
            "sfu",
            "vpm_base",
        ),
    ),
    Fixture(
        "mixed_surface_lock_vpm_pipeline_full_vc4kernel",
        "vc4kernel",
        "vpm_pipeline_dynamic_coordinates_selectors",
        ("uses_vdr", "uses_vdw", "uses_vpm_qpu_read", "dynamic_vpmvcd"),
    ),
    Fixture(
        "mixed_surface_lock_dynamic_pingpong_qpu_compute_vc4kernel",
        "vc4kernel",
        "checked_dynamic_pingpong_vpm_qpu_readback",
        ("uses_vpm_qpu_read", "dynamic_vpmvcd", "checked_pingpong_read"),
    ),
    Fixture(
        "mixed_blocked_gemv_vpm_fullstack_vc4kernel",
        "vc4kernel",
        "blocked_gemv",
        ("uses_vdr", "uses_vdw", "vpm_base"),
    ),
    Fixture(
        "mixed_blocked_gemm_vpm_fullstack_vc4kernel",
        "vc4kernel",
        "blocked_gemm",
        ("uses_vdr", "uses_vdw", "vpm_base"),
    ),
    Fixture(
        "mixed_surface_lock_cooperative_barrier_full_vc4kernel",
        "vc4kernel",
        "cooperative_barrier_vpm",
        ("uses_barrier", "semaphore_base", "uses_vpm_qpu_read"),
    ),
    Fixture(
        "mixed_lower_half_spill_dma_branch_ssavc4",
        "ssavc4",
        "lower_half_spill_dma_branch",
        ("uses_vdr", "uses_vdw", "spill_rows", "branch", "no_tmu_if_resource_says_no"),
    ),
)


def fail(message: str) -> None:
    print(f"audit_final_artifact_boundary.py: ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        rendered = " ".join(cmd)
        fail(
            f"command failed ({proc.returncode}): {rendered}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def find_tool(repo_root: Path, name: str, required: bool = True) -> Path | None:
    candidate = repo_root / "compiler" / "build" / "bin" / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    if required:
        fail(f"required tool not found: {name}")
    return None


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"required file not found: {path}")


def dialect_prefixes(text: str) -> set[str]:
    prefixes: set[str] = set()
    op_pattern = re.compile(
        r'^\s*(?:%[A-Za-z0-9_]+(?:\s*,\s*%[A-Za-z0-9_]+)*\s*=\s*)?'
        r'"?([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_.]*"?(?=\s|<|\{|\()'
    )
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('"') and '" =' in stripped:
            continue
        match = op_pattern.match(line)
        if match:
            prefixes.add(match.group(1))
    return prefixes


def require_no_old_resource_fields(path: Path) -> None:
    text = read(path)
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])("
        + "|".join(re.escape(field) for field in OLD_RESOURCE_FIELDS)
        + r")(?![A-Za-z0-9_])"
    )
    match = pattern.search(text)
    if match:
        fail(f"{path}: old resource field remains: {match.group(1)}")


def require_no_source_producers(path: Path, text: str) -> None:
    hits = sorted(dialect_prefixes(text) & PRODUCER_DIALECTS)
    if hits:
        fail(f"{path}: source producer dialect ops remain: {', '.join(hits)}")


def validate_vc4kernel_boundary(path: Path) -> None:
    text = read(path)
    if "vc4kernel.kernel" not in text:
        fail(f"{path}: expected vc4kernel.kernel")
    if "ssavc4." in text:
        fail(f"{path}: VC4Kernel boundary contains ssavc4.*")
    if re.search(r'(?<![A-Za-z0-9_.])"?vc4\.(module|qpu)\b', text):
        fail(f"{path}: VC4Kernel boundary contains scheduled vc4")
    require_no_source_producers(path, text)


def validate_ssavc4_boundary(path: Path, *, lowered_from_vc4kernel: bool) -> None:
    text = read(path)
    if "ssavc4.func" not in text:
        fail(f"{path}: expected ssavc4.func")
    if "vc4kernel." in text:
        fail(f"{path}: SSAVC4 boundary still contains vc4kernel.*")
    if re.search(r'(?<![A-Za-z0-9_.])"?vc4\.(module|qpu)\b', text):
        fail(f"{path}: SSAVC4 boundary already contains scheduled VC4")
    prefixes = dialect_prefixes(text)
    allowed = {"ssavc4", "arith", "cf"}
    extra = sorted(prefixes - allowed)
    if extra:
        fail(f"{path}: SSAVC4 boundary has unexpected dialect ops: {', '.join(extra)}")
    if lowered_from_vc4kernel:
        require_no_source_producers(path, text)


def validate_vc4_boundary(path: Path) -> None:
    text = read(path)
    if "vc4.module" not in text:
        fail(f"{path}: expected vc4.module")
    if "vc4.qpu." not in text:
        fail(f"{path}: scheduled VC4 output contains no vc4.qpu.*")
    for forbidden in ("vc4kernel.", "ssavc4."):
        if forbidden in text:
            fail(f"{path}: scheduled VC4 boundary contains {forbidden}")
    prefixes = dialect_prefixes(text)
    extra = sorted(prefixes - {"vc4"})
    if extra:
        fail(f"{path}: scheduled VC4 boundary has unexpected dialect ops: {', '.join(extra)}")


def load_manifest(path: Path) -> dict:
    require_file(path)
    try:
        return json.loads(read(path))
    except json.JSONDecodeError as exc:
        fail(f"{path}: invalid JSON: {exc}")


def resource_limit(total: int, per_block: int) -> int:
    if per_block <= 0:
        return total
    return total // per_block


def expected_resident_blocks(resources: dict) -> int:
    return min(
        resource_limit(12, int(resources["warps_per_block"])),
        resource_limit(16, int(resources["semaphore_count_per_block"])),
        resource_limit(64, int(resources["total_vpm_rows_per_block"])),
    )


def validate_resources(fixture: Fixture, bundle: Path, manifest: dict) -> list[str]:
    source = read(bundle / "kernel_launch.c")
    header = read(bundle / "kernel_launch.h")
    if ".resource = {" not in source or "VC4_KERNEL_RESOURCE" not in source:
        fail(f"{fixture.name}: generated C lacks semantic resource descriptor")
    if "struct vc4_kernel_resource" not in header and ".resource = {" not in source:
        fail(f"{fixture.name}: generated artifacts do not expose libpi resource fields")
    for artifact in (bundle / "kernel_launch.c", bundle / "kernel_launch.h", bundle / "manifest.json"):
        require_no_old_resource_fields(artifact)

    kernels = manifest.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        fail(f"{fixture.name}: manifest must contain non-empty kernels[]")

    findings: list[str] = []
    saw_barrier = False
    saw_vpm_rows = False
    saw_spill_rows = False
    saw_tmu = False
    saw_vdr = False
    saw_vdw = False
    saw_vpm_qpu_read = False

    for kernel in kernels:
        resources = kernel.get("resources")
        if not isinstance(resources, dict):
            fail(f"{fixture.name}: manifest kernel missing resources object")
        missing = sorted(SEMANTIC_RESOURCE_FIELDS - set(resources))
        if missing:
            fail(f"{fixture.name}: resources missing fields: {missing}")
        rows_expected = (
            int(resources["user_vpm_rows_per_block"])
            + int(resources["compiler_vpm_staging_rows_per_block"])
            + int(resources["warps_per_block"])
            * int(resources["compiler_vpm_staging_rows_per_warp"])
            + int(resources["spill_vpm_rows_per_block"])
        )
        if int(resources["total_vpm_rows_per_block"]) != rows_expected:
            fail(
                f"{fixture.name}: total_vpm_rows_per_block does not include "
                f"user/compiler/spill rows"
            )
        if int(resources["max_resident_blocks"]) != expected_resident_blocks(resources):
            fail(f"{fixture.name}: max_resident_blocks is stale or inconsistent")
        if bool(resources["requires_vpm_base_row_builtin"]) != (
            int(resources["total_vpm_rows_per_block"]) > 0
        ):
            fail(f"{fixture.name}: requires_vpm_base_row_builtin mismatch")
        if bool(resources["requires_semaphore_base_builtin"]) != (
            int(resources["semaphore_count_per_block"]) > 0
        ):
            fail(f"{fixture.name}: requires_semaphore_base_builtin mismatch")
        if int(resources["total_vpm_rows_per_block"]) > 0:
            saw_vpm_rows = True
            if "vpm_base_row" not in source and "spill_vpm_row" not in source:
                fail(f"{fixture.name}: VPM rows require runtime VPM row assignment")
        if int(resources["semaphore_count_per_block"]) > 0 or bool(resources["uses_barrier"]):
            saw_barrier = True
            if "semaphore_base" not in source:
                fail(f"{fixture.name}: barrier resources require semaphore_base assignment")
        saw_spill_rows |= int(resources["spill_vpm_rows_per_block"]) > 0
        saw_tmu |= bool(resources["uses_tmu"])
        saw_vdr |= bool(resources["uses_vdr"])
        saw_vdw |= bool(resources["uses_vdw"])
        saw_vpm_qpu_read |= bool(resources["uses_vpm_qpu_read"])

    if "uses_tmu" in fixture.markers and not saw_tmu:
        fail(f"{fixture.name}: expected TMU resource use")
    if "uses_vdr" in fixture.markers and not saw_vdr:
        fail(f"{fixture.name}: expected VDR resource use")
    if "uses_vdw" in fixture.markers and not saw_vdw:
        fail(f"{fixture.name}: expected VDW resource use")
    if "uses_barrier" in fixture.markers and not saw_barrier:
        fail(f"{fixture.name}: expected barrier/semaphore resources")
    if "uses_vpm_qpu_read" in fixture.markers and not saw_vpm_qpu_read:
        fail(f"{fixture.name}: expected QPU VPM read resource use")
    if "spill_rows" in fixture.markers and not saw_spill_rows:
        fail(f"{fixture.name}: expected spill VPM rows")
    if "vpm_base" in fixture.markers and not saw_vpm_rows:
        fail(f"{fixture.name}: expected nonzero VPM rows")

    findings.append(
        "resources: semantic descriptors present, old fields absent, "
        "resident_blocks recomputed, VPM/semaphore builtins checked"
    )
    return findings


def qasm_paths(bundle: Path, manifest: dict) -> list[Path]:
    paths: list[Path] = []
    for kernel in manifest.get("kernels", []):
        qasm = kernel.get("qasm_path")
        if not isinstance(qasm, str) or not qasm:
            fail(f"{bundle}: manifest kernel missing qasm_path")
        path = bundle / qasm
        require_file(path)
        paths.append(path)
    return paths


def validate_qasm_with_assembler(vc4asm: Path | None, bundle: Path, qasms: Iterable[Path]) -> None:
    if vc4asm is None:
        fail("vc4asm is required for final QASM boundary audit")
    asm_dir = bundle / "_asm_check"
    asm_dir.mkdir(parents=True, exist_ok=True)
    for qasm in qasms:
        rel = qasm.relative_to(bundle)
        out_c = asm_dir / (qasm.stem + ".c")
        out_h = asm_dir / (qasm.stem + ".h")
        proc = subprocess.run(
            [str(vc4asm), "-c", str(out_c), "-h", str(out_h), str(rel)],
            cwd=str(bundle),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            fail(
                f"vc4asm failed for {qasm}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        combined = (proc.stdout + proc.stderr).strip()
        if combined:
            fail(f"vc4asm produced warning/output for {qasm}: {combined}")
        require_file(out_c)
        require_file(out_h)


def validate_feature_markers(
    fixture: Fixture,
    repo_root: Path,
    source_text: str,
    ssavc4_text: str,
    vc4_text: str,
    qasm_text: str,
    manifest: dict,
) -> list[str]:
    findings: list[str] = []
    if "dynamic_rotate" in fixture.markers:
        if "fragment_rotate" not in source_text or "ssavc4.rotate" not in ssavc4_text:
            fail(f"{fixture.name}: dynamic rotate did not cross VC4Kernel->SSAVC4")
        if "<< r5" not in qasm_text and "small_imm = 48" not in vc4_text:
            fail(f"{fixture.name}: dynamic rotate r5 handling not visible in artifacts")
        findings.append("dynamic rotate r5 handling present")
    if "pack_unpack" in fixture.markers:
        if "fragment_pack" not in source_text or "fragment_unpack" not in source_text:
            fail(f"{fixture.name}: source lacks pack/unpack representative ops")
        if "pack" not in ssavc4_text or "unpack" not in ssavc4_text:
            fail(f"{fixture.name}: pack/unpack not visible after VC4Kernel lowering")
        findings.append("pack/unpack lowering boundary present")
    if "sfu" in fixture.markers:
        if "fragment_sfu" not in source_text or "ssavc4.sfu" not in ssavc4_text:
            fail(f"{fixture.name}: SFU did not cross VC4Kernel->SSAVC4")
        if "r4" not in qasm_text or "nop" not in qasm_text:
            fail(f"{fixture.name}: SFU artifact lacks visible r4 wait padding")
        findings.append("SFU r4 wait padding present")
    if "dynamic_vpmvcd" in fixture.markers:
        if "dynamic_" not in source_text:
            fail(f"{fixture.name}: source lacks dynamic coordinate/selector spelling")
        for op in ("vc4.qpu.vpmvcd_setup", "vc4.qpu.vpmvcd_addr", "vc4.qpu.vpmvcd_wait"):
            if op not in vc4_text:
                fail(f"{fixture.name}: scheduled VC4 missing {op}")
        findings.append("dynamic VDR/VDW/VPM setup fields present")
    if "checked_pingpong_read" in fixture.markers:
        if "saw_checked_qpu_readback" not in read(
            fixture_expected_path(fixture, repo_root)
        ):
            fail(f"{fixture.name}: checked ping-pong readback expected metadata absent")
        if "vpm_read_fragment" not in source_text or "ssavc4.vpm.read" not in ssavc4_text:
            fail(f"{fixture.name}: checked ping-pong QPU readback path missing")
        findings.append("checked ping-pong QPU readback present")
    if "branch" in fixture.markers:
        if "vc4.qpu.branch" not in vc4_text or re.search(r"^brr?\b", qasm_text, re.M) is None:
            fail(f"{fixture.name}: branch lowering not visible in artifacts")
        findings.append("branch/layout path present")
    if "no_tmu_if_resource_says_no" in fixture.markers:
        any_tmu = any(
            bool(k.get("resources", {}).get("uses_tmu"))
            for k in manifest.get("kernels", [])
            if isinstance(k, dict)
        )
        if not any_tmu and "ldtmu" in qasm_text:
            fail(f"{fixture.name}: hidden spill path uses TMU/ldtmu despite uses_tmu=false")
        findings.append("hidden spill reload has no TMU/ldtmu when resources say no TMU")
    return findings


def fixture_expected_path(fixture: Fixture, repo_root: Path) -> Path:
    if fixture.dialect == "vc4kernel":
        return (
            repo_root
            / "compiler/test/CodeGen/VC4Kernel/Hardware/Run"
            / fixture.name
            / "expected.json"
        )
    return (
        repo_root
        / "compiler/test/CodeGen/SSAVC4/Hardware/Run"
        / fixture.name
        / "expected.json"
    )


def fixture_input_path(fixture: Fixture, repo_root: Path) -> Path:
    if fixture.dialect == "vc4kernel":
        return (
            repo_root
            / "compiler/test/CodeGen/VC4Kernel/Hardware/Run"
            / fixture.name
            / "input.mlir"
        )
    return (
        repo_root
        / "compiler/test/CodeGen/SSAVC4/Hardware/Run"
        / fixture.name
        / "input.mlir"
    )


def audit_fixture(
    fixture: Fixture,
    repo_root: Path,
    work_root: Path,
    vc4_opt: Path,
    vc4_codegen: Path,
    vc4asm: Path | None,
) -> list[str]:
    input_path = fixture_input_path(fixture, repo_root)
    expected_path = fixture_expected_path(fixture, repo_root)
    require_file(input_path)
    require_file(expected_path)

    fixture_dir = work_root / fixture.name
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True)

    source_text = read(input_path)
    if fixture.dialect == "vc4kernel":
        validate_vc4kernel_boundary(input_path)
        verified = fixture_dir / "verified.vc4kernel.mlir"
        ssavc4 = fixture_dir / "lowered.ssavc4.mlir"
        run([str(vc4_opt), str(input_path), "--verify-vc4kernel", "-o", str(verified)])
        validate_vc4kernel_boundary(verified)
        run(
            [
                str(vc4_opt),
                str(verified),
                "--convert-vc4kernel-to-ssavc4",
                "-o",
                str(ssavc4),
            ]
        )
        validate_ssavc4_boundary(ssavc4, lowered_from_vc4kernel=True)
    else:
        validate_ssavc4_boundary(input_path, lowered_from_vc4kernel=False)
        ssavc4 = fixture_dir / "verified.ssavc4.mlir"
        run([str(vc4_opt), str(input_path), "-o", str(ssavc4)])
        validate_ssavc4_boundary(ssavc4, lowered_from_vc4kernel=False)

    vc4 = fixture_dir / "scheduled.vc4.mlir"
    run(
        [
            str(vc4_opt),
            str(ssavc4),
            "--convert-ssavc4-to-vc4",
            "--vc4-verify-emit-contract",
            "--vc4-verify-scheduled-hardware-rules",
            "--vc4-verify-scheduled-adjacent-hazards",
            "--vc4-verify-scheduled-io-spacing",
            "--vc4-verify-scheduled-peripheral-accesses",
            "-o",
            str(vc4),
        ]
    )
    validate_vc4_boundary(vc4)

    bundle = fixture_dir / "bundle"
    run([str(vc4_codegen), str(vc4), "--emit-bundle", str(bundle)])
    for artifact in ("kernel_launch.c", "kernel_launch.h", "manifest.json"):
        require_file(bundle / artifact)
    manifest = load_manifest(bundle / "manifest.json")
    qasms = qasm_paths(bundle, manifest)
    validate_qasm_with_assembler(vc4asm, bundle, qasms)

    ssavc4_text = read(ssavc4)
    vc4_text = read(vc4)
    qasm_text = "\n".join(read(path) for path in qasms)
    findings = [
        "chain: input -> SSAVC4 -> scheduled VC4 -> QASM/C/H/manifest",
        "boundaries: VC4Kernel/SSAVC4/scheduled VC4 dialect separation checked",
    ]
    findings.extend(validate_resources(fixture, bundle, manifest))
    findings.extend(
        validate_feature_markers(
            fixture, repo_root, source_text, ssavc4_text, vc4_text, qasm_text, manifest
        )
    )
    return findings


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--work-dir")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "compiler").is_dir():
        fail(f"repo root does not contain compiler/: {repo_root}")
    vc4_opt = find_tool(repo_root, "vc4-opt")
    vc4_codegen = find_tool(repo_root, "vc4-codegen")
    vc4asm = find_tool(repo_root, "vc4asm")

    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if args.work_dir:
        work_root = Path(args.work_dir).resolve()
        work_root.mkdir(parents=True, exist_ok=True)
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="vc4kernel-artifact-boundary-")
        work_root = Path(temp_ctx.name)

    audited: list[str] = []
    try:
        for fixture in REPRESENTATIVE_FIXTURES:
            findings = audit_fixture(
                fixture, repo_root, work_root, vc4_opt, vc4_codegen, vc4asm
            )
            audited.append(f"{fixture.name} [{fixture.group}]")
            print(
                "AUDITED "
                f"{fixture.name} group={fixture.group} findings="
                + "; ".join(findings)
            )
    finally:
        if temp_ctx is not None and not args.keep:
            temp_ctx.cleanup()

    print(
        "PASS VC4Kernel final artifact boundary audit: "
        f"fixtures={len(audited)} groups={len(REPRESENTATIVE_FIXTURES)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
