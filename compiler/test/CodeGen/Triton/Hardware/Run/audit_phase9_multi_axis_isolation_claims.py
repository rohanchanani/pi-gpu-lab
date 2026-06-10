#!/usr/bin/env python3
import json
import re
import subprocess
import sys
from pathlib import Path


RUN_ROOT = Path("compiler/test/CodeGen/Triton/Hardware/Run")
CLAIMS_PATH = RUN_ROOT / "phase9_multi_axis_isolation_claims.json"
CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_RESULT_FIELDS = {
    "total_mismatches",
    "sentinel_mismatches",
    "launch_failures",
    "active_qpus",
    "lanes",
    "output_hash",
    "saw_nonzero_output_hash",
    "saw_real_ttir_snapshot",
    "saw_cpp_ttir_importer",
    "saw_ttir_multi_axis_launch",
    "saw_value_multi_axis_launch",
}
FORBIDDEN_TTIR_MARKERS = (
    "arith.sitofp",
    "arith.fptosi",
    "tt.dot",
    "tt.reduce",
    "tt.make_block_ptr",
    "tt.advance",
    "ttg.",
    "triton_gpu.",
    "nvgpu.",
    "nvvm.",
    "gpu.",
)


def fail(message: str) -> None:
    raise SystemExit(f"phase9 multi-axis isolation claim audit failed: {message}")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"{path}: invalid json: {exc}")


def require_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label}: missing {needle!r}")


def is_tracked(repo: Path, rel: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--error-unmatch", rel],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def axis_field(axis: str) -> str:
    return {"x": "axis0", "y": "axis1", "z": "axis2"}[axis]


def audit_fixture(repo: Path, fixture: dict) -> None:
    name = fixture.get("name")
    if not isinstance(name, str) or not name:
        fail("fixture missing name")

    fixture_dir = repo / RUN_ROOT / name
    input_ttir = fixture_dir / "input.ttir.mlir"
    source_rel = fixture.get("source")
    snapshot_rel = fixture.get("snapshot")
    source = repo / source_rel
    snapshot = repo / snapshot_rel
    expected_path = fixture_dir / "expected.json"
    harness_path = fixture_dir / "candidate" / f"{name}_candidate_harness.c"

    for path in (fixture_dir, input_ttir, source, snapshot, expected_path, harness_path):
        if not path.exists():
            fail(f"{name}: missing {path}")
    for rel in (source_rel, snapshot_rel):
        if not is_tracked(repo, rel):
            fail(f"{name}: provenance file is not source-controlled: {rel}")
    if input_ttir.read_bytes() != snapshot.read_bytes():
        fail(f"{name}: input.ttir.mlir is not an exact copy of {snapshot_rel}")

    source_text = source.read_text(encoding="utf-8")
    require_contains(source_text, "triton.jit", f"{name} Triton source")
    if "vc4_emit_ttir.py" in source_text:
        fail(f"{name}: source file must not encode TTIR generation")

    ttir_text = input_ttir.read_text(encoding="utf-8")
    require_contains(ttir_text, "tt.func", f"{name} TTIR")
    require_contains(ttir_text, "tt.make_range", f"{name} TTIR")
    require_contains(ttir_text, "tt.store", f"{name} TTIR")
    for op in fixture.get("expected_ops", []):
        require_contains(ttir_text, op, f"{name} TTIR")
    for axis in fixture.get("program_id_axes", []):
        require_contains(ttir_text, f"tt.get_program_id {axis}", f"{name} TTIR")
    for axis in fixture.get("num_programs_axes", []):
        require_contains(ttir_text, f"tt.get_num_programs {axis}", f"{name} TTIR")
    for forbidden in FORBIDDEN_TTIR_MARKERS:
        if forbidden in ttir_text:
            fail(f"{name}: accepted snapshot contains forbidden marker {forbidden}")
    if re.search(r"tensor<\d+x\d+x", ttir_text):
        fail(f"{name}: accepted snapshot contains rank-2 tensor form")

    expected = load_json(expected_path)
    if expected.get("name") != name or expected.get("status") != "PASS":
        fail(f"{name}: expected.json has wrong name/status")
    required = expected.get("required")
    if not isinstance(required, dict):
        fail(f"{name}: expected.json missing required object")
    missing_fields = sorted(REQUIRED_RESULT_FIELDS - set(required))
    if missing_fields:
        fail(f"{name}: expected.json missing required fields {missing_fields}")
    if required.get("active_qpus") != 12 or required.get("lanes") != 16:
        fail(f"{name}: expected.json must require active_qpus=12 and lanes=16")
    for zero_field in ("total_mismatches", "sentinel_mismatches", "launch_failures"):
        if required.get(zero_field) != 0:
            fail(f"{name}: expected.json must require {zero_field}=0")
    if not isinstance(required.get("output_hash"), int) or required["output_hash"] == 0:
        fail(f"{name}: expected.json must require a nonzero output_hash")
    if required.get("saw_cpp_ttir_importer") != 1:
        fail(f"{name}: expected.json must require the C++ TTIR importer path")
    if required.get("saw_real_ttir_snapshot") != 1:
        fail(f"{name}: expected.json must require a real TTIR snapshot")
    if required.get("saw_value_multi_axis_launch") != 1:
        fail(f"{name}: expected.json must require value multi-axis lowering")
    for axis in fixture.get("program_id_axes", []):
        field = f"saw_ttir_program_id_{axis_field(axis)}"
        if required.get(field) != 1:
            fail(f"{name}: expected.json missing {field}=1")
    for axis in fixture.get("num_programs_axes", []):
        field = f"saw_ttir_num_programs_{axis_field(axis)}"
        if required.get(field) != 1:
            fail(f"{name}: expected.json missing {field}=1")

    harness = harness_path.read_text(encoding="utf-8")
    for token in (
        "VC4_TEST_RESULT",
        "VC4_CASE_SAW_CPP_TTIR_IMPORTER",
        "verify_results",
        "verify_sentinels",
        "output_hash != 0u",
    ):
        require_contains(harness, token, f"{name} harness")
    for field in REQUIRED_RESULT_FIELDS:
        require_contains(harness, field, f"{name} harness result fields")
    if "vc4_emit_ttir.py" in harness or "python" in harness.lower() or "triton" in harness.lower() and "vc4triton" not in name:
        fail(f"{name}: harness must not regenerate TTIR or use Python semantic lowering")

    claim_names = set()
    for claim in fixture.get("claims", []):
        if claim.get("claim_kind") not in CLAIM_KINDS:
            fail(f"{name}: invalid claim kind {claim.get('claim_kind')!r}")
        claim_name = claim.get("claim")
        if not isinstance(claim_name, str) or not claim_name:
            fail(f"{name}: invalid claim entry")
        claim_names.add(claim_name)
    for required_claim in (
        "source_controlled_real_triton_source",
        "source_controlled_real_ttir_snapshot",
        "cpp_ttir_importer_path",
        "value_multi_axis_lowering",
        "hardware_output_oracle",
        "no_ttir_regeneration",
    ):
        if required_claim not in claim_names:
            fail(f"{name}: missing claim {required_claim}")


def main(argv: list[str]) -> int:
    repo = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd().resolve()
    claims = load_json(repo / CLAIMS_PATH)
    fixtures = claims.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 4:
        fail("claims file must list exactly four Phase 9 multi-axis isolation fixtures")
    for fixture in fixtures:
        audit_fixture(repo, fixture)
    print("TTIR_MULTI_AXIS_ISOLATION_CLAIM_AUDIT=PASS")
    print(f"fixtures={len(fixtures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
