#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path


RUN_ROOT = Path("compiler/test/CodeGen/Triton/Hardware/Run")
CLAIMS_PATH = RUN_ROOT / "phase15_sfu_softmax_isolation_claims.json"
CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
COMMON_RESULT_FIELDS = {
    "total_mismatches",
    "sentinel_mismatches",
    "launch_failures",
    "active_qpus",
    "lanes",
    "output_hash_nonzero",
    "saw_real_ttir_snapshot",
    "saw_cpp_ttir_importer",
    "saw_ttir_approx_sfu",
    "saw_value_approx_sfu",
    "saw_approx_math_policy",
    "saw_no_exact_default_math",
}
FORBIDDEN_ACCEPTED_TTIR_MARKERS = (
    "tt.dot",
    "tt.make_block_ptr",
    "tt.advance",
    "ttg.",
    "triton_gpu.",
    "nvgpu.",
    "nvvm.",
    "gpu.",
)


def fail(message: str) -> None:
    raise SystemExit(f"phase15 TTIR SFU/softmax isolation claim audit failed: {message}")


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


def audit_fixture(repo: Path, fixture: dict) -> None:
    name = fixture.get("name")
    if not isinstance(name, str) or not name:
        fail("fixture missing name")

    fixture_dir = repo / RUN_ROOT / name
    input_ttir = fixture_dir / "input.ttir.mlir"
    source_rel = fixture.get("source")
    snapshot_rel = fixture.get("snapshot")
    expected_path = fixture_dir / "expected.json"
    harness_path = fixture_dir / "candidate" / f"{name}_candidate_harness.c"

    for path in (fixture_dir, input_ttir, repo / source_rel, repo / snapshot_rel, expected_path, harness_path):
        if not path.exists():
            fail(f"{name}: missing {path}")
    for rel in (source_rel, snapshot_rel):
        if not is_tracked(repo, rel):
            fail(f"{name}: provenance file is not source-controlled: {rel}")
    if input_ttir.read_bytes() != (repo / snapshot_rel).read_bytes():
        fail(f"{name}: input.ttir.mlir is not an exact copy of {snapshot_rel}")

    source_text = (repo / source_rel).read_text(encoding="utf-8")
    require_contains(source_text, "triton.jit", f"{name} Triton source")
    if "tl.dot" in source_text:
        fail(f"{name}: accepted source must not use tl.dot")

    ttir_text = input_ttir.read_text(encoding="utf-8")
    require_contains(ttir_text, "tt.func", f"{name} TTIR")
    for op in fixture.get("expected_ops", []):
        require_contains(ttir_text, op, f"{name} TTIR")
    for forbidden in FORBIDDEN_ACCEPTED_TTIR_MARKERS:
        if forbidden in ttir_text:
            fail(f"{name}: accepted snapshot contains forbidden marker {forbidden}")

    expected = load_json(expected_path)
    if expected.get("name") != name or expected.get("status") != "PASS":
        fail(f"{name}: expected.json has wrong name/status")
    required = expected.get("required")
    if not isinstance(required, dict):
        fail(f"{name}: expected.json missing required object")
    required_fields = COMMON_RESULT_FIELDS | set(fixture.get("expected_result_fields", []))
    missing_fields = sorted(required_fields - set(required))
    if missing_fields:
        fail(f"{name}: expected.json missing required fields {missing_fields}")
    if required.get("active_qpus") != 12 or required.get("lanes") != 16:
        fail(f"{name}: expected.json must require active_qpus=12 and lanes=16")
    for zero_field in ("total_mismatches", "sentinel_mismatches", "launch_failures"):
        if required.get(zero_field) != 0:
            fail(f"{name}: expected.json must require {zero_field}=0")
    for one_field in required_fields - {
        "total_mismatches",
        "sentinel_mismatches",
        "launch_failures",
        "active_qpus",
        "lanes",
    }:
        if required.get(one_field) != 1:
            fail(f"{name}: expected.json must require {one_field}=1")

    harness = harness_path.read_text(encoding="utf-8")
    for token in (
        "VC4_TEST_RESULT",
        "VC4_CASE_SAW_CPP_TTIR_IMPORTER",
        "verify_sentinels",
        "output_hash != 0u",
        "ACTIVE_QPUS 12u",
    ):
        require_contains(harness, token, f"{name} harness")
    for field in required_fields:
        require_contains(harness, field, f"{name} harness result fields")
    if "vc4_emit_ttir.py" in harness or "python" in harness.lower():
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
        "real_ttir_snapshot",
        "cpp_ttir_importer",
        "no_exact_default_math",
    ):
        if required_claim not in claim_names:
            fail(f"{name}: missing claim {required_claim}")


def audit_static_negative(repo: Path, entry: dict) -> None:
    rel = entry.get("path")
    diagnostic = entry.get("diagnostic")
    if not isinstance(rel, str) or not isinstance(diagnostic, str):
        fail("static negative coverage entry missing path/diagnostic")
    path = repo / rel
    if not path.exists():
        fail(f"missing static negative coverage file: {rel}")
    text = path.read_text(encoding="utf-8")
    require_contains(text, diagnostic, rel)
    require_contains(text, "RUN: not ", rel)
    require_contains(text, "FileCheck", rel)


def main(argv: list[str]) -> int:
    repo = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd().resolve()
    claims = load_json(repo / CLAIMS_PATH)
    fixtures = claims.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 5:
        fail("claims file must list exactly five Phase 15 TTIR isolation fixtures")
    for fixture in fixtures:
        audit_fixture(repo, fixture)

    static_negative = claims.get("static_negative_coverage")
    if not isinstance(static_negative, list) or len(static_negative) != 3:
        fail("claims file must list exact/default, scalar-load, and dot static negatives")
    for entry in static_negative:
        audit_static_negative(repo, entry)

    print("TTIR_SFU_SOFTMAX_CLAIM_AUDIT=PASS")
    print(f"fixtures={len(fixtures)}")
    print(f"static_negative={len(static_negative)}")
    print("READY_FOR_TRITON=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
