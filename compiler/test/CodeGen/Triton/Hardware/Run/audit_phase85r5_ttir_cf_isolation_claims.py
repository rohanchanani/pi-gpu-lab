#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REQUIRED_RESULT_FIELDS = {
    "total_mismatches",
    "sentinel_mismatches",
    "launch_failures",
    "active_qpus",
    "lanes",
    "saw_nonzero_output_hash",
    "saw_real_ttir_snapshot",
    "saw_cpp_ttir_importer",
    "saw_value_scf_cf",
    "saw_vc4value_to_vc4kernel",
    "saw_tail_mask_load_store",
    "saw_cpu_oracle_sentinel",
    "saw_no_arith_sitofp",
}

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}


def fail(message: str) -> None:
    raise SystemExit(f"phase85r5 claim audit failed: {message}")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"{path}: invalid json: {exc}")


def require_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        fail(f"{label}: missing {needle!r}")


def audit_fixture(repo: Path, fixture: dict) -> None:
    name = fixture.get("name")
    if not isinstance(name, str) or not name:
        fail("fixture missing name")

    fixture_dir = repo / "compiler/test/CodeGen/Triton/Hardware/Run" / name
    input_ttir = fixture_dir / "input.ttir.mlir"
    snapshot = repo / fixture["snapshot"]
    expected_path = fixture_dir / "expected.json"
    harness_path = fixture_dir / "candidate" / f"{name}_candidate_harness.c"

    for path in (fixture_dir, input_ttir, snapshot, expected_path, harness_path):
        if not path.exists():
            fail(f"{name}: missing {path}")

    if input_ttir.read_bytes() != snapshot.read_bytes():
        fail(f"{name}: input.ttir.mlir is not an exact copy of {fixture['snapshot']}")

    ttir_text = input_ttir.read_text(encoding="utf-8")
    require_contains(ttir_text, "tt.func", f"{name} TTIR")
    require_contains(ttir_text, "tt.get_program_id", f"{name} TTIR")
    require_contains(ttir_text, "tt.make_range", f"{name} TTIR")
    require_contains(ttir_text, "tt.load", f"{name} TTIR")
    require_contains(ttir_text, "tt.store", f"{name} TTIR")
    if "arith.sitofp" in ttir_text:
        fail(f"{name}: accepted snapshot contains arith.sitofp")
    for forbidden in ("tt.dot", "tt.reduce", "tt.make_block_ptr", "tt.advance",
                      "triton_gpu.", "ttg.", "nvgpu.", "nvvm.", "gpu."):
        if forbidden in ttir_text:
            fail(f"{name}: accepted snapshot contains forbidden op marker {forbidden}")
    for op in fixture.get("expected_ops", []):
        require_contains(ttir_text, op, f"{name} TTIR")

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
    if required.get("total_mismatches") != 0 or required.get("sentinel_mismatches") != 0:
        fail(f"{name}: expected.json must require zero result and sentinel mismatches")
    if required.get("launch_failures") != 0:
        fail(f"{name}: expected.json must require zero launch failures")

    harness = harness_path.read_text(encoding="utf-8")
    for token in (
        "VC4_TEST_RESULT",
        "VC4_CASE_SAW_CPP_TTIR_IMPORTER",
        "verify_results",
        "verify_sentinels",
        "expected_bits",
        "output_hash != 0u",
    ):
        require_contains(harness, token, f"{name} harness")
    for field in REQUIRED_RESULT_FIELDS:
        require_contains(harness, field, f"{name} harness result fields")

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
        "cpp_ttir_importer_path",
        "tail_mask_load_store",
        "cpu_oracle_sentinel",
        "no_arith.sitofp_accepted_path",
    ):
        if required_claim not in claim_names:
            fail(f"{name}: missing claim {required_claim}")


def main(argv: list[str]) -> int:
    repo = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd().resolve()
    claims_path = repo / "compiler/test/CodeGen/Triton/Hardware/Run/phase85r5_ttir_cf_isolation_claims.json"
    claims = load_json(claims_path)
    fixtures = claims.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 4:
        fail("claims file must list exactly four R5 isolation fixtures")
    for fixture in fixtures:
        audit_fixture(repo, fixture)
    print("PHASE85R5_TTIR_CF_ISOLATION_CLAIMS_AUDIT=PASS")
    print(f"fixtures={len(fixtures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
