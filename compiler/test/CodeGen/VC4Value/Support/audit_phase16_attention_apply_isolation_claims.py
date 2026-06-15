#!/usr/bin/env python3
"""Audit Phase 16 VC4Value attention-apply hardware isolation claims."""

import argparse
import json
import sys
from pathlib import Path

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_FIXTURES = {
    "value_attention_apply_v0_f32_b16_vc4value",
    "value_attention_apply_v0_scaled_f32_b16_vc4value",
    "value_attention_apply_v0_repeat_vc4value",
    "value_attention_apply_v0_f16_storage_vc4value",
    "phase16_static_staged_guards",
}
HARDWARE_FIXTURES = REQUIRED_FIXTURES - {"phase16_static_staged_guards"}
REQUIRED_CLAIMS = {
    "saw_value_attention_apply_v0",
    "saw_value_precomputed_scores",
    "saw_value_transposed_v_layout",
    "saw_value_softmax_v0",
    "saw_value_weighted_sum_reduction",
    "saw_value_scalar_result_store",
    "saw_softmax_uses_natural_exp",
    "saw_value_attention_apply_scaled_scores",
    "saw_scalar_scale_arg",
    "saw_repeat_invocation",
    "saw_k1_attention_apply",
    "saw_k16_attention_apply",
    "saw_value_f16_storage_load",
    "saw_value_f32_compute_after_f16_load",
    "saw_f16_storage_finite_policy",
    "staged_nontransposed_v_gather",
    "staged_scalar_global_load",
    "staged_k_zero_attention_apply",
    "staged_multiblock_attention_apply",
    "staged_full_attention_qk",
    "staged_dot_contract",
    "no_after_boot_timeout",
}


def fail(message: str) -> None:
    print(f"FAIL Phase 16 attention-apply isolation claim audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def require_marker(text: str, marker: str, context: str) -> None:
    if marker not in text:
        fail(f"{context} missing marker {marker!r}")


def validate_static_reject(repo_root: Path, static_test: str) -> None:
    path = repo_root / static_test
    if not path.exists():
        fail(f"missing static staged/reject test {static_test}")
    text = path.read_text()
    if "RUN:" not in text:
        fail(f"{static_test} missing RUN line")
    if "expected-error" not in text and "expected-remark" not in text and "CHECK:" not in text:
        fail(f"{static_test} missing checked diagnostic or FileCheck marker")


def validate_claim(repo_root: Path, name: str, claim: dict, required: dict,
                   input_text: str, harness_text: str) -> str:
    claim_name = claim.get("claim")
    kind = claim.get("kind")
    if kind not in CLAIM_KINDS:
        fail(f"{name}:{claim_name} invalid kind {kind!r}")
    if "expected_field" in claim and required.get(claim["expected_field"]) != 1:
        fail(f"{name}:{claim_name} expected field {claim['expected_field']} must be 1")
    for field in claim.get("expected_zero_fields", []):
        if required.get(field) != 0:
            fail(f"{name}:{claim_name} expected zero field {field} must be 0")
    if "input_marker" in claim:
        require_marker(input_text, claim["input_marker"], f"{name}:{claim_name} input")
    if "forbidden_input_marker" in claim and claim["forbidden_input_marker"] in input_text:
        fail(f"{name}:{claim_name} input contains forbidden marker {claim['forbidden_input_marker']!r}")
    if "oracle_marker" in claim:
        require_marker(harness_text, claim["oracle_marker"], f"{name}:{claim_name} harness")
    for static_test in claim.get("static_reject_tests", []):
        validate_static_reject(repo_root, static_test)
    return claim_name


def validate_hardware_fixture(repo_root: Path, fixture: dict) -> set[str]:
    name = fixture.get("name")
    if name not in HARDWARE_FIXTURES:
        fail(f"unexpected hardware fixture {name!r}")
    fixture_dir = repo_root / fixture["path"]
    input_path = fixture_dir / "input.mlir"
    expected_path = fixture_dir / "expected.json"
    harness_path = fixture_dir / "candidate" / f"{name}_candidate_harness.c"
    for path in (fixture_dir, input_path, expected_path, harness_path):
        if not path.exists():
            fail(f"{name} missing required path {path}")

    input_text = input_path.read_text()
    harness_text = harness_path.read_text()
    expected = load_json(expected_path)
    required = expected.get("required", {})
    if expected.get("status") != "PASS":
        fail(f"{name} expected.json must require PASS")
    for field in ("total_mismatches", "sentinel_mismatches", "launch_failures"):
        if required.get(field) != 0:
            fail(f"{name} expected.json must require {field}=0")
    if required.get("active_qpus") != 12:
        fail(f"{name} expected.json must require active_qpus=12")
    if required.get("lanes") != 16:
        fail(f"{name} expected.json must require lanes=16")
    if required.get("output_hash_nonzero") != 1:
        fail(f"{name} expected.json must require output_hash_nonzero=1")
    if required.get("runtime_launches", 0) < 12:
        fail(f"{name} expected.json must require at least twelve runtime launches")
    require_marker(harness_text, "#define ACTIVE_QPUS 12u", f"{name} harness")
    require_marker(harness_text, "VC4_TEST_RESULT", f"{name} harness")
    if "SENTINEL_BITS" not in harness_text and "SENTINEL_H" not in harness_text:
        fail(f"{name} harness must contain sentinel checks")

    for forbidden in (
        "tt.", "ttg.", "vc4kernel.", "ssavc4.", "vc4.qpu.",
        "vector.contract", "tt.dot", "tl.dot", "FlashAttention",
    ):
        if forbidden in input_text:
            fail(f"{name} input contains forbidden marker {forbidden}")
    if "memref.load" in input_text:
        fail(f"{name} input must not use scalar global loads")
    if "vc4value.math_policy = \"approx_sfu\"" not in input_text:
        fail(f"{name} input missing explicit approximate-SFU policy")
    if "vector<16x" not in input_text:
        fail(f"{name} input must use vector<16xT> fragments")

    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = validate_claim(repo_root, name, claim, required, input_text, harness_text)
        if claim_name in seen:
            fail(f"{name} duplicate claim {claim_name}")
        seen.add(claim_name)
    return seen


def validate_static_fixture(repo_root: Path, fixture: dict) -> set[str]:
    if fixture.get("name") != "phase16_static_staged_guards":
        fail(f"unexpected static fixture {fixture.get('name')!r}")
    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = validate_claim(repo_root, "phase16_static_staged_guards", claim, {}, "", "")
        if claim_name in seen:
            fail(f"phase16_static_staged_guards duplicate claim {claim_name}")
        seen.add(claim_name)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--claims", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    claims_path = Path(args.claims)
    if not claims_path.is_absolute():
        claims_path = repo_root / claims_path
    data = load_json(claims_path)
    if data.get("schema_version") != 1:
        fail("schema_version must be 1")

    fixtures = data.get("fixtures", [])
    fixture_names = {fixture.get("name") for fixture in fixtures}
    if fixture_names != REQUIRED_FIXTURES:
        fail(f"fixture set mismatch got={sorted(fixture_names)}")

    claims = set()
    for fixture in fixtures:
        if fixture.get("name") == "phase16_static_staged_guards":
            claims.update(validate_static_fixture(repo_root, fixture))
        else:
            claims.update(validate_hardware_fixture(repo_root, fixture))
    missing = REQUIRED_CLAIMS - claims
    if missing:
        fail(f"missing required claims {sorted(missing)}")

    print(f"PASS Phase 16 attention-apply isolation claim audit: fixtures={len(fixtures)} claims={len(claims)}")
    print("VALUE_ATTENTION_APPLY_CLAIM_AUDIT=PASS")


if __name__ == "__main__":
    main()
