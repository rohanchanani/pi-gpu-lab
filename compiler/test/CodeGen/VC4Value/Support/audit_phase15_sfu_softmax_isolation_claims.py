#!/usr/bin/env python3
"""Audit Phase 15 VC4Value SFU/softmax hardware isolation claims."""

import argparse
import json
import sys
from pathlib import Path

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_FIXTURES = {
    "value_sfu_exp_f32_b16_vc4value",
    "value_sfu_recip_div_f32_b16_vc4value",
    "value_reduce_max_f32_b16_vc4value",
    "value_softmax_stable_f32_b16_vc4value",
    "value_sfu_softmax_empty_repeat_vc4value",
    "phase15_static_staged_guards",
}
HARDWARE_FIXTURES = REQUIRED_FIXTURES - {"phase15_static_staged_guards"}
REQUIRED_CLAIMS = {
    "saw_value_approx_sfu_exp",
    "saw_value_approx_sfu_recip_div",
    "saw_value_finite_f32_max_reduction",
    "saw_scalar_to_vector_f32_broadcast",
    "saw_value_softmax_v0",
    "saw_approx_math_policy",
    "sfu_tolerance_policy",
    "exact_default_math_static_reject",
    "generic_division_static_reject",
    "zero_active_softmax_static_reject",
    "multiblock_softmax_static_reject",
    "no_tt_dot_vector_contract_full_attention_claim",
    "saw_repeat_invocation",
    "empty_launch_sentinel_preserve",
    "row_padding_sentinels",
    "no_after_boot_timeout",
}


def fail(message: str) -> None:
    print(f"FAIL Phase 15 SFU/softmax isolation claim audit: {message}", file=sys.stderr)
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
    expected = {
        "invalid-exact-math-no-approx-policy.mlir": "exact/default math requires explicit approximate-SFU policy",
        "invalid-generic-divf-no-approx-policy.mlir": "generic division without approximate reciprocal policy is staged",
        "invalid-softmax-zero-active-without-guard.mlir": "active-count-zero softmax is staged",
        "invalid-multiblock-softmax-staged.mlir": "multiblock softmax is staged",
    }
    path = repo_root / "compiler/test/Conversion/VC4ValueToVC4Kernel" / static_test
    if not path.exists():
        fail(f"missing static reject test {static_test}")
    text = path.read_text()
    require_marker(text, expected[static_test], static_test)
    if static_test.startswith("invalid-") and "RUN:" not in text:
        fail(f"{static_test} missing RUN line")


def validate_claim(repo_root: Path, name: str, claim: dict, required: dict, input_text: str, harness_text: str) -> str:
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
    if required.get("runtime_launches", 0) < 5:
        fail(f"{name} expected.json must require at least five runtime launches")
    require_marker(harness_text, "#define ACTIVE_QPUS 12u", f"{name} harness")
    require_marker(harness_text, "VC4_TEST_RESULT", f"{name} harness")
    require_marker(harness_text, "SENTINEL_BITS", f"{name} harness")

    for forbidden in (
        "tt.", "ttg.", "vc4kernel.", "ssavc4.", "vc4.qpu.",
        "vector.contract", "tt.dot", "tl.dot", "attention", "FlashAttention",
        "arith.fptosi", "arith.fptoui",
    ):
        if forbidden in input_text:
            fail(f"{name} input contains forbidden marker {forbidden}")
    if "vc4value.math_policy = \"approx_sfu\"" in input_text:
        require_marker(input_text, "vc4value.fp_domain", f"{name} approximate policy")
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
    if fixture.get("name") != "phase15_static_staged_guards":
        fail(f"unexpected static fixture {fixture.get('name')!r}")
    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = validate_claim(repo_root, "phase15_static_staged_guards", claim, {}, "", "")
        if claim_name in seen:
            fail(f"phase15_static_staged_guards duplicate claim {claim_name}")
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
        if fixture.get("name") == "phase15_static_staged_guards":
            claims.update(validate_static_fixture(repo_root, fixture))
        else:
            claims.update(validate_hardware_fixture(repo_root, fixture))
    missing = REQUIRED_CLAIMS - claims
    if missing:
        fail(f"missing required claims {sorted(missing)}")

    print(f"PASS Phase 15 SFU/softmax isolation claim audit: fixtures={len(fixtures)} claims={len(claims)}")
    print("VALUE_SFU_SOFTMAX_CLAIM_AUDIT=PASS")


if __name__ == "__main__":
    main()
