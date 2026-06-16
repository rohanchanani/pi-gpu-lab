#!/usr/bin/env python3
"""Audit Phase 17 online softmax hardware isolation claims."""

import argparse
import json
import sys
from pathlib import Path

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_FIXTURES = {
    "value_online_softmax_normalizer_f32_b16_vc4value",
    "value_online_attention_apply_v0_f32_b16_vc4value",
    "value_online_attention_apply_v0_scaled_f32_b16_vc4value",
    "value_online_attention_apply_repeat_vc4value",
    "phase17_static_staged_guards",
}
HARDWARE_FIXTURES = REQUIRED_FIXTURES - {"phase17_static_staged_guards"}
REQUIRED_CLAIMS = {
    "saw_value_online_softmax_state",
    "saw_value_loop_carried_m_l_state",
    "saw_value_online_attention_apply",
    "saw_value_precomputed_scores",
    "saw_value_transposed_v_layout",
    "saw_value_loop_carried_m_l_acc_state",
    "saw_value_k_gt_16",
    "saw_value_weighted_sum_reduction",
    "saw_value_scalar_result_store",
    "saw_scalar_scale_arg",
    "saw_value_attention_apply_scaled_scores",
    "saw_repeat_invocation",
    "saw_k1_attention_apply",
    "saw_k17_attention_apply",
    "saw_k64_attention_apply",
    "saw_softmax_uses_natural_exp",
    "staged_nontransposed_v_gather",
    "staged_scalar_global_load",
    "staged_k_zero_online_softmax",
    "staged_qk_score_generation",
    "staged_dot_contract",
    "staged_full_attention_flashattention",
    "exact_default_math_static_reject",
    "no_after_boot_timeout",
}


def fail(message: str) -> None:
    print(f"FAIL Phase 17 online softmax isolation claim audit: {message}", file=sys.stderr)
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
    if "CHECK:" not in text and "expected-error" not in text:
        fail(f"{static_test} missing checked diagnostic")


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
    for doc_marker in claim.get("doc_markers", []):
        docs = (
            repo_root / "compiler/docs/vc4_vector_triton_phase17_online_softmax_state_fixtures.md"
        ).read_text()
        require_marker(docs, doc_marker, f"{name}:{claim_name} docs")
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
    if required.get("max_k", 0) < 64 and name != "value_online_attention_apply_repeat_vc4value":
        fail(f"{name} expected.json must require max_k=64")
    require_marker(harness_text, "#define ACTIVE_QPUS 12u", f"{name} harness")
    require_marker(harness_text, "VC4_TEST_RESULT", f"{name} harness")
    require_marker(harness_text, "SENTINEL_BITS", f"{name} harness")
    require_marker(input_text, "vc4value.math_policy = \"approx_sfu\"", f"{name} input")
    require_marker(input_text, "vc4value.max_policy = \"finite\"", f"{name} input")
    require_marker(input_text, "vector<16xf32>", f"{name} input")
    for forbidden in (
        "tt.", "ttg.", "vc4kernel.", "ssavc4.", "vc4.qpu.",
        "vector.contract", "tt.dot", "tl.dot", "memref.load",
        "FlashAttention",
    ):
        if forbidden in input_text:
            fail(f"{name} input contains forbidden marker {forbidden}")

    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = validate_claim(repo_root, name, claim, required, input_text, harness_text)
        if claim_name in seen:
            fail(f"{name} duplicate claim {claim_name}")
        seen.add(claim_name)
    return seen


def validate_static_fixture(repo_root: Path, fixture: dict) -> set[str]:
    if fixture.get("name") != "phase17_static_staged_guards":
        fail(f"unexpected static fixture {fixture.get('name')!r}")
    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = validate_claim(repo_root, "phase17_static_staged_guards", claim, {}, "", "")
        if claim_name in seen:
            fail(f"phase17_static_staged_guards duplicate claim {claim_name}")
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
        if fixture.get("name") == "phase17_static_staged_guards":
            claims.update(validate_static_fixture(repo_root, fixture))
        else:
            claims.update(validate_hardware_fixture(repo_root, fixture))
    missing = REQUIRED_CLAIMS - claims
    if missing:
        fail(f"missing required claims {sorted(missing)}")

    print(f"PASS Phase 17 online softmax isolation claim audit: fixtures={len(fixtures)} claims={len(claims)}")
    print("VALUE_ONLINE_SOFTMAX_CLAIM_AUDIT=PASS")


if __name__ == "__main__":
    main()
