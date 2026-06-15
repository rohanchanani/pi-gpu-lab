#!/usr/bin/env python3
"""Audit Phase 15B natural-math hardware reproof claims."""

import argparse
import json
import sys
from pathlib import Path

CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
REQUIRED_FIXTURES = {
    "value_sfu_natural_exp_f32_b16_vc4value",
    "value_sfu_natural_log_f32_b16_vc4value",
    "value_sfu_sqrt_f32_b16_vc4value",
    "value_sfu_rsqrt_f32_b16_vc4value",
    "value_softmax_stable_f32_b16_vc4value",
    "phase15b_static_guards",
}
HARDWARE_FIXTURES = REQUIRED_FIXTURES - {"phase15b_static_guards"}
REQUIRED_CLAIMS = {
    "saw_value_math_exp_natural",
    "saw_target_sfu_exp2_scale_log2e",
    "saw_not_raw_exp2_oracle",
    "saw_approx_math_policy",
    "exp_oracle_is_natural_exp",
    "saw_value_math_log_natural",
    "saw_target_sfu_log2_scale_ln2",
    "saw_not_raw_log2_oracle",
    "saw_positive_finite_domain",
    "log_oracle_is_natural_log",
    "saw_value_math_sqrt",
    "saw_target_sfu_rsqrt_times_x",
    "saw_value_math_rsqrt",
    "saw_target_sfu_rsqrt",
    "saw_not_sqrt_oracle",
    "saw_value_softmax_v0",
    "saw_softmax_uses_natural_exp",
    "softmax_oracle_is_natural_exp",
    "row_padding_sentinels",
    "exact_default_math_static_reject",
    "exp2_log2_target_only_claims_honest",
}


def fail(message: str) -> None:
    print(f"FAIL Phase 15B natural math claim audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def require_marker(text: str, marker: str, context: str) -> None:
    if marker not in text:
        fail(f"{context} missing marker {marker!r}")


def forbid_marker(text: str, marker: str, context: str) -> None:
    if marker in text:
        fail(f"{context} contains forbidden marker {marker!r}")


def validate_static_test(repo_root: Path, static_test: str) -> None:
    path = repo_root / "compiler/test/Conversion/VC4ValueToVC4Kernel" / static_test
    if not path.exists():
        fail(f"missing static test {static_test}")
    text = path.read_text()
    if "RUN:" not in text:
        fail(f"{static_test} missing RUN line")
    if static_test == "approx-sfu-natural-exp-f32-lowers-via-exp2-scale.mlir":
        require_marker(text, "1.442695", static_test)
        require_marker(text, "kind = #vc4kernel.sfu_kind<exp>", static_test)
    elif static_test == "approx-sfu-natural-log-f32-lowers-via-log2-scale.mlir":
        require_marker(text, "0.693147", static_test)
        require_marker(text, "kind = #vc4kernel.sfu_kind<log>", static_test)
    elif static_test == "approx-sfu-sqrt-f32-lowers-via-rsqrt-times-x.mlir":
        require_marker(text, "kind = #vc4kernel.sfu_kind<rsqrt>", static_test)
        require_marker(text, "vc4kernel.fragment_alu.mul", static_test)
    elif static_test == "approx-sfu-rsqrt-f32-lowers.mlir":
        require_marker(text, "math.rsqrt", static_test)
        require_marker(text, "kind = #vc4kernel.sfu_kind<rsqrt>", static_test)
    elif static_test == "softmax-v0-composite-natural-exp-lowers.mlir":
        require_marker(text, "1.442695", static_test)
        require_marker(text, "kind = #vc4kernel.sfu_kind<exp>", static_test)
    elif static_test == "invalid-exact-default-exp-log-sqrt.mlir":
        require_marker(text, "exact/default math requires explicit approximate-SFU policy", static_test)


def validate_doc_marker(repo_root: Path, marker: str) -> None:
    docs = [
        repo_root / "compiler/docs/vc4_vector_triton_phase15_base2_sfu_semantics.md",
        repo_root / "compiler/docs/vc4_vector_triton_phase15_sfu_softmax_fixtures.md",
    ]
    if not any(path.exists() and marker in path.read_text() for path in docs):
        fail(f"missing doc marker {marker!r}")


def validate_claim(
    repo_root: Path,
    name: str,
    claim: dict,
    required: dict,
    input_text: str,
    harness_text: str,
) -> str:
    claim_name = claim.get("claim")
    kind = claim.get("kind")
    if kind not in CLAIM_KINDS:
        fail(f"{name}:{claim_name} invalid kind {kind!r}")
    if "expected_field" in claim and required.get(claim["expected_field"]) != 1:
        fail(f"{name}:{claim_name} expected field {claim['expected_field']} must be 1")
    if "input_marker" in claim:
        require_marker(input_text, claim["input_marker"], f"{name}:{claim_name} input")
    if "oracle_marker" in claim:
        require_marker(harness_text, claim["oracle_marker"], f"{name}:{claim_name} harness")
    if "forbidden_oracle_marker" in claim:
        forbid_marker(harness_text, claim["forbidden_oracle_marker"], f"{name}:{claim_name} harness")
    if "static_test" in claim:
        validate_static_test(repo_root, claim["static_test"])
    if "doc_marker" in claim:
        validate_doc_marker(repo_root, claim["doc_marker"])
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
    require_marker(harness_text, "#define ACTIVE_QPUS 12u", f"{name} harness")
    require_marker(harness_text, "SENTINEL_BITS", f"{name} harness")
    require_marker(harness_text, "VC4_TEST_RESULT", f"{name} harness")
    if "vc4value.math_policy = \"approx_sfu\"" in input_text:
        require_marker(input_text, "vc4value.fp_domain", f"{name} policy")
    for forbidden in (
        "tt.", "ttg.", "vc4kernel.", "ssavc4.", "vc4.qpu.",
        "vector.contract", "tt.dot", "tl.dot", "FlashAttention",
    ):
        forbid_marker(input_text, forbidden, f"{name} input")

    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = validate_claim(repo_root, name, claim, required, input_text, harness_text)
        if claim_name in seen:
            fail(f"{name} duplicate claim {claim_name}")
        seen.add(claim_name)
    return seen


def validate_static_fixture(repo_root: Path, fixture: dict) -> set[str]:
    if fixture.get("name") != "phase15b_static_guards":
        fail(f"unexpected static fixture {fixture.get('name')!r}")
    seen = set()
    for claim in fixture.get("claims", []):
        claim_name = validate_claim(repo_root, "phase15b_static_guards", claim, {}, "", "")
        if claim_name in seen:
            fail(f"phase15b_static_guards duplicate claim {claim_name}")
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
        if fixture.get("name") == "phase15b_static_guards":
            claims.update(validate_static_fixture(repo_root, fixture))
        else:
            claims.update(validate_hardware_fixture(repo_root, fixture))
    missing = REQUIRED_CLAIMS - claims
    if missing:
        fail(f"missing required claims {sorted(missing)}")

    print(f"PASS Phase 15B natural math claim audit: fixtures={len(fixtures)} claims={len(claims)}")
    print("VALUE_NATURAL_MATH_CLAIM_AUDIT=PASS")
    print("VALUE_EXP_NATURAL_HARDWARE=PASS")
    print("VALUE_LOG_NATURAL_HARDWARE=PASS")
    print("VALUE_SQRT_HARDWARE=PASS")
    print("VALUE_RSQRT_HARDWARE=PASS")
    print("VALUE_SOFTMAX_NATURAL_EXP_REPROOF=PASS")
    print("EXP2_LOG2_TARGET_ONLY_CLAIMS_HONEST=YES")
    print("EXP_ORACLE_IS_NATURAL_EXP=YES")
    print("LOG_ORACLE_IS_NATURAL_LOG=YES")
    print("SOFTMAX_ORACLE_IS_NATURAL_EXP=YES")


if __name__ == "__main__":
    main()
