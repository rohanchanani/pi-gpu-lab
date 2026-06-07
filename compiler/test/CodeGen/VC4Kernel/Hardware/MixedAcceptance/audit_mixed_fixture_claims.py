#!/usr/bin/env python3
"""Audit VC4Kernel mixed fixture result-field feature claims."""

import argparse
import json
import re
import sys
from pathlib import Path


CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
CLAIM_PREFIXES = ("saw_", "no_", "hidden_")

FEATURE_RESULT_OPS = {
    "vc4kernel.fragment_alu.add",
    "vc4kernel.fragment_alu.mul",
    "vc4kernel.fragment_bitcast",
    "vc4kernel.fragment_cmp",
    "vc4kernel.fragment_pack",
    "vc4kernel.fragment_reduce",
    "vc4kernel.fragment_rotate",
    "vc4kernel.fragment_select",
    "vc4kernel.fragment_sfu",
    "vc4kernel.fragment_unpack",
    "vc4kernel.splat",
    "vc4kernel.tmu_load_fragment",
    "vc4kernel.vpm_read_fragment",
}

TERMINAL_OP_MARKERS = (
    "cf.br",
    "cf.cond_br",
    "vc4kernel.vdw_store",
    "vc4kernel.vpm_write_fragment",
    "ssavc4.vdw.",
    "ssavc4.store",
)


def fail(message):
    print(f"FAIL mixed fixture claim audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def require_string(value, context):
    if not isinstance(value, str) or not value.strip():
        fail(f"{context} must be a non-empty string")


def require_claim_evidence(claim, fixture_name):
    kind = claim.get("claim_kind")
    if kind not in CLAIM_KINDS:
        fail(f"{fixture_name}:{claim.get('claim')} has invalid claim_kind {kind!r}")
    evidence = claim.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        fail(f"{fixture_name}:{claim.get('claim')} lacks evidence")
    if kind in {"CHECKED_OUTPUT", "CHECKED_AUDIT"}:
        if not any(key in evidence for key in ("checked_output", "checked_audit", "checksum", "audit_buffer")):
            fail(f"{fixture_name}:{claim.get('claim')} lacks checked output/audit evidence")
    if kind == "PHASE_GUARD":
        guards = evidence.get("guard_fixtures")
        if not isinstance(guards, list) or not guards:
            fail(f"{fixture_name}:{claim.get('claim')} PHASE_GUARD lacks guard_fixtures")


def claim_fields_from_expected(repo_root, fixture):
    expected_path = repo_root / fixture["path"] / "expected.json"
    expected = load_json(expected_path)
    fields = set()
    for field in fixture["required_result_fields"]:
        if field.startswith(CLAIM_PREFIXES):
            fields.add(field)
    for field in expected.get("required", {}):
        if field.startswith(CLAIM_PREFIXES):
            fields.add(field)
    return fields


def parse_mlir_dataflow(text):
    defs = {}
    uses = {}
    terminal_uses = set()
    for raw_line in text.splitlines():
        line = raw_line.split("//", 1)[0]
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"((?:%[-\w\d]+(?:\s*,\s*)?)+)\s*=\s*([\w.]+)", stripped)
        if match:
            results = re.findall(r"%[-\w\d]+", match.group(1))
            op_name = match.group(2)
            rhs = stripped[match.end(1):]
            operands = set(re.findall(r"%[-\w\d]+", rhs))
            for result in results:
                defs[result] = op_name
                uses[result] = operands
            continue
        if any(marker in stripped for marker in TERMINAL_OP_MARKERS):
            terminal_uses.update(re.findall(r"%[-\w\d]+", stripped))
    live = set(terminal_uses)
    changed = True
    while changed:
        changed = False
        for result, operands in uses.items():
            if result in live:
                before = len(live)
                live.update(operands)
                changed |= len(live) != before
    return defs, live


def audit_dead_feature_results(fixture_name, mlir_text):
    defs, live = parse_mlir_dataflow(mlir_text)
    dead = [
        f"{value}={op_name}"
        for value, op_name in sorted(defs.items())
        if op_name in FEATURE_RESULT_OPS and value not in live
    ]
    if dead:
        fail(
            f"{fixture_name} has feature SSA results not feeding a checked "
            f"store/audit path: {', '.join(dead[:12])}"
        )


def audit_known_patterns(repo_root, fixture_name, fixture_claims):
    input_path = repo_root / fixture_claims["path"] / "input.mlir"
    harness_path = (
        repo_root
        / fixture_claims["path"]
        / "candidate"
        / f"{fixture_name}_candidate_harness.c"
    )
    mlir_text = input_path.read_text()
    harness_text = harness_path.read_text()
    by_claim = {claim["claim"]: claim for claim in fixture_claims["claims"]}

    if fixture_name == "dynamic_vpm_pingpong_coord_selector_loop_vc4kernel":
        if "saw_vpm_qpu_read" in by_claim:
            defs, live = parse_mlir_dataflow(mlir_text)
            read_results = [
                value for value, op_name in defs.items()
                if op_name == "vc4kernel.vpm_read_fragment"
            ]
            if not read_results or any(value not in live for value in read_results):
                fail(
                    "dynamic_vpm_pingpong_coord_selector_loop_vc4kernel claims "
                    "saw_vpm_qpu_read without checked QPU readback dataflow"
                )

    x_claim = by_claim.get("saw_dynamic_vpm_x")
    x_values_zero_only = re.search(r"x_values=0(?!,)", harness_text) is not None
    if x_claim and x_values_zero_only:
        evidence = x_claim.get("evidence", {})
        guards = evidence.get("guard_fixtures", [])
        if x_claim.get("claim_kind") != "PHASE_GUARD" or not guards:
            fail(
                f"{fixture_name} claims saw_dynamic_vpm_x with x_values=0 only "
                "without a PHASE_GUARD nonzero-X claim"
            )
        notes = x_claim.get("notes", "").lower()
        if "x_values=0" not in notes or "nonzero" not in notes:
            fail(f"{fixture_name}:saw_dynamic_vpm_x lacks honest x=0/nonzero-X notes")

    vertical_claim = by_claim.get("saw_vertical_subword")
    if vertical_claim:
        has_vertical_subword = (
            "#vc4kernel.vpm_orientation<vertical>" in mlir_text
            and "#vc4kernel.vpm_subword<packed>" in mlir_text
        )
        if not has_vertical_subword and vertical_claim.get("claim_kind") != "PHASE_GUARD":
            fail(f"{fixture_name} claims saw_vertical_subword without vertical subword MLIR")

    for claim in fixture_claims["claims"]:
        text = json.dumps(claim, sort_keys=True).lower()
        if re.search(r"\b(?:full|all)\s+(?:surface|coverage)\b", text):
            fail(
                f"{fixture_name}:{claim['claim']} claims full/all surface coverage; "
                "enumerate feature claims instead"
            )

    audit_dead_feature_results(fixture_name, mlir_text)


def validate_claim_manifest(repo_root, mixed_manifest, claim_manifest):
    if claim_manifest.get("schema_version") != 1:
        fail("claim manifest schema_version must be 1")
    require_string(claim_manifest.get("suite_name"), "suite_name")
    mixed_fixtures = {fixture["name"]: fixture for fixture in mixed_manifest["fixtures"]}
    claim_fixtures = {fixture.get("name"): fixture for fixture in claim_manifest.get("fixtures", [])}
    if set(claim_fixtures) != set(mixed_fixtures):
        missing = sorted(set(mixed_fixtures) - set(claim_fixtures))
        extra = sorted(set(claim_fixtures) - set(mixed_fixtures))
        fail(f"claim fixtures mismatch missing={missing} extra={extra}")

    total_claims = 0
    phase_guard_claims = 0
    for fixture_name, mixed_fixture in sorted(mixed_fixtures.items()):
        claim_fixture = claim_fixtures[fixture_name]
        if claim_fixture.get("path") != mixed_fixture.get("path"):
            fail(f"{fixture_name} claim path does not match mixed manifest")
        required_claims = claim_fields_from_expected(repo_root, mixed_fixture)
        claims = claim_fixture.get("claims")
        if not isinstance(claims, list) or not claims:
            fail(f"{fixture_name} must have non-empty claims")
        by_claim = {}
        for claim in claims:
            claim_name = claim.get("claim")
            require_string(claim_name, f"{fixture_name} claim")
            if not claim_name.startswith(CLAIM_PREFIXES):
                fail(f"{fixture_name}:{claim_name} is not a result metadata claim")
            if claim_name in by_claim:
                fail(f"{fixture_name} duplicate claim {claim_name}")
            by_claim[claim_name] = claim
            require_claim_evidence(claim, fixture_name)
            source_paths = claim.get("source_paths")
            if not isinstance(source_paths, list) or len(source_paths) < 2:
                fail(f"{fixture_name}:{claim_name} must list input and harness source_paths")
            for path_text in source_paths:
                path = repo_root / path_text
                if not path.is_file():
                    fail(f"{fixture_name}:{claim_name} source path missing: {path_text}")
            if claim["claim_kind"] == "PHASE_GUARD":
                phase_guard_claims += 1
                guards = claim["evidence"]["guard_fixtures"]
                for guard in guards:
                    if guard not in mixed_fixtures:
                        fail(
                            f"{fixture_name}:{claim_name} guard fixture {guard} "
                            "is not required by the same mixed acceptance manifest"
                        )
            total_claims += 1
        if set(by_claim) != required_claims:
            missing = sorted(required_claims - set(by_claim))
            extra = sorted(set(by_claim) - required_claims)
            fail(f"{fixture_name} claim mismatch missing={missing} extra={extra}")
        audit_known_patterns(repo_root, fixture_name, claim_fixture)

    print(
        "PASS VC4Kernel mixed fixture claim audit: "
        f"fixtures={len(mixed_fixtures)} claims={total_claims} "
        f"phase_guards={phase_guard_claims}"
    )


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    mixed_manifest = load_json(Path(args.manifest))
    claim_manifest = load_json(Path(args.claims))
    validate_claim_manifest(repo_root, mixed_manifest, claim_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
