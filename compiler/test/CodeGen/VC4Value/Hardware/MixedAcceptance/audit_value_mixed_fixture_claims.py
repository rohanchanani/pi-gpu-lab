#!/usr/bin/env python3
"""Audit VC4Value mixed fixture result-field feature claims."""

import argparse
import json
import re
import sys
from pathlib import Path


CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
CLAIM_PREFIXES = ("saw_", "no_")


def fail(message):
    print(f"FAIL VC4Value mixed fixture claim audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {path}: {error}")


def require_string(value, context):
    if not isinstance(value, str) or not value.strip():
        fail(f"{context} must be a non-empty string")


def claim_fields_from_expected(repo_root, fixture):
    expected = load_json(repo_root / fixture["path"] / "expected.json")
    fields = {
        field
        for field in fixture["required_result_fields"]
        if field.startswith(CLAIM_PREFIXES)
    }
    fields.update(
        field
        for field in expected.get("required", {})
        if field.startswith(CLAIM_PREFIXES)
    )
    return fields


def require_marker(text, marker, context):
    if marker not in text:
        fail(f"{context} missing marker {marker}")


def audit_known_patterns(repo_root, fixture_name, fixture_claims):
    fixture_dir = repo_root / fixture_claims["path"]
    input_path = fixture_dir / "input.mlir"
    harness_path = fixture_dir / "candidate" / f"{fixture_name}_candidate_harness.c"
    mlir = input_path.read_text()
    harness = harness_path.read_text()
    claim_names = {claim["claim"] for claim in fixture_claims["claims"]}

    if "saw_value_mixed_elementwise" in claim_names:
        for marker in ("arith.mulf", "arith.addf", "arith.select", "vector.transfer_write"):
            require_marker(mlir, marker, "saw_value_mixed_elementwise input")
        if "arith.cmpf" not in mlir and "arith.cmpi" not in mlir:
            fail("saw_value_mixed_elementwise input lacks a checked compare/select predicate")
        if "tmp > THRESHOLD" in harness:
            for marker in ("fallback_values[index]", "total_mismatches"):
                require_marker(harness, marker, "saw_value_mixed_elementwise harness")
        else:
            for marker in ("expected_f_values[index]", "expected_i_values[index]", "total_mismatches"):
                require_marker(harness, marker, "saw_value_mixed_elementwise harness")

    if "saw_value_tail_mask" in claim_names:
        require_marker(mlir, "vector.create_mask", "saw_value_tail_mask input")
        for marker in ("verify_sentinels", "sentinel_mismatches"):
            require_marker(harness, marker, "saw_value_tail_mask harness")

    if "saw_value_tmu_load" in claim_names:
        if mlir.count("vector.transfer_read") < 1:
            fail("saw_value_tmu_load requires checked transfer_read operations")
        require_marker(harness, "vc4_m2_copy_htod", "saw_value_tmu_load harness")
        require_marker(harness, "total_mismatches", "saw_value_tmu_load harness")

    if "saw_value_vdw_preserve" in claim_names or "saw_value_store_preserve" in claim_names:
        require_marker(mlir, "vector.transfer_write", "store preserve input")
        for marker in ("fill_buffers", "verify_sentinels", "sentinel_mismatches"):
            require_marker(harness, marker, "store preserve harness")

    if "saw_value_f32_cmp_select" in claim_names:
        for marker in ('vc4value.fp_domain = "finite"', "arith.cmpf", "arith.select"):
            require_marker(mlir, marker, "saw_value_f32_cmp_select input")
        if "tmp > THRESHOLD" not in harness and "looped > THRESHOLD" not in harness:
            fail("saw_value_f32_cmp_select harness missing finite threshold oracle")

    if "saw_value_mixed_i32_f32" in claim_names:
        for marker in ("memref<?xi32", "memref<?xf32", "arith.cmpi", "arith.mulf", "arith.addf", "arith.select"):
            require_marker(mlir, marker, "saw_value_mixed_i32_f32 input")
        for marker in ("xi_values[index] > THRESHOLD_I", "candidate", "yf_values[index]"):
            require_marker(harness, marker, "saw_value_mixed_i32_f32 harness")

    if "saw_value_i32_cmp" in claim_names:
        require_marker(mlir, "arith.cmpi sgt", "saw_value_i32_cmp input")
        require_marker(harness, "xi_values[index] > THRESHOLD_I", "saw_value_i32_cmp harness")

    if "saw_value_f32_alu" in claim_names:
        for marker in ("arith.mulf", "arith.addf"):
            require_marker(mlir, marker, "saw_value_f32_alu input")
        if (
            "a * xf_values[index] + yf_values[index]" not in harness
            and "factor * xf_values[index] + yf_values[index]" not in harness
            and "factor * yf_values[index] + xf_values[index]" not in harness
        ):
            fail("saw_value_f32_alu harness missing checked f32 arithmetic oracle")

    if "saw_value_cf_loop" in claim_names:
        for marker in ("cf.br ^loop", "cf.cond_br", "arith.cmpi ult"):
            require_marker(mlir, marker, "saw_value_cf_loop input")
        if "trip_cases" not in harness or "trip" not in harness:
            fail("saw_value_cf_loop harness missing varied trip-count oracle")

    if "saw_value_cf_cond_br" in claim_names:
        require_marker(mlir, "cf.cond_br", "saw_value_cf_cond_br input")
        if "flag_cases" not in harness and "use_select" not in harness and "use_i32_path" not in harness:
            fail("saw_value_cf_cond_br harness missing runtime branch flag")

    if "saw_value_scf_cf" in claim_names:
        for marker in ("cf.br ^loop", "cf.cond_br", "arith.cmpi ult"):
            require_marker(mlir, marker, "saw_value_scf_cf input")
        for marker in ("trip_cases", "flag_cases", "use_i32_path"):
            require_marker(harness, marker, "saw_value_scf_cf harness")

    if "saw_value_multi_axis_launch" in claim_names:
        for marker in (
            "vc4value.grid_rank = 3",
            "vc4value.program_id {axis = 0",
            "vc4value.program_id {axis = 1",
            "vc4value.program_id {axis = 2",
            "vc4value.num_programs {axis = 0",
            "vc4value.num_programs {axis = 1",
            "vc4value.num_programs {axis = 2",
            "%block_id = arith.addi %row_base, %pid0",
        ):
            require_marker(mlir, marker, "saw_value_multi_axis_launch input")
        for marker in ("{3u, 2u, 2u", "{4u, 3u, 2u", "active_qpus=%d"):
            require_marker(harness, marker, "saw_value_multi_axis_launch harness")

    if "saw_value_multi_axis" in claim_names:
        for marker in (
            "vc4value.grid_rank = 2",
            "vc4value.program_id {axis = 0",
            "vc4value.program_id {axis = 1",
            "vc4value.num_programs {axis = 0",
            "%row_base = arith.muli %pid1, %num0",
            "%block_id = arith.addi %row_base, %pid0",
        ):
            require_marker(mlir, marker, "saw_value_multi_axis input")
        for marker in ("grid_x", "grid_y", "block_id / cfg->grid_x", "active_qpus=%d"):
            require_marker(harness, marker, "saw_value_multi_axis harness")

    if "saw_value_program_id_axis1" in claim_names:
        require_marker(mlir, "vc4value.program_id {axis = 1", "saw_value_program_id_axis1 input")
        require_marker(harness, "(block_id / grid->x) % grid->y", "saw_value_program_id_axis1 harness")

    if "saw_value_program_id_axis2" in claim_names:
        require_marker(mlir, "vc4value.program_id {axis = 2", "saw_value_program_id_axis2 input")
        require_marker(harness, "block_id / (grid->x * grid->y)", "saw_value_program_id_axis2 harness")

    if "saw_value_num_programs_axis1" in claim_names:
        require_marker(mlir, "vc4value.num_programs {axis = 1", "saw_value_num_programs_axis1 input")
        require_marker(mlir, "%plane_base = arith.muli %pid2, %num1", "saw_value_num_programs_axis1 input")
        require_marker(harness, "grid->y", "saw_value_num_programs_axis1 harness")

    if "saw_value_vector_block_arg" in claim_names:
        require_marker(mlir, "vector<16x", "saw_value_vector_block_arg input")
        if mlir.count("vector<16x") < 4:
            fail("saw_value_vector_block_arg requires multiple vector block argument sites")
        if "merged vector" in harness:
            fail("saw_value_vector_block_arg cannot be claimed by comments only")

    if "saw_value_scf_while" in claim_names:
        for marker in ("scf.while", "scf.condition", "scf.yield"):
            require_marker(mlir, marker, "saw_value_scf_while input")
        for marker in ("scf_while_acc", "trip"):
            require_marker(harness, marker, "saw_value_scf_while harness")

    if "saw_nested_structured_cf" in claim_names:
        for marker in ("scf.for", "scf.if", "scf.while"):
            require_marker(mlir, marker, "saw_nested_structured_cf input")
        for marker in ("use_alt", "range_acc"):
            require_marker(harness, marker, "saw_nested_structured_cf harness")

    if "saw_tl_range_style_loop" in claim_names:
        for marker in ("scf.for %tile = %start to %end step %step", "iter_args"):
            require_marker(mlir, marker, "saw_tl_range_style_loop input")
        for marker in ("range_acc", "start", "end", "step"):
            require_marker(harness, marker, "saw_tl_range_style_loop harness")

    if "saw_persistent_loop_skeleton" in claim_names:
        for marker in ("%start", "%end", "%step", "iter_args"):
            require_marker(mlir, marker, "saw_persistent_loop_skeleton input")
        for marker in ("step", "range_acc"):
            require_marker(harness, marker, "saw_persistent_loop_skeleton harness")

    if "saw_vector_loop_carried" in claim_names:
        for marker in ("scf.while", "vector<16xi32>", "iter_args"):
            require_marker(mlir, marker, "saw_vector_loop_carried input")
        for marker in ("scf_while_acc", "expected_value"):
            require_marker(harness, marker, "saw_vector_loop_carried harness")

    if "saw_multi_exit_reducible_cf" in claim_names:
        for marker in ("cf.cond_br", "^exit_done", "^exit_early", "^merge"):
            require_marker(mlir, marker, "saw_multi_exit_reducible_cf input")
        for marker in ("multi_exit_acc", "early"):
            require_marker(harness, marker, "saw_multi_exit_reducible_cf harness")

    if "saw_scf_to_cf_boundary" in claim_names:
        for marker in ("scf.while", "scf.for", "scf.if"):
            require_marker(mlir, marker, "saw_scf_to_cf_boundary input")
        runner = (
            repo_root
            / "compiler/test/CodeGen/VC4Value/Support/run_vc4value_candidate_codegen_test.sh"
        ).read_text()
        require_marker(runner, "--convert-scf-to-cf", "saw_scf_to_cf_boundary runner")
        require_marker(runner, "after-scf-to-cf", "saw_scf_to_cf_boundary runner")

    if "saw_value_control_flow" in claim_names:
        for marker in ("cf.br ^loop", "cf.cond_br", "arith.cmpi ult"):
            require_marker(mlir, marker, "saw_value_control_flow input")
        for marker in ("trip", "use_i32_path", "MIXED_VALUE_MASK_MEMORY_AXIS_CF_CASE"):
            require_marker(harness, marker, "saw_value_control_flow harness")

    if "saw_value_mask_full" in claim_names:
        for marker in ("vector.transfer_read %xi[%base], %zero_i32 :", "vector.transfer_write %full_biased"):
            require_marker(mlir, marker, "saw_value_mask_full input")
        for marker in ("verify_full_and_policy", "expected_full", "grid_coverage"):
            require_marker(harness, marker, "saw_value_mask_full harness")

    if "saw_value_mask_empty" in claim_names:
        for marker in ("%empty = vector.create_mask %c0_index", "vector.transfer_write %empty_value"):
            require_marker(mlir, marker, "saw_value_mask_empty input")
        for marker in ("empty_out_i", "SENTINEL_I", "verify_sentinels"):
            require_marker(harness, marker, "saw_value_mask_empty harness")

    if "saw_value_mask_tail" in claim_names:
        for marker in ("%mask = vector.create_mask %remaining", "vector.transfer_write %store_f", "vector.transfer_write %store_i"):
            require_marker(mlir, marker, "saw_value_mask_tail input")
        for marker in ("verify_tail_results", "verify_sentinels", "sentinel_mismatches"):
            require_marker(harness, marker, "saw_value_mask_tail harness")

    if "saw_value_compute_mask_select" in claim_names:
        for marker in ("arith.cmpi sgt", "arith.cmpf olt", "arith.select"):
            require_marker(mlir, marker, "saw_value_compute_mask_select input")
        for marker in ("cfg->use_i32_path", "cond ? candidate", "expected_tail_f"):
            require_marker(harness, marker, "saw_value_compute_mask_select harness")

    if "saw_value_load_inactive_zero" in claim_names:
        for marker in ("vector.transfer_read %xi[%base], %zero_i32, %mask", "vector.transfer_write %tail_xi, %policy_out_i[%base]"):
            require_marker(mlir, marker, "saw_value_load_inactive_zero input")
        for marker in ("expected_policy = logical < cfg->n ? xi_value(logical, cfg) : 0", "policy_out_i"):
            require_marker(harness, marker, "saw_value_load_inactive_zero harness")

    if "saw_value_store_inactive_preserve" in claim_names:
        for marker in ("vector.transfer_write %store_f, %tail_out_f[%base], %mask", "vector.transfer_write %store_i, %tail_out_i[%base], %mask"):
            require_marker(mlir, marker, "saw_value_store_inactive_preserve input")
        for marker in ("tail_out_f[i] = sentinel_f", "tail_out_i[i] = SENTINEL_I", "verify_sentinels"):
            require_marker(harness, marker, "saw_value_store_inactive_preserve harness")

    if "saw_no_sparse_memory_mask" in claim_names:
        for marker in ("%cond_i = arith.cmpi", "%cond_f = arith.cmpf", "arith.select"):
            require_marker(mlir, marker, "saw_no_sparse_memory_mask compute input")
        forbidden_transfer_masks = (
            "vector.transfer_read %xi[%base], %zero_i32, %cond_i",
            "vector.transfer_read %xf[%base], %zero_f, %cond_f",
            "vector.transfer_write %store_f, %tail_out_f[%base], %cond_i",
            "vector.transfer_write %store_f, %tail_out_f[%base], %cond_f",
            "vector.transfer_write %store_i, %tail_out_i[%base], %cond_i",
            "vector.transfer_write %store_i, %tail_out_i[%base], %cond_f",
        )
        for marker in forbidden_transfer_masks:
            if marker in mlir:
                fail(f"saw_no_sparse_memory_mask input uses sparse transfer mask {marker}")

    for claim in fixture_claims["claims"]:
        evidence_text = json.dumps(claim.get("evidence", {}), sort_keys=True).lower()
        if "fixture name" in evidence_text or "status string" in evidence_text or "generated path" in evidence_text:
            fail(f"{fixture_name}:{claim['claim']} uses decorative evidence")


def validate_claim_manifest(repo_root, mixed_manifest, claim_manifest):
    if claim_manifest.get("schema_version") != 1:
        fail("claim manifest schema_version must be 1")
    require_string(claim_manifest.get("suite_name"), "suite_name")

    mixed_fixtures = {fixture["name"]: fixture for fixture in mixed_manifest["fixtures"]}
    claim_fixtures = {fixture.get("name"): fixture for fixture in claim_manifest.get("fixtures", [])}
    if set(claim_fixtures) != set(mixed_fixtures):
        fail(
            "claim fixture set mismatch "
            f"missing={sorted(set(mixed_fixtures) - set(claim_fixtures))} "
            f"extra={sorted(set(claim_fixtures) - set(mixed_fixtures))}"
        )

    total_claims = 0
    phase_guard_claims = 0
    for fixture_name, mixed_fixture in sorted(mixed_fixtures.items()):
        fixture_claims = claim_fixtures[fixture_name]
        if fixture_claims.get("path") != mixed_fixture.get("path"):
            fail(f"{fixture_name} claim path does not match manifest path")
        source_paths = fixture_claims.get("source_paths")
        if not isinstance(source_paths, list) or len(source_paths) < 2:
            fail(f"{fixture_name} must list input and harness source_paths")
        for rel_path in source_paths:
            path = repo_root / rel_path
            if not path.exists():
                fail(f"{fixture_name} source path does not exist: {rel_path}")

        required_claims = claim_fields_from_expected(repo_root, mixed_fixture)
        claims = fixture_claims.get("claims")
        if not isinstance(claims, list) or not claims:
            fail(f"{fixture_name} must have non-empty claims")
        by_claim = {}
        for claim in claims:
            claim_name = claim.get("claim")
            require_string(claim_name, f"{fixture_name} claim")
            if not claim_name.startswith(CLAIM_PREFIXES):
                fail(f"{fixture_name}:{claim_name} is not a saw_/no_ result claim")
            if claim_name in by_claim:
                fail(f"{fixture_name} duplicate claim {claim_name}")
            kind = claim.get("claim_kind")
            if kind not in CLAIM_KINDS:
                fail(f"{fixture_name}:{claim_name} invalid claim_kind {kind!r}")
            evidence = claim.get("evidence")
            if not isinstance(evidence, dict) or not evidence:
                fail(f"{fixture_name}:{claim_name} lacks evidence")
            if kind == "CHECKED_OUTPUT" and "checked_output" not in evidence:
                fail(f"{fixture_name}:{claim_name} lacks checked_output evidence")
            if kind == "CHECKED_AUDIT" and "checked_audit" not in evidence:
                fail(f"{fixture_name}:{claim_name} lacks checked_audit evidence")
            if kind == "PHASE_GUARD":
                phase_guard_claims += 1
                guards = evidence.get("guard_fixtures")
                if not isinstance(guards, list) or not guards:
                    fail(f"{fixture_name}:{claim_name} PHASE_GUARD lacks guard_fixtures")
            by_claim[claim_name] = claim
        if set(by_claim) != required_claims:
            fail(
                f"{fixture_name} claim set mismatch "
                f"missing={sorted(required_claims - set(by_claim))} "
                f"extra={sorted(set(by_claim) - required_claims)}"
            )
        audit_known_patterns(repo_root, fixture_name, fixture_claims)
        total_claims += len(claims)

    print(
        "PASS VC4Value mixed fixture claim audit: "
        f"fixtures={len(mixed_fixtures)} claims={total_claims} "
        f"phase_guards={phase_guard_claims}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    validate_claim_manifest(
        args.repo_root.resolve(),
        load_json(args.manifest),
        load_json(args.claims),
    )


if __name__ == "__main__":
    main()
