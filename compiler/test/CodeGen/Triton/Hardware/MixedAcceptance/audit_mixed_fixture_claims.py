#!/usr/bin/env python3
"""Audit Phase 7 TTIR mixed fixture result-field feature claims."""

import argparse
import json
import re
import sys
from pathlib import Path


CLAIM_KINDS = {"CHECKED_OUTPUT", "CHECKED_AUDIT", "PHASE_GUARD"}
CLAIM_PREFIXES = ("saw_", "no_")


def fail(message):
    print(f"FAIL TTIR mixed fixture claim audit: {message}", file=sys.stderr)
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


def fixture_ttir_text(fixture_dir):
    single = fixture_dir / "input.ttir.mlir"
    if single.exists():
        return single.read_text(encoding="utf-8", errors="replace")
    inputs = sorted((fixture_dir / "inputs").glob("*.ttir.mlir"))
    return "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in inputs)


def audit_known_patterns(repo_root, fixture_name, fixture_claims):
    fixture_dir = repo_root / fixture_claims["path"]
    harness_path = fixture_dir / "candidate" / f"{fixture_name}_candidate_harness.c"
    ttir = fixture_ttir_text(fixture_dir)
    harness = harness_path.read_text()
    claim_names = {claim["claim"] for claim in fixture_claims["claims"]}

    if "saw_real_ttir_input" in claim_names:
        require_marker(ttir, "tt.func", "saw_real_ttir_input TTIR")
        require_marker(ttir, "tt.return", "saw_real_ttir_input TTIR")

    if "saw_real_ttir_snapshot" in claim_names:
        fixture_input = fixture_dir / "input.ttir.mlir"
        phase85_snapshot = (
            repo_root
            / "examples/triton/phase8_5_control_flow/generated/mixed_ttir_cf_loop_if_tail_b16.ttir.mlir"
        )
        phase9_snapshot = (
            repo_root
            / "examples/triton/phase9_multi_axis_launch/generated/mixed_ttir_multi_axis_cf_tail_b16.ttir.mlir"
        )
        phase10_snapshot = (
            repo_root
            / "examples/triton/phase10_mask_memory/generated/mixed_ttir_mask_memory_cf_axes_b16.ttir.mlir"
        )
        phase11_snapshot = (
            repo_root
            / "examples/triton/phase11_strided_ranked_memory/generated/mixed_ttir_strided_memory_axes_mask_cf_b16.ttir.mlir"
        )
        phase12_mixed_reduction_snapshot = (
            repo_root
            / "examples/triton/phase12_reductions/generated/mixed_ttir_reduction_axes_mask_cf_strided_b16.ttir.mlir"
        )
        phase12_i32_reduction_snapshot = (
            repo_root
            / "examples/triton/phase12_reductions/generated/ttir_reduce_sum_i32_b16.ttir.mlir"
        )
        phase13_mixed_gemv_snapshot = (
            repo_root
            / "examples/triton/phase13_gemv_rowwise_dot/generated/mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16.ttir.mlir"
        )
        phase13_partial_gemv_snapshot = (
            repo_root
            / "examples/triton/phase13_gemv_rowwise_dot/generated/ttir_gemv_partial_kblock_f32_b16.ttir.mlir"
        )
        phase14_mixed_storage_snapshot = (
            repo_root
            / "examples/triton/phase14_ml_storage_numeric/generated/mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16.ttir.mlir"
        )
        phase14_f16_store_snapshot = (
            repo_root
            / "examples/triton/phase14_ml_storage_numeric/generated/ttir_f32_compute_store_f16_b16.ttir.mlir"
        )
        phase15_mixed_sfu_softmax_snapshot = (
            repo_root
            / "examples/triton/phase15_sfu_softmax/generated/mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16.ttir.mlir"
        )
        if fixture_name == "mixed_ttir_cf_loop_if_tail_vc4triton":
            if not fixture_input.exists() or not phase85_snapshot.exists():
                fail(f"{fixture_name} snapshot provenance paths are missing")
            if fixture_input.read_bytes() != phase85_snapshot.read_bytes():
                fail(f"{fixture_name} input.ttir.mlir does not match the source-controlled Phase 8.5 mixed snapshot")
        if fixture_name == "mixed_ttir_multi_axis_cf_tail_vc4triton":
            if not fixture_input.exists() or not phase9_snapshot.exists():
                fail(f"{fixture_name} snapshot provenance paths are missing")
            if fixture_input.read_bytes() != phase9_snapshot.read_bytes():
                fail(f"{fixture_name} input.ttir.mlir does not match the source-controlled Phase 9 mixed snapshot")
        if fixture_name == "mixed_ttir_mask_memory_cf_axes_b16_vc4triton":
            if not fixture_input.exists() or not phase10_snapshot.exists():
                fail(f"{fixture_name} snapshot provenance paths are missing")
            if fixture_input.read_bytes() != phase10_snapshot.read_bytes():
                fail(f"{fixture_name} input.ttir.mlir does not match the source-controlled Phase 10 mixed snapshot")
        if fixture_name == "mixed_ttir_strided_memory_axes_mask_cf_b16_vc4triton":
            if not fixture_input.exists() or not phase11_snapshot.exists():
                fail(f"{fixture_name} snapshot provenance paths are missing")
            if fixture_input.read_bytes() != phase11_snapshot.read_bytes():
                fail(f"{fixture_name} input.ttir.mlir does not match the source-controlled Phase 11 mixed snapshot")
        if fixture_name == "mixed_ttir_reduction_axes_mask_cf_strided_b16_vc4triton":
            fixture_inputs = fixture_dir / "inputs"
            expected_inputs = {
                "mixed_ttir_strided_memory_axes_mask_cf_b16.ttir.mlir": phase11_snapshot,
                "mixed_ttir_reduction_axes_mask_cf_strided_b16.ttir.mlir": phase12_mixed_reduction_snapshot,
                "ttir_reduce_sum_i32_b16.ttir.mlir": phase12_i32_reduction_snapshot,
            }
            for input_name, snapshot in expected_inputs.items():
                input_path = fixture_inputs / input_name
                if not input_path.exists() or not snapshot.exists():
                    fail(f"{fixture_name} snapshot provenance paths are missing for {input_name}")
                if input_path.read_bytes() != snapshot.read_bytes():
                    fail(f"{fixture_name} {input_name} does not match {snapshot}")
        if fixture_name == "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton":
            fixture_inputs = fixture_dir / "inputs"
            expected_inputs = {
                "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16.ttir.mlir": phase13_mixed_gemv_snapshot,
                "ttir_gemv_partial_kblock_f32_b16.ttir.mlir": phase13_partial_gemv_snapshot,
            }
            for input_name, snapshot in expected_inputs.items():
                input_path = fixture_inputs / input_name
                if not input_path.exists() or not snapshot.exists():
                    fail(f"{fixture_name} snapshot provenance paths are missing for {input_name}")
                if input_path.read_bytes() != snapshot.read_bytes():
                    fail(f"{fixture_name} {input_name} does not match {snapshot}")
        if fixture_name == "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton":
            fixture_inputs = fixture_dir / "inputs"
            expected_inputs = {
                "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16.ttir.mlir": phase14_mixed_storage_snapshot,
                "ttir_f32_compute_store_f16_b16.ttir.mlir": phase14_f16_store_snapshot,
            }
            for input_name, snapshot in expected_inputs.items():
                input_path = fixture_inputs / input_name
                if not input_path.exists() or not snapshot.exists():
                    fail(f"{fixture_name} snapshot provenance paths are missing for {input_name}")
                if input_path.read_bytes() != snapshot.read_bytes():
                    fail(f"{fixture_name} {input_name} does not match {snapshot}")
        if fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            if not fixture_input.exists() or not phase15_mixed_sfu_softmax_snapshot.exists():
                fail(f"{fixture_name} snapshot provenance paths are missing")
            if fixture_input.read_bytes() != phase15_mixed_sfu_softmax_snapshot.read_bytes():
                fail(f"{fixture_name} input.ttir.mlir does not match the source-controlled Phase 15 mixed SFU/softmax snapshot")

    if "saw_cpp_ttir_importer" in claim_names:
        require_marker(harness, "VC4_CASE_SAW_CPP_TTIR_IMPORTER", "saw_cpp_ttir_importer harness")

    if "saw_value_surface_verification" in claim_names or "saw_value_to_vc4kernel" in claim_names:
        if "runner_audit" not in json.dumps(fixture_claims["claims"]).lower():
            fail(f"{fixture_name} value/vc4kernel claims must cite runner_audit evidence")

    if "saw_value_scf_cf" in claim_names:
        require_marker(harness, "saw_value_scf_cf", "saw_value_scf_cf harness")
        if "--convert-scf-to-cf" not in (repo_root / "compiler/test/CodeGen/Triton/Support/run_ttir_candidate_codegen_test.sh").read_text():
            fail(f"{fixture_name} saw_value_scf_cf requires runner --convert-scf-to-cf")

    if "saw_program_id_axis0" in claim_names:
        require_marker(ttir, "tt.get_program_id", "saw_program_id_axis0 TTIR")

    if "saw_arange_make_range_16" in claim_names:
        require_marker(ttir, "tt.make_range", "saw_arange_make_range_16 TTIR")
        require_marker(ttir, "end = 16", "saw_arange_make_range_16 TTIR")

    if "saw_masked_load_other_zero" in claim_names:
        require_marker(ttir, "tt.load", "saw_masked_load_other_zero TTIR")
        require_marker(ttir, "arith.constant dense<0", "saw_masked_load_other_zero TTIR")
        require_marker(harness, "total_mismatches", "saw_masked_load_other_zero harness")

    if "saw_masked_store_tail" in claim_names:
        require_marker(ttir, "tt.store", "saw_masked_store_tail TTIR")
        if fixture_name == "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton":
            require_marker(harness, "verify_row_sentinels", "saw_masked_store_tail harness")
        else:
            require_marker(harness, "verify_sentinels", "saw_masked_store_tail harness")

    if "saw_tmu_load" in claim_names:
        min_loads = 1 if fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton" else 2
        if ttir.count("tt.load") < min_loads:
            fail(f"{fixture_name} saw_tmu_load requires at least two TTIR loads")
        require_marker(harness, "vc4_m2_copy_htod", "saw_tmu_load harness")

    if "saw_vdw_preserve_store" in claim_names or "saw_sentinel_preserve" in claim_names:
        require_marker(harness, "sentinel_mismatches", "preserve harness")
        if fixture_name == "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton":
            require_marker(harness, "verify_row_sentinels", "preserve harness")
        else:
            require_marker(harness, "verify_sentinels", "preserve harness")

    if "saw_tail_mask_clamp_overlaunch" in claim_names:
        if fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            require_marker(harness, "EXTRA_ROWS", "saw_tail_mask_clamp_overlaunch harness")
            require_marker(harness, "launched_rows", "saw_tail_mask_clamp_overlaunch harness")
        elif fixture_name == "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton":
            require_marker(harness, "EXTRA_ROWS", "saw_tail_mask_clamp_overlaunch harness")
            require_marker(harness, "MAX_K", "saw_tail_mask_clamp_overlaunch harness")
        elif "rounded_waves" not in harness:
            require_marker(harness, "active_elements", "saw_tail_mask_clamp_overlaunch harness")
            if fixture_name == "mixed_ttir_mask_memory_cf_axes_b16_vc4triton":
                require_marker(harness, "grid_coverage", "saw_tail_mask_clamp_overlaunch harness")
            else:
                require_marker(harness, "total_elements", "saw_tail_mask_clamp_overlaunch harness")
        else:
            require_marker(harness, "1000u", "saw_tail_mask_clamp_overlaunch harness")

    if "saw_ttir_elementwise" in claim_names:
        if fixture_name in (
            "mixed_ttir_mask_memory_cf_axes_b16_vc4triton",
            "mixed_ttir_strided_memory_axes_mask_cf_b16_vc4triton",
            "mixed_ttir_reduction_axes_mask_cf_strided_b16_vc4triton",
        ):
            for marker in ("arith.addf", "arith.cmpf", "arith.select"):
                require_marker(ttir, marker, "saw_ttir_elementwise TTIR")
        elif fixture_name in (
            "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton",
            "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton",
        ):
            for marker in ("arith.mulf", "arith.addf"):
                require_marker(ttir, marker, "saw_ttir_elementwise TTIR")
        elif fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            for marker in ("arith.mulf", "arith.subf", "arith.divf", "arith.select"):
                require_marker(ttir, marker, "saw_ttir_elementwise TTIR")
        else:
            for marker in ("arith.addf", "arith.subf", "arith.mulf", "arith.cmpf", "arith.select"):
                require_marker(ttir, marker, "saw_ttir_elementwise TTIR")
        if fixture_name == "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton":
            require_marker(harness, "verify_row_outputs", "saw_ttir_elementwise harness")
        elif fixture_name == "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton":
            require_marker(harness, "verify_mixed_outputs", "saw_ttir_elementwise harness")
        elif fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            require_marker(harness, "verify_outputs", "saw_ttir_elementwise harness")
        else:
            require_marker(harness, "verify_results", "saw_ttir_elementwise harness")

    if "saw_ttir_tail_mask" in claim_names:
        require_marker(ttir, "arith.cmpi slt", "saw_ttir_tail_mask TTIR")
        require_marker(ttir, "tt.load", "saw_ttir_tail_mask TTIR")
        require_marker(ttir, "tt.store", "saw_ttir_tail_mask TTIR")
        require_marker(harness, "verify_sentinels", "saw_ttir_tail_mask harness")

    if "saw_ttir_scf_if" in claim_names:
        require_marker(ttir, "scf.if", "saw_ttir_scf_if TTIR")
        require_marker(harness, "flag", "saw_ttir_scf_if harness")

    if "saw_ttir_scf_for_or_while" in claim_names:
        if "scf.for" not in ttir and "scf.while" not in ttir:
            fail(f"{fixture_name} saw_ttir_scf_for_or_while requires scf.for or scf.while")
        require_marker(harness, "trip_count", "saw_ttir_scf_for_or_while harness")

    if "saw_ttir_cf_control_flow" in claim_names:
        require_marker(ttir, "scf.if", "saw_ttir_cf_control_flow TTIR")
        if "scf.for" not in ttir and "scf.while" not in ttir:
            fail(f"{fixture_name} saw_ttir_cf_control_flow requires loop control flow")
        require_marker(harness, "trip_count", "saw_ttir_cf_control_flow harness")
        require_marker(harness, "flag", "saw_ttir_cf_control_flow harness")

    if "saw_f32_alu" in claim_names:
        if "arith.addf" not in ttir and "arith.mulf" not in ttir:
            fail(f"{fixture_name} saw_f32_alu requires f32 arithmetic in TTIR")
        require_marker(harness, "float", "saw_f32_alu harness")

    if "saw_f32_cmp_select" in claim_names:
        require_marker(ttir, "arith.cmpf", "saw_f32_cmp_select TTIR")
        require_marker(ttir, "arith.select", "saw_f32_cmp_select TTIR")
        require_marker(harness, "threshold", "saw_f32_cmp_select harness")

    if "saw_i32_alu" in claim_names:
        if "arith.addi" not in ttir and "arith.subi" not in ttir:
            fail(f"{fixture_name} saw_i32_alu requires i32 arithmetic in TTIR")
        require_marker(harness, "int32_t", "saw_i32_alu harness")

    if "saw_i32_cmp_select" in claim_names:
        require_marker(ttir, "arith.cmpi", "saw_i32_cmp_select TTIR")
        require_marker(ttir, "arith.select", "saw_i32_cmp_select TTIR")
        require_marker(harness, "tmp > threshold", "saw_i32_cmp_select harness")

    if "saw_nonzero_output_hash" in claim_names:
        require_marker(harness, "output_hash != 0u", "saw_nonzero_output_hash harness")

    if "saw_ttir_multi_axis_launch" in claim_names:
        for marker in ("tt.get_program_id y", "tt.get_program_id z", "tt.get_num_programs y"):
            require_marker(ttir, marker, "saw_ttir_multi_axis_launch TTIR")
        require_marker(harness, "grid_z", "saw_ttir_multi_axis_launch harness")

    if "saw_ttir_multi_axis" in claim_names:
        require_marker(ttir, "tt.get_program_id y", "saw_ttir_multi_axis TTIR")
        if fixture_name == "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton":
            require_marker(harness, "mixed_grid", "saw_ttir_multi_axis harness")
        elif fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            require_marker(harness, "vc4_m2_dim3(tc->rows, 2u, 1u)", "saw_ttir_multi_axis harness")
        else:
            require_marker(harness, "grid_y", "saw_ttir_multi_axis harness")

    if "saw_ttir_program_id_axis1" in claim_names:
        require_marker(ttir, "tt.get_program_id y", "saw_ttir_program_id_axis1 TTIR")

    if "saw_ttir_program_id_axis2" in claim_names:
        require_marker(ttir, "tt.get_program_id z", "saw_ttir_program_id_axis2 TTIR")

    if "saw_ttir_num_programs_axis1" in claim_names:
        require_marker(ttir, "tt.get_num_programs y", "saw_ttir_num_programs_axis1 TTIR")

    if "saw_ttir_num_programs_axis0" in claim_names:
        require_marker(ttir, "tt.get_num_programs x", "saw_ttir_num_programs_axis0 TTIR")

    if "saw_value_multi_axis_launch" in claim_names:
        require_marker(harness, "saw_value_multi_axis_launch", "saw_value_multi_axis_launch harness")

    if "saw_ttir_control_flow" in claim_names:
        require_marker(ttir, "scf.if", "saw_ttir_control_flow TTIR")
        if fixture_name == "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton":
            require_marker(harness, "MIXED_KBLOCKS", "saw_ttir_control_flow harness")
        elif fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            require_marker(harness, "tc->rows, 2u, 1u", "saw_ttir_control_flow harness")
        else:
            require_marker(harness, "flag", "saw_ttir_control_flow harness")

    if "saw_ttir_mask_tail" in claim_names:
        require_marker(ttir, "arith.cmpi slt", "saw_ttir_mask_tail TTIR")
        require_marker(harness, "saw_ttir_mask_tail", "saw_ttir_mask_tail harness")

    if "saw_ttir_mask_full" in claim_names:
        require_marker(harness, "saw_ttir_mask_full", "saw_ttir_mask_full harness")

    if "saw_ttir_mask_empty" in claim_names:
        require_marker(harness, "saw_ttir_mask_empty", "saw_ttir_mask_empty harness")

    if "saw_ttir_compute_mask_select" in claim_names:
        require_marker(ttir, "arith.cmpf", "saw_ttir_compute_mask_select TTIR")
        require_marker(ttir, "arith.select", "saw_ttir_compute_mask_select TTIR")
        require_marker(harness, "saw_select_x_arm", "saw_ttir_compute_mask_select harness")
        require_marker(harness, "saw_select_y_arm", "saw_ttir_compute_mask_select harness")

    if "saw_value_mask_classifier" in claim_names:
        require_marker(harness, "saw_value_mask_classifier", "saw_value_mask_classifier harness")

    if "saw_no_sparse_memory_mask" in claim_names:
        if "sparse" in ttir.lower():
            fail(f"{fixture_name} saw_no_sparse_memory_mask claimed but TTIR contains sparse marker")
        require_marker(harness, "saw_no_sparse_memory_mask", "saw_no_sparse_memory_mask harness")

    if "saw_ttir_row_strided_memory" in claim_names:
        require_marker(ttir, "tt.get_program_id y", "saw_ttir_row_strided_memory TTIR")
        if fixture_name == "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton":
            require_marker(ttir, "arith.muli %row, %LDA", "saw_ttir_row_strided_memory TTIR")
        elif fixture_name == "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton":
            require_marker(ttir, "arith.muli %row", "saw_ttir_row_strided_memory TTIR")
            require_marker(ttir, "!tt.ptr<f16>", "saw_ttir_row_strided_memory TTIR")
        elif fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            require_marker(ttir, "arith.muli %row", "saw_ttir_row_strided_memory TTIR")
            require_marker(ttir, "!tt.ptr<f16>", "saw_ttir_row_strided_memory TTIR")
        else:
            for marker in ("arith.muli %row, %ldx", "arith.muli %row, %ldy", "arith.muli %row, %ldo"):
                require_marker(ttir, marker, "saw_ttir_row_strided_memory TTIR")
        require_marker(harness, "saw_ttir_row_strided_memory", "saw_ttir_row_strided_memory harness")

    if "saw_value_strided_address" in claim_names:
        require_marker(harness, "saw_value_strided_address", "saw_value_strided_address harness")

    if "saw_no_gather_lane_stride" in claim_names:
        if "arith.muli %lanes" in ttir:
            fail(f"{fixture_name} saw_no_gather_lane_stride claimed but TTIR multiplies lanes")
        require_marker(harness, "saw_no_gather_lane_stride", "saw_no_gather_lane_stride harness")

    if "saw_no_hidden_memref_descriptor" in claim_names:
        if "descriptor" in ttir.lower():
            fail(f"{fixture_name} saw_no_hidden_memref_descriptor claimed but TTIR contains descriptor marker")
        require_marker(harness, "saw_no_hidden_memref_descriptor", "saw_no_hidden_memref_descriptor harness")

    if "saw_row_padding_sentinels" in claim_names:
        if fixture_name == "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton":
            require_marker(harness, "verify_row_sentinels", "saw_row_padding_sentinels harness")
        elif fixture_name == "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton":
            require_marker(harness, "verify_mixed_sentinels", "saw_row_padding_sentinels harness")
        elif fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            require_marker(harness, "verify_sentinels", "saw_row_padding_sentinels harness")
        else:
            require_marker(harness, "verify_sentinels", "saw_row_padding_sentinels harness")
        require_marker(harness, "saw_row_padding_sentinels", "saw_row_padding_sentinels harness")

    if "saw_ttir_reduction_i32_add" in claim_names:
        require_marker(ttir, "tt.reduce", "saw_ttir_reduction_i32_add TTIR")
        require_marker(ttir, "arith.addi", "saw_ttir_reduction_i32_add TTIR")
        require_marker(harness, "verify_i32_reduction", "saw_ttir_reduction_i32_add harness")

    if "saw_ttir_reduction_f32_finite_add" in claim_names:
        require_marker(ttir, "tt.reduce", "saw_ttir_reduction_f32_finite_add TTIR")
        require_marker(ttir, "arith.addf", "saw_ttir_reduction_f32_finite_add TTIR")
        if fixture_name == "mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_vc4triton":
            require_marker(harness, "verify_row_outputs", "saw_ttir_reduction_f32_finite_add harness")
        elif fixture_name == "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton":
            require_marker(harness, "verify_mixed_outputs", "saw_ttir_reduction_f32_finite_add harness")
        elif fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            require_marker(harness, "verify_outputs", "saw_ttir_reduction_f32_finite_add harness")
        else:
            require_marker(harness, "verify_f32_reduction", "saw_ttir_reduction_f32_finite_add harness")

    if "saw_ttir_scalar_reduction_store" in claim_names:
        require_marker(ttir, "tt.store", "saw_ttir_scalar_reduction_store TTIR")
        require_marker(harness, "saw_ttir_scalar_reduction_store", "saw_ttir_scalar_reduction_store harness")

    if "saw_f32_finite_tree_policy" in claim_names:
        require_marker(harness, "max_abs_diff", "saw_f32_finite_tree_policy harness")

    if "saw_no_dot_gemv" in claim_names:
        if "tt.dot" in ttir or "gemv" in ttir.lower():
            fail(f"{fixture_name} saw_no_dot_gemv claimed but TTIR contains dot/GEMV marker")
        require_marker(harness, "saw_no_dot_gemv", "saw_no_dot_gemv harness")

    if "saw_ttir_gemv_f32_row_dot" in claim_names:
        for marker in ("arith.mulf", "tt.reduce", "arith.addf", "tt.store"):
            require_marker(ttir, marker, "saw_ttir_gemv_f32_row_dot TTIR")
        if fixture_name == "mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16_vc4triton":
            require_marker(harness, "verify_mixed_outputs", "saw_ttir_gemv_f32_row_dot harness")
        else:
            require_marker(harness, "verify_row_outputs", "saw_ttir_gemv_f32_row_dot harness")

    if "saw_ttir_gemv_partial_kblock" in claim_names:
        require_marker(ttir, "partials", "saw_ttir_gemv_partial_kblock TTIR")
        require_marker(harness, "verify_partial_outputs", "saw_ttir_gemv_partial_kblock harness")

    if "saw_ttir_gemv_rowwise_dot" in claim_names:
        require_marker(ttir, "arith.mulf", "saw_ttir_gemv_rowwise_dot TTIR")
        require_marker(ttir, "tt.reduce", "saw_ttir_gemv_rowwise_dot TTIR")
        require_marker(harness, "saw_ttir_gemv_rowwise_dot", "saw_ttir_gemv_rowwise_dot harness")

    if "saw_value_gemv_rowwise_dot" in claim_names:
        require_marker(harness, "saw_value_gemv_rowwise_dot", "saw_value_gemv_rowwise_dot harness")

    if "saw_no_tl_dot_tt_dot" in claim_names:
        if "tt.dot" in ttir:
            fail(f"{fixture_name} saw_no_tl_dot_tt_dot claimed but TTIR contains tt.dot")
        require_marker(harness, "saw_no_tl_dot_tt_dot", "saw_no_tl_dot_tt_dot harness")

    if "saw_no_vector_contract" in claim_names:
        if "vector.contract" in ttir:
            fail(f"{fixture_name} saw_no_vector_contract claimed but TTIR contains vector.contract")
        require_marker(harness, "saw_no_vector_contract", "saw_no_vector_contract harness")

    if "saw_no_multiblock_k_accumulation" in claim_names:
        if "atomic" in ttir.lower():
            fail(f"{fixture_name} saw_no_multiblock_k_accumulation claimed but TTIR contains atomic marker")
        require_marker(harness, "saw_no_multiblock_k_accumulation", "saw_no_multiblock_k_accumulation harness")

    if "saw_ttir_f16_storage_load" in claim_names:
        require_marker(ttir, "!tt.ptr<f16>", "saw_ttir_f16_storage_load TTIR")
        require_marker(ttir, "arith.extf", "saw_ttir_f16_storage_load TTIR")
        require_marker(harness, "f16_to_f32", "saw_ttir_f16_storage_load harness")

    if "saw_ttir_f16_storage_store" in claim_names:
        require_marker(ttir, "arith.truncf", "saw_ttir_f16_storage_store TTIR")
        require_marker(ttir, "tt.store", "saw_ttir_f16_storage_store TTIR")
        require_marker(harness, "verify_store_outputs", "saw_ttir_f16_storage_store harness")

    if "saw_ttir_f32_compute_after_f16_load" in claim_names:
        require_marker(ttir, "arith.extf", "saw_ttir_f32_compute_after_f16_load TTIR")
        require_marker(ttir, "arith.mulf", "saw_ttir_f32_compute_after_f16_load TTIR")
        if fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            require_marker(harness, "expected_softmax", "saw_ttir_f32_compute_after_f16_load harness")
        else:
            require_marker(harness, "expected_dot", "saw_ttir_f32_compute_after_f16_load harness")

    if "saw_value_f16_storage" in claim_names:
        require_marker(ttir, "!tt.ptr<f16>", "saw_value_f16_storage TTIR")
        require_marker(harness, "saw_value_f16_storage", "saw_value_f16_storage harness")

    if "saw_f16_storage_finite_policy" in claim_names:
        require_marker(ttir, "arith.truncf", "saw_f16_storage_finite_policy TTIR")
        require_marker(harness, "saw_f16_storage_finite_policy", "saw_f16_storage_finite_policy harness")

    if "saw_no_native_f16_arithmetic" in claim_names:
        if re.search(r"arith\.(addf|subf|mulf|divf).*xf16", ttir) or re.search(
            r"arith\.(addf|subf|mulf|divf).*: f16", ttir
        ):
            fail(f"{fixture_name} saw_no_native_f16_arithmetic claimed but TTIR contains native f16 arithmetic")
        require_marker(harness, "saw_no_native_f16_arithmetic", "saw_no_native_f16_arithmetic harness")

    if "saw_no_bf16_fp8" in claim_names:
        if "bf16" in ttir or "fp8" in ttir.lower():
            fail(f"{fixture_name} saw_no_bf16_fp8 claimed but TTIR contains bf16/fp8")
        require_marker(harness, "saw_no_bf16_fp8", "saw_no_bf16_fp8 harness")

    if "saw_no_softmax_sfu" in claim_names:
        if any(marker in ttir.lower() for marker in ("softmax", "exp", "log", "sqrt", "rsqrt")):
            fail(f"{fixture_name} saw_no_softmax_sfu claimed but TTIR contains softmax/SFU marker")
        require_marker(harness, "saw_no_softmax_sfu", "saw_no_softmax_sfu harness")

    if "saw_output_padding_sentinels" in claim_names:
        if fixture_name == "mixed_ttir_sfu_softmax_axes_mask_cf_f16_storage_b16_vc4triton":
            require_marker(harness, "verify_sentinels", "saw_output_padding_sentinels harness")
        else:
            require_marker(harness, "verify_store_sentinels", "saw_output_padding_sentinels harness")
        require_marker(harness, "saw_output_padding_sentinels", "saw_output_padding_sentinels harness")

    if "saw_ttir_approx_sfu_exp" in claim_names:
        require_marker(ttir, "math.exp", "saw_ttir_approx_sfu_exp TTIR")
        require_marker(harness, "natural_exp_ref", "saw_ttir_approx_sfu_exp harness")

    if "saw_ttir_approx_sfu_recip_div" in claim_names:
        require_marker(ttir, "arith.divf", "saw_ttir_approx_sfu_recip_div TTIR")
        require_marker(harness, "expected_softmax", "saw_ttir_approx_sfu_recip_div harness")

    if "saw_ttir_finite_f32_max_reduction" in claim_names:
        require_marker(ttir, "arith.maxnumf", "saw_ttir_finite_f32_max_reduction TTIR")
        require_marker(harness, "maxv", "saw_ttir_finite_f32_max_reduction harness")

    if "saw_ttir_softmax_v0" in claim_names:
        for marker in ("arith.maxnumf", "math.exp", "arith.addf", "arith.divf", "tt.store"):
            require_marker(ttir, marker, "saw_ttir_softmax_v0 TTIR")
        require_marker(harness, "expected_softmax", "saw_ttir_softmax_v0 harness")

    if "saw_softmax_uses_natural_exp" in claim_names:
        require_marker(harness, "natural_exp_ref", "saw_softmax_uses_natural_exp harness")

    if "saw_no_full_attention" in claim_names:
        if any(marker in ttir.lower() for marker in ("attention", "flashattention", "flash_attention")):
            fail(f"{fixture_name} saw_no_full_attention claimed but TTIR contains attention marker")
        require_marker(harness, "saw_no_full_attention", "saw_no_full_attention harness")

    if "saw_no_multiblock_softmax" in claim_names:
        if "atomic" in ttir.lower():
            fail(f"{fixture_name} saw_no_multiblock_softmax claimed but TTIR contains atomic marker")
        require_marker(harness, "saw_no_multiblock_softmax", "saw_no_multiblock_softmax harness")

    if "saw_no_gemm" in claim_names:
        if any(marker in ttir.lower() for marker in ("gemm", "matmul", "tt.dot")):
            fail(f"{fixture_name} saw_no_gemm claimed but TTIR contains GEMM marker")
        require_marker(harness, "saw_no_gemm", "saw_no_gemm harness")

    if "saw_select_x_arm" in claim_names:
        require_marker(harness, "saw_x_arm", "saw_select_x_arm harness")

    if "saw_select_y_arm" in claim_names:
        require_marker(harness, "saw_y_arm", "saw_select_y_arm harness")

    if "saw_scf_if_true" in claim_names:
        require_marker(harness, "saw_flag_true", "saw_scf_if_true harness")

    if "saw_scf_if_false" in claim_names:
        require_marker(harness, "saw_flag_false", "saw_scf_if_false harness")

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
            fail(f"{fixture_name} must list TTIR input and harness source_paths")
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
        "PASS TTIR mixed fixture claim audit: "
        f"fixtures={len(mixed_fixtures)} claims={total_claims} phase_guards={phase_guard_claims}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    args = parser.parse_args()
    validate_claim_manifest(args.repo_root.resolve(), load_json(args.manifest), load_json(args.claims))


if __name__ == "__main__":
    main()
