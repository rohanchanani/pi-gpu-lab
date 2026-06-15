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
        ):
            require_marker(mlir, marker, "saw_value_multi_axis input")
        has_flattened_grid = (
            "vc4value.num_programs {axis = 0" in mlir
            and "%row_base = arith.muli %pid1, %num0" in mlir
            and "%block_id = arith.addi %row_base, %pid0" in mlir
        )
        has_row_block_grid = (
            "%block = vc4value.program_id {axis = 0" in mlir
            and "%row = vc4value.program_id {axis = 1" in mlir
        )
        if not has_flattened_grid and not has_row_block_grid:
            fail("saw_value_multi_axis input lacks accepted flattened or row/block grid shape")
        has_phase10_grid_oracle = all(
            marker in harness
            for marker in ("grid_x", "grid_y", "block_id / cfg->grid_x", "active_qpus=%d")
        )
        has_phase11_grid_oracle = all(
            marker in harness
            for marker in ("col_blocks(cfg->cols)", "cfg->rows + EXTRA_ROWS", "active_qpus=%d")
        )
        has_phase15_grid_oracle = all(
            marker in harness
            for marker in ("vc4_m2_dim3(1u, tc->rows, 1u)", "active_qpus=%d")
        )
        if not has_phase10_grid_oracle and not has_phase11_grid_oracle and not has_phase15_grid_oracle:
            fail("saw_value_multi_axis harness missing checked grid coordinate oracle")

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
        require_marker(mlir, "cf.cond_br", "saw_value_control_flow input")
        if "arith.cmpi ult" not in mlir and "arith.cmpi eq" not in mlir:
            fail("saw_value_control_flow input lacks checked scalar branch predicate")
        if "cf.br ^loop" in mlir:
            require_marker(mlir, "cf.br ^loop", "saw_value_control_flow input")
        else:
            if "cf.cond_br %inside" in mlir:
                require_marker(mlir, "cf.cond_br %use_i32", "saw_value_control_flow input")
            else:
                require_marker(mlir, "cf.cond_br %is_first", "saw_value_control_flow input")
        if "MIXED_VALUE_SFU_SOFTMAX_AXIS_MASK_CF_F16_CASE" in harness:
            require_marker(harness, "MIXED_VALUE_SFU_SOFTMAX_AXIS_MASK_CF_F16_CASE", "saw_value_control_flow harness")
        else:
            for marker in ("trip", "use_i32_path"):
                require_marker(harness, marker, "saw_value_control_flow harness")
        if (
            "MIXED_VALUE_MASK_MEMORY_AXIS_CF_CASE" not in harness
            and "MIXED_VALUE_STRIDED_RANKED_AXIS_MASK_CF_CASE" not in harness
            and "MIXED_VALUE_GEMV_ROW_DOT_AXIS_MASK_CF_CASE" not in harness
            and "MIXED_VALUE_F16_STORAGE_GEMV_AXIS_MASK_CF_CASE" not in harness
            and "MIXED_VALUE_SFU_SOFTMAX_AXIS_MASK_CF_F16_CASE" not in harness
        ):
            fail("saw_value_control_flow harness missing mixed case result marker")

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
        if "%mask = vector.create_mask %remaining" not in mlir:
            require_marker(mlir, "%mask = vector.create_mask %cols", "saw_value_mask_tail input")
        if "vector.transfer_write %store_i" not in mlir:
            require_marker(mlir, "vector.transfer_write %outh", "saw_value_mask_tail input")
        if (
            "vector.transfer_write %store_f" not in mlir
            and "vector.transfer_write %store_h" not in mlir
            and "vector.transfer_write %outh" not in mlir
        ):
            fail("saw_value_mask_tail input missing checked f32/f16 tail transfer_write")
        for marker in ("verify_tail_results", "verify_sentinels", "sentinel_mismatches"):
            require_marker(harness, marker, "saw_value_mask_tail harness")

    if "saw_value_compute_mask_select" in claim_names:
        for marker in ("arith.cmpi sgt", "arith.cmpf olt", "arith.select"):
            require_marker(mlir, marker, "saw_value_compute_mask_select input")
        for marker in ("cfg->use_i32_path", "cond ? candidate"):
            require_marker(harness, marker, "saw_value_compute_mask_select harness")
        if "expected_tail_f" not in harness and "expected_rank_out_f" not in harness:
            fail("saw_value_compute_mask_select harness missing checked f32 oracle array")

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

    if "saw_value_rank1_flattened_stride" in claim_names:
        for marker in (
            "memref<?xi32, #vc4value.global>",
            "%row_base_idx = arith.muli %pid1, %row_stride",
            "%flat_idx = arith.addi %row_base_idx, %col",
            "vector.transfer_read %flat_i[%flat_idx]",
        ):
            require_marker(mlir, marker, "saw_value_rank1_flattened_stride input")
        for marker in ("r * cfg->lda + c", "flat_i_value", "verify_policy_results"):
            require_marker(harness, marker, "saw_value_rank1_flattened_stride harness")

    if "saw_value_rank2_row_slice" in claim_names:
        for marker in (
            "memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>",
            "vector.transfer_read %rank_i[%pid1, %col]",
            "vector.transfer_write %store_i, %rank_out_i[%pid1, %col]",
        ):
            require_marker(mlir, marker, "saw_value_rank2_row_slice input")
        for marker in ("rank_i_value", "rank_out_i", "verify_tail_results"):
            require_marker(harness, marker, "saw_value_rank2_row_slice harness")

    if "saw_value_stride_args" in claim_names:
        require_marker(mlir, "vc4value.stride_args = [\"row_stride\"]", "saw_value_stride_args input")
        for marker in ("cfg->lda", "MAX_LDA"):
            require_marker(harness, marker, "saw_value_stride_args harness")

    if "saw_value_memref_dim_metadata" in claim_names:
        for marker in ("memref.dim %rank_i", "cf.cond_br %inside"):
            require_marker(mlir, marker, "saw_value_memref_dim_metadata input")
        for marker in ("rows + EXTRA_ROWS", "grid"):
            require_marker(harness, marker, "saw_value_memref_dim_metadata harness")

    if "saw_no_gather_lane_stride" in claim_names:
        for marker in ("vector.gather", "vector.scatter", "strided<[?, ?]", "offs * stride"):
            if marker in mlir:
                fail(f"saw_no_gather_lane_stride input uses forbidden marker {marker}")
        require_marker(mlir, "strided<[?, 1], offset: 0>", "saw_no_gather_lane_stride input")

    if "saw_no_hidden_memref_descriptor" in claim_names:
        for marker in ("memref.extract_strided_metadata", "memref.reinterpret_cast"):
            if marker in mlir:
                fail(f"saw_no_hidden_memref_descriptor input uses forbidden marker {marker}")
        require_marker(mlir, "vc4value.shape_args", "saw_no_hidden_memref_descriptor input")

    if "saw_value_row_strided_memory" in claim_names:
        has_i32_row = (
            "memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>" in mlir
            and "vector.transfer_read %rank_i[%pid1, %col]" in mlir
        )
        has_f32_row = (
            "memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>" in mlir
            and "vector.transfer_read %rank_f[%pid1, %col]" in mlir
        )
        has_f16_row = (
            "memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>" in mlir
            and (
                "vector.transfer_read %rank_f16[%pid1, %col]" in mlir
                or "vector.transfer_read %x[%row, %col]" in mlir
            )
        )
        if not has_i32_row and not has_f32_row and not has_f16_row:
            fail("saw_value_row_strided_memory input lacks accepted row-strided value read")
        if "r * cfg->lda + c" in harness:
            row_markers = ("r * cfg->lda + c", "rank_i_value", "rank_f_value", "row padding")
        else:
            row_markers = ("r * tc->stride + c", "logit_value", "row padding")
        for marker in row_markers:
            require_marker(harness, marker, "saw_value_row_strided_memory harness")

    if "saw_value_reduction_i32_add" in claim_names:
        require_marker(mlir, "vector.reduction <add>, %reduce_i : vector<16xi32> into i32", "saw_value_reduction_i32_add input")
        for marker in ("row_sum_i += expected_tail_i[index]", "verify_row_reductions"):
            require_marker(harness, marker, "saw_value_reduction_i32_add harness")

    if "saw_value_reduction_f32_finite_add" in claim_names:
        for marker in ('vc4value.fp_domain = "finite"', 'vc4value.reduction_policy = "finite_tree"'):
            require_marker(mlir, marker, "saw_value_reduction_f32_finite_add input")
        if (
            "vector.reduction <add>, %reduce_f : vector<16xf32> into f32" not in mlir
            and "vector.reduction <add>, %prod : vector<16xf32> into f32" not in mlir
            and "vector.reduction <add>, %active_e : vector<16xf32> into f32" not in mlir
        ):
            fail("saw_value_reduction_f32_finite_add input lacks accepted f32 add reduction")
        if (
            "row_sum_f += expected_tail_f[index]" not in harness
            and "partial_dot += prod" not in harness
            and "row_sum += got" not in harness
        ):
            fail("saw_value_reduction_f32_finite_add harness missing checked f32 sum oracle")
        if "EPSILON" not in harness:
            require_marker(harness, "SOFTMAX_ABS_TOL", "saw_value_reduction_f32_finite_add harness")
        require_marker(harness, "max_abs_diff", "saw_value_reduction_f32_finite_add harness")

    if "saw_value_scalar_reduction_store" in claim_names:
        has_row_store = (
            "memref.store %row_sum_i, %row_out_i[%pid1]" in mlir
            and "memref.store %row_sum_f, %row_out_f[%pid1]" in mlir
        )
        has_dot_store = "memref.store %dot, %partial_out_f[%partial_index]" in mlir
        if not has_row_store and not has_dot_store:
            fail("saw_value_scalar_reduction_store input lacks accepted scalar reduction store")
        if has_row_store:
            for marker in ("row_out_i", "row_out_f", "verify_row_reductions", "verify_sentinels"):
                require_marker(harness, marker, "saw_value_scalar_reduction_store harness")
        else:
            for marker in ("partial_out_f", "verify_partials", "verify_sentinels"):
                require_marker(harness, marker, "saw_value_scalar_reduction_store harness")

    if "saw_value_gemv_f32_row_dot" in claim_names:
        for marker in (
            "%prod = arith.mulf %rank_tail_f, %x_tail : vector<16xf32>",
            "%dot = vector.reduction <add>, %prod : vector<16xf32> into f32",
            "memref.store %dot, %partial_out_f[%partial_index]",
        ):
            require_marker(mlir, marker, "saw_value_gemv_f32_row_dot input")
        has_f32_inputs = (
            "vector.transfer_read %rank_f[%pid1, %col]" in mlir
            and "vector.transfer_read %x[%col]" in mlir
        )
        has_f16_inputs = (
            "vector.transfer_read %rank_f16[%pid1, %col]" in mlir
            and "vector.transfer_read %x_f16[%col]" in mlir
            and "arith.extf %rank_tail_h" in mlir
            and "arith.extf %x_tail_h" in mlir
        )
        if not has_f32_inputs and not has_f16_inputs:
            fail("saw_value_gemv_f32_row_dot input lacks accepted f32 or f16-to-f32 dot inputs")
        for marker in ("x_value", "partial_dot += prod", "verify_partials"):
            require_marker(harness, marker, "saw_value_gemv_f32_row_dot harness")

    if "saw_value_gemv_partial_kblock" in claim_names:
        for marker in (
            "%partial_row_base = arith.muli %pid1, %num_kblocks : index",
            "%partial_index = arith.addi %partial_row_base, %pid0 : index",
            "memref.store %dot, %partial_out_f[%partial_index]",
        ):
            require_marker(mlir, marker, "saw_value_gemv_partial_kblock input")
        for marker in ("col_blocks(cfg->cols)", "partial_out_f", "verify_partials"):
            require_marker(harness, marker, "saw_value_gemv_partial_kblock harness")

    if "saw_no_tl_dot_tt_dot" in claim_names:
        for marker in ("tt.dot", "tl.dot", "\"tt.dot\""):
            if marker in mlir:
                fail(f"saw_no_tl_dot_tt_dot input uses forbidden marker {marker}")
        require_marker(harness, "saw_no_tl_dot_tt_dot=1", "saw_no_tl_dot_tt_dot harness")

    if "saw_no_vector_contract" in claim_names:
        if "vector.contract" in mlir:
            fail("saw_no_vector_contract input uses vector.contract")
        require_marker(harness, "saw_no_vector_contract=1", "saw_no_vector_contract harness")

    if "saw_no_multiblock_k_accumulation" in claim_names:
        for marker in ("memref.load %partial_out_f", "atomic", "cross_program"):
            if marker in mlir:
                fail(f"saw_no_multiblock_k_accumulation input uses forbidden marker {marker}")
        for marker in ("partial_dot += prod", "saw_no_multiblock_k_accumulation=1"):
            require_marker(harness, marker, "saw_no_multiblock_k_accumulation harness")

    if "saw_f32_finite_tree_policy" in claim_names:
        for marker in ('vc4value.fp_domain = "finite"', 'vc4value.reduction_policy = "finite_tree"'):
            require_marker(mlir, marker, "saw_f32_finite_tree_policy input")
        require_marker(harness, "EPSILON", "saw_f32_finite_tree_policy harness")

    if "saw_value_f16_storage_load" in claim_names:
        common_markers = (
            "memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>",
            "arith.extf",
        )
        for marker in common_markers:
            require_marker(mlir, marker, "saw_value_f16_storage_load input")
        has_gemv_f16_load = (
            "vector.transfer_read %rank_f16[%pid1, %col]" in mlir
            and "vector.transfer_read %x_f16[%col]" in mlir
        )
        has_softmax_f16_load = "vector.transfer_read %x[%row, %col]" in mlir
        if not has_gemv_f16_load and not has_softmax_f16_load:
            fail("saw_value_f16_storage_load input lacks accepted f16 transfer_read")
        if "rank_f16_values" in harness:
            harness_markers = ("rank_f16_values", "x_f16_values", "f16_to_f32")
        else:
            harness_markers = ("x_values", "logit_value", "f16_to_f32")
        for marker in harness_markers:
            require_marker(harness, marker, "saw_value_f16_storage_load harness")

    if "saw_value_f16_storage_store" in claim_names:
        if "arith.truncf %store_f : vector<16xf32> to vector<16xf16>" not in mlir:
            require_marker(mlir, "arith.truncf %outf : vector<16xf32> to vector<16xf16>", "saw_value_f16_storage_store input")
        if "vector.transfer_write %store_h, %tail_out_f16" not in mlir:
            require_marker(mlir, "vector.transfer_write %outh, %y", "saw_value_f16_storage_store input")
        if "tail_out_f16" in harness:
            harness_markers = ("tail_out_f16", "f16_to_f32(tail_out_f16[index])", "SENTINEL_H")
        else:
            harness_markers = ("y_values", "f16_to_f32(y_values[index])", "SENTINEL_H")
        for marker in harness_markers:
            require_marker(harness, marker, "saw_value_f16_storage_store harness")

    if "saw_value_f32_compute_after_f16_load" in claim_names:
        for marker in ("arith.extf", "arith.mulf", "arith.addf", "vector.reduction <add>"):
            if marker not in mlir and marker == "arith.addf":
                continue
            require_marker(mlir, marker, "saw_value_f32_compute_after_f16_load input")
        if "partial_dot += prod" in harness:
            harness_markers = ("rank_f_value", "x_value", "partial_dot += prod")
        else:
            harness_markers = ("logit_value", "expected_softmax", "natural_exp_ref")
        for marker in harness_markers:
            require_marker(harness, marker, "saw_value_f32_compute_after_f16_load harness")

    if "saw_f16_storage_finite_policy" in claim_names:
        require_marker(mlir, 'vc4value.f16_storage_policy = "finite"', "saw_f16_storage_finite_policy input")
        if "EPSILON" not in harness:
            require_marker(harness, "SOFTMAX_ABS_TOL", "saw_f16_storage_finite_policy harness")

    if "saw_no_native_f16_arithmetic" in claim_names:
        for line in mlir.splitlines():
            if ("arith.addf" in line or "arith.mulf" in line) and "vector<16xf16>" in line:
                fail(f"saw_no_native_f16_arithmetic input uses native f16 arithmetic: {line.strip()}")

    if "saw_no_bf16_fp8" in claim_names:
        for marker in ("bf16", "fp8", "f8E", "f8e"):
            if marker in mlir:
                fail(f"saw_no_bf16_fp8 input uses forbidden marker {marker}")

    if "saw_no_softmax_sfu" in claim_names:
        for marker in ("math.", "softmax", "exp", "log", "rsqrt", "recip"):
            if marker in mlir.lower():
                fail(f"saw_no_softmax_sfu input uses forbidden marker {marker}")
        require_marker(harness, "saw_no_softmax_sfu=1", "saw_no_softmax_sfu harness")

    if "saw_value_approx_sfu_exp" in claim_names:
        for marker in ('vc4value.math_policy = "approx_sfu"', "math.exp"):
            require_marker(mlir, marker, "saw_value_approx_sfu_exp input")
        for marker in ("natural_exp_ref", "expected_softmax", "saw_softmax_uses_natural_exp=1"):
            require_marker(harness, marker, "saw_value_approx_sfu_exp harness")

    if "saw_value_approx_sfu_recip_div" in claim_names:
        for marker in ("%inv = arith.divf %one, %denom", "vector.broadcast %inv"):
            require_marker(mlir, marker, "saw_value_approx_sfu_recip_div input")
        for marker in ("row_sum", "expected_softmax", "saw_value_approx_sfu_recip_div=1"):
            require_marker(harness, marker, "saw_value_approx_sfu_recip_div harness")

    if "saw_value_finite_f32_max_reduction" in claim_names:
        for marker in ('vc4value.max_policy = "finite"', "vector.reduction <maxnumf>"):
            require_marker(mlir, marker, "saw_value_finite_f32_max_reduction input")
        for marker in ("maxv", "logit_value", "saw_value_finite_f32_max_reduction=1"):
            require_marker(harness, marker, "saw_value_finite_f32_max_reduction harness")

    if "saw_value_softmax_v0" in claim_names:
        for marker in (
            'vc4value.softmax_v0 = "one_block_active_1_to_16"',
            "vector.reduction <maxnumf>",
            "math.exp",
            "vector.reduction <add>",
            "arith.divf",
            "vector.transfer_write",
        ):
            require_marker(mlir, marker, "saw_value_softmax_v0 input")
        for marker in ("expected_softmax", "ROW_SUM_TOL", "saw_value_softmax_v0=1"):
            require_marker(harness, marker, "saw_value_softmax_v0 harness")

    if "saw_scalar_to_vector_f32_broadcast" in claim_names:
        for marker in ("vector.broadcast %max", "vector.broadcast %inv"):
            require_marker(mlir, marker, "saw_scalar_to_vector_f32_broadcast input")
        require_marker(harness, "saw_scalar_to_vector_f32_broadcast=1", "saw_scalar_to_vector_f32_broadcast harness")

    if "saw_approx_math_policy" in claim_names:
        require_marker(mlir, 'vc4value.math_policy = "approx_sfu"', "saw_approx_math_policy input")
        require_marker(harness, "saw_approx_math_policy=1", "saw_approx_math_policy harness")

    if "saw_softmax_uses_natural_exp" in claim_names:
        require_marker(mlir, "math.exp", "saw_softmax_uses_natural_exp input")
        if "exp2" in harness.lower():
            fail("saw_softmax_uses_natural_exp harness must not use exp2 oracle text")
        for marker in ("natural_exp_ref", "saw_softmax_uses_natural_exp=1"):
            require_marker(harness, marker, "saw_softmax_uses_natural_exp harness")

    if "saw_no_exact_default_math" in claim_names:
        require_marker(mlir, 'vc4value.math_policy = "approx_sfu"', "saw_no_exact_default_math input")
        require_marker(harness, "saw_no_exact_default_math=1", "saw_no_exact_default_math harness")

    if "saw_no_multiblock_softmax" in claim_names:
        for marker in ("atomic", "memref.load %y", "cross_program"):
            if marker in mlir:
                fail(f"saw_no_multiblock_softmax input uses forbidden marker {marker}")
        for marker in ("vc4_m2_dim3(1u, tc->rows, 1u)", "saw_no_multiblock_softmax=1"):
            require_marker(harness, marker, "saw_no_multiblock_softmax harness")

    if "saw_no_full_attention" in claim_names:
        for marker in ("attention", "flashattention", "vector.contract", "tt.dot", "tl.dot"):
            if marker in mlir.lower():
                fail(f"saw_no_full_attention input uses forbidden marker {marker}")
        require_marker(harness, "saw_no_full_attention=1", "saw_no_full_attention harness")

    if "saw_no_dot_gemv" in claim_names:
        lowered = mlir.lower()
        for marker in ("vector.contract", "linalg.", "dot", "gemv", "gemm"):
            if marker in lowered:
                fail(f"saw_no_dot_gemv input uses forbidden marker {marker}")
        for marker in ("saw_no_dot_gemv=1", "row_sum_f += expected_tail_f[index]"):
            require_marker(harness, marker, "saw_no_dot_gemv harness")

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
