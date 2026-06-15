// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_attention_apply_nontransposed_v_gather(
    %v: memref<?x?xf32, #vc4value.global> {vc4value.arg_name = "v", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.attention_apply_v0 = "precomputed_transposed_v_active_1_to_16",
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %k = vc4value.program_id {axis = 0 : i32} : index
  %d = vc4value.program_id {axis = 1 : i32} : index
  %zero = arith.constant 0.000000e+00 : f32
  %loaded = vector.transfer_read %v[%k, %d], %zero {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true]} : memref<?x?xf32, #vc4value.global>, vector<16xf32>
  return
}

// CHECK: rank-2 row-slice transfer map must project the innermost dimension
// CHECK: column slices, transposes, and gather-like maps are staged
// CHECK: READY_FOR_TRITON remains NO
