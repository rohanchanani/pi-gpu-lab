// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_rank2_transposed_transfer_map(
    %in: memref<?x?xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%c0, %c0], %zero {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true]} : memref<?x?xi32, #vc4value.global>, vector<16xi32>
  return
}

// CHECK: rank-2 row-slice transfer map must project the innermost dimension
// CHECK: column slices, transposes, and gather-like maps are staged
// CHECK: READY_FOR_TRITON remains NO
