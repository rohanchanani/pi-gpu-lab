// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_rank2_missing_shape_args(
    %in: memref<?x?xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %row: index {vc4value.arg_name = "row"},
    %col: index {vc4value.arg_name = "col"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%row, %col], %zero {in_bounds = [true]} : memref<?x?xi32, #vc4value.global>, vector<16xi32>
  return
}

// CHECK: rank-2 row-slice requires explicit vc4value.shape_args metadata
// CHECK: READY_FOR_TRITON remains NO
