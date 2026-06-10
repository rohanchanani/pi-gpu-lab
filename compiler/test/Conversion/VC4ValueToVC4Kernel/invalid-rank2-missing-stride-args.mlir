// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_rank2_missing_stride_args(
    %in: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"]},
    %rows: index {vc4value.arg_name = "rows"},
    %cols: index {vc4value.arg_name = "cols"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%c0, %c0], %zero {in_bounds = [true]} : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xi32>
  return
}

// CHECK: rank-2 strided row-slice requires explicit vc4value.stride_args metadata
// CHECK: READY_FOR_TRITON remains NO
