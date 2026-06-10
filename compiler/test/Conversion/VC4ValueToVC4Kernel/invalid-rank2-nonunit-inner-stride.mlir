// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_rank2_nonunit_inner_stride(
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"},
    %inner_stride: index {vc4value.arg_name = "inner_stride", vc4value.scalar_role = "stride"},
    %in: memref<?x?xi32, strided<[?, ?], offset: 0>, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride", "inner_stride"]})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  return
}

// CHECK: rank-2 strided row-slice layout must have static inner stride 1
