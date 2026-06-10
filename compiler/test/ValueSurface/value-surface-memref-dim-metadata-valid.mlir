// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @phase11_memref_dim_metadata
  func.func @phase11_memref_dim_metadata(
      %x: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride"]},
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
      %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
      %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: memref.dim
    %m = memref.dim %x, %c0 : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>
    // CHECK: memref.dim
    %n = memref.dim %x, %c1 : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>
    func.return
  }
}
