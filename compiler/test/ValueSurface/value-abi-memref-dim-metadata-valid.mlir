// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @dim_metadata
  func.func @dim_metadata(
      %m: index {vc4value.arg_name = "m", vc4value.scalar_role = "extent"},
      %n: i32 {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
      %tile: memref<?x?xi32, #vc4value.global> {vc4value.arg_name = "tile",
                                                vc4value.direction = "inout",
                                                vc4value.shape_args = ["m", "n"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: memref.dim
    %dx = memref.dim %x, %c0 : memref<?xf32, #vc4value.global>
    %dm = memref.dim %tile, %c0 : memref<?x?xi32, #vc4value.global>
    %dn = memref.dim %tile, %c1 : memref<?x?xi32, #vc4value.global>
    func.return
  }
}
