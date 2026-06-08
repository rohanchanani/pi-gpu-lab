// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @rank1_memrefs
  func.func @rank1_memrefs(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %a: memref<?xi8, #vc4value.global> {vc4value.arg_name = "a",
                                         vc4value.direction = "in",
                                         vc4value.shape_args = ["n"]},
      %b: memref<?xi16, #vc4value.global> {vc4value.arg_name = "b",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
      %c: memref<?xi32, #vc4value.global> {vc4value.arg_name = "c",
                                           vc4value.direction = "inout",
                                           vc4value.shape_args = ["n"]},
      %h: memref<?xf16, #vc4value.global> {vc4value.arg_name = "h",
                                           vc4value.direction = "inout",
                                           vc4value.shape_args = ["n"]},
      %f: memref<?xf32, #vc4value.global> {vc4value.arg_name = "f",
                                           vc4value.direction = "out",
                                           vc4value.shape_args = ["n"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}
