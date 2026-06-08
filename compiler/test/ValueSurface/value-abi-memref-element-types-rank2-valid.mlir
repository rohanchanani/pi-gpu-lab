// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @rank2_memrefs
  func.func @rank2_memrefs(
      %m: index {vc4value.arg_name = "m", vc4value.scalar_role = "extent"},
      %n: i32 {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %a: memref<?x?xi8, #vc4value.global> {vc4value.arg_name = "a",
                                            vc4value.direction = "in",
                                            vc4value.shape_args = ["m", "n"]},
      %b: memref<?x?xi16, #vc4value.global> {vc4value.arg_name = "b",
                                             vc4value.direction = "in",
                                             vc4value.shape_args = ["m", "n"]},
      %c: memref<?x?xi32, #vc4value.global> {vc4value.arg_name = "c",
                                             vc4value.direction = "inout",
                                             vc4value.shape_args = ["m", "n"]},
      %h: memref<?x?xf16, #vc4value.global> {vc4value.arg_name = "h",
                                             vc4value.direction = "inout",
                                             vc4value.shape_args = ["m", "n"]},
      %f: memref<?x?xf32, #vc4value.global> {vc4value.arg_name = "f",
                                             vc4value.direction = "out",
                                             vc4value.shape_args = ["m", "n"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}
