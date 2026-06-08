// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @arg_attrs
  func.func @arg_attrs(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %stride: i32 {vc4value.arg_name = "stride", vc4value.scalar_role = "stride"},
      %scale: f32 {vc4value.arg_name = "scale", vc4value.scalar_role = "value"},
      %src: memref<?xf32, #vc4value.global> {vc4value.arg_name = "src",
                                             vc4value.direction = "in",
                                             vc4value.shape_args = ["n"]},
      %dst: memref<?xf32, #vc4value.global> {vc4value.arg_name = "dst",
                                             vc4value.direction = "out",
                                             vc4value.shape_args = ["n"]},
      %acc: memref<?xf32, #vc4value.global> {vc4value.arg_name = "acc",
                                             vc4value.direction = "inout",
                                             vc4value.shape_args = ["n"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}
