// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @public_scalar_args
  func.func @public_scalar_args(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %stride: i32 {vc4value.arg_name = "stride", vc4value.scalar_role = "stride"},
      %scale: f32 {vc4value.arg_name = "scale", vc4value.scalar_role = "value"},
      %grid: i32 {vc4value.arg_name = "grid", vc4value.scalar_role = "grid_dim"},
      %policy: i32 {vc4value.arg_name = "policy", vc4value.scalar_role = "policy"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}
