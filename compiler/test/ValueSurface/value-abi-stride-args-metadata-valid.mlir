// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @stride_args_metadata
  func.func @stride_args_metadata(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %s: i32 {vc4value.arg_name = "s", vc4value.scalar_role = "stride"},
      %x: memref<?xf32, strided<[?]>, #vc4value.global> {vc4value.arg_name = "x",
                                                          vc4value.direction = "in",
                                                          vc4value.shape_args = ["n"],
                                                          vc4value.stride_args = ["s"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}
