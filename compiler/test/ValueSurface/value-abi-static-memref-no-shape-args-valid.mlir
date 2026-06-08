// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @static_memrefs
  func.func @static_memrefs(
      %x: memref<16xf32, #vc4value.global> {vc4value.arg_name = "x",
                                            vc4value.direction = "in"},
      %y: memref<4x16xi32, #vc4value.global> {vc4value.arg_name = "y",
                                              vc4value.direction = "out",
                                              vc4value.shape_args = []})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}
