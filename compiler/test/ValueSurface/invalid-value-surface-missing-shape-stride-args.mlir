// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  // expected-error @+1 {{public value kernel @missing_rank2_shape_args argument #1 'x': dynamic public memref argument requires vc4value.shape_args}}
  func.func @missing_rank2_shape_args(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %x: memref<?x?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @missing_rank2_stride_args argument #3 'x': vc4value.stride_args length must equal the number of dynamic memref strides}}
  func.func @missing_rank2_stride_args(
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
      %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
      %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"},
      %x: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @nonunit_inner_stride argument #4 'x': rank-2 strided row-slice layout must have static inner stride 1}}
  func.func @nonunit_inner_stride(
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
      %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
      %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"},
      %inner_stride: index {vc4value.arg_name = "inner_stride", vc4value.scalar_role = "stride"},
      %x: memref<?x?xf32, strided<[?, ?], offset: 0>, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride", "inner_stride"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    func.return
  }
}
