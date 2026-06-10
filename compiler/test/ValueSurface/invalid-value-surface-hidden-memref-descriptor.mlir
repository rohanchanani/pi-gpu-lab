// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @hidden_descriptor(
      %x: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride"]},
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
      %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
      %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{hidden memref descriptor extraction is not legal in the VC4 value surface}}
    %base, %offset, %size0, %size1, %stride0, %stride1 = memref.extract_strided_metadata %x : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> -> memref<f32, #vc4value.global>, index, index, index, index, index
    func.return
  }
}
