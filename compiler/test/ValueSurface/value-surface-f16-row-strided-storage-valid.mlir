// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @f16_row_strided_storage
  func.func @f16_row_strided_storage(
      %in: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["row_stride"]},
      %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
      %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
      %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %row = vc4value.program_id {axis = 0 : i32} : index
    %c0 = arith.constant 0 : index
    %mask = vector.create_mask %cols : vector<16xi1>
    %zero_h = arith.constant 0.000000e+00 : f16
    // CHECK: vector.transfer_read
    %loaded = vector.transfer_read %in[%row, %c0], %zero_h, %mask {in_bounds = [true]} : memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf16>
    // CHECK: arith.extf
    %wide = arith.extf %loaded : vector<16xf16> to vector<16xf32>
    %sum = vector.reduction <add>, %wide : vector<16xf32> into f32
    // CHECK: memref.store
    memref.store %sum, %out[%row] : memref<?xf32, #vc4value.global>
    return
  }
}
