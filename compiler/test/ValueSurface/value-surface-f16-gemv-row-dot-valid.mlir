// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @f16_gemv_row_dot
  func.func @f16_gemv_row_dot(
      %a: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["lda"]},
      %x: memref<?xf16, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["cols"]},
      %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
      %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
      %lda: index {vc4value.arg_name = "lda", vc4value.scalar_role = "stride"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %row = vc4value.program_id {axis = 0 : i32} : index
    %c0 = arith.constant 0 : index
    %mask = vector.create_mask %cols : vector<16xi1>
    %zero_h = arith.constant 0.000000e+00 : f16
    // CHECK: vector.transfer_read
    %av_h = vector.transfer_read %a[%row, %c0], %zero_h, %mask {in_bounds = [true]} : memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf16>
    %xv_h = vector.transfer_read %x[%c0], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
    // CHECK: arith.extf
    %av = arith.extf %av_h : vector<16xf16> to vector<16xf32>
    %xv = arith.extf %xv_h : vector<16xf16> to vector<16xf32>
    // CHECK: arith.mulf
    %prod = arith.mulf %av, %xv : vector<16xf32>
    // CHECK: vector.reduction
    %dot = vector.reduction <add>, %prod : vector<16xf32> into f32
    // CHECK: memref.store
    memref.store %dot, %y[%row] : memref<?xf32, #vc4value.global>
    return
  }
}
