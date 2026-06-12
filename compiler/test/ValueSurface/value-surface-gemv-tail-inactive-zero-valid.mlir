// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @gemv_tail_inactive_zero
  func.func @gemv_tail_inactive_zero(
      %a: memref<?xf32, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in", vc4value.shape_args = ["n"]},
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["n"]},
      %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %row = vc4value.program_id {axis = 0 : i32} : index
    %c0 = arith.constant 0 : index
    // CHECK: vector.create_mask
    %mask = vector.create_mask %n : vector<16xi1>
    %zero = arith.constant 0.000000e+00 : f32
    // CHECK: vector.transfer_read
    %av = vector.transfer_read %a[%c0], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
    %xv = vector.transfer_read %x[%c0], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
    %prod = arith.mulf %av, %xv : vector<16xf32>
    // CHECK: vector.reduction
    %dot = vector.reduction <add>, %prod : vector<16xf32> into f32
    memref.store %dot, %y[%row] : memref<?xf32, #vc4value.global>
    return
  }
}
