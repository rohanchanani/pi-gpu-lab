// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @gemv_row_dot_i32
  func.func @gemv_row_dot_i32(
      %a: memref<?xi32, #vc4value.global> {vc4value.arg_name = "a", vc4value.direction = "in", vc4value.shape_args = ["n"]},
      %x: memref<?xi32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["n"]},
      %y: memref<?xi32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["rows"]},
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.i32_mul_policy = "mul24_safe"} {
    %row = vc4value.program_id {axis = 0 : i32} : index
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0 : i32
    %av = vector.transfer_read %a[%c0], %zero {in_bounds = [true]} : memref<?xi32, #vc4value.global>, vector<16xi32>
    %xv = vector.transfer_read %x[%c0], %zero {in_bounds = [true]} : memref<?xi32, #vc4value.global>, vector<16xi32>
    // CHECK: arith.muli
    %prod = arith.muli %av, %xv : vector<16xi32>
    // CHECK: vector.reduction
    %dot = vector.reduction <add>, %prod : vector<16xi32> into i32
    // CHECK: memref.store
    memref.store %dot, %y[%row] : memref<?xi32, #vc4value.global>
    return
  }
}
