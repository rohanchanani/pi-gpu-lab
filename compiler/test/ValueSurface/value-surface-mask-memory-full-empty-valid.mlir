// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @full_and_empty_masks
  func.func @full_and_empty_masks(
      %in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
      %out: memref<64xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %zero = arith.constant 0 : i32
    // CHECK: vector.transfer_read
    %full_read = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xi32, #vc4value.global>, vector<16xi32>
    // CHECK: vector.transfer_write
    vector.transfer_write %full_read, %out[%c0] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, #vc4value.global>
    // CHECK: vector.create_mask
    %full_mask = vector.create_mask %c32 : vector<16xi1>
    %empty_mask = vector.create_mask %c0 : vector<16xi1>
    %c16_i = arith.constant 16 : index
    %ones = arith.constant dense<1> : vector<16xi32>
    vector.transfer_write %ones, %out[%c16_i], %full_mask {in_bounds = [true]} : vector<16xi32>, memref<64xi32, #vc4value.global>
    vector.transfer_write %ones, %out[%c16_i], %empty_mask {in_bounds = [true]} : vector<16xi32>, memref<64xi32, #vc4value.global>
    return
  }
}
