// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @kernel
  func.func @kernel(
      %in: memref<16xf32> {vc4value.arg_name = "in", vc4value.direction = "in"},
      %out: memref<16xf32> {vc4value.arg_name = "out", vc4value.direction = "out"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f32
    // CHECK: vector.transfer_read
    %read = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<16xf32>, vector<16xf32>
    // CHECK: vector.transfer_write
    vector.transfer_write %read, %out[%c0] {in_bounds = [true]} : vector<16xf32>, memref<16xf32>
    return
  }
}
