// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @scalar_to_vector_f32_broadcast
  func.func @scalar_to_vector_f32_broadcast()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %s = arith.constant 1.000000e+00 : f32
    // CHECK: vector.broadcast
    %v = vector.broadcast %s : f32 to vector<16xf32>
    return
  }
}
