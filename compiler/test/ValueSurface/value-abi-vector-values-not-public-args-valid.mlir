// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @body_vectors_not_public_args
  func.func @body_vectors_not_public_args(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // Public ABI args are scalar/memref only; body vectors remain governed by
    // the Phase 3.5 value-surface type policy.
    // CHECK: vector<8xf32>
    %v8 = arith.constant dense<0.000000e+00> : vector<8xf32>
    // CHECK: vector<4x16xf32>
    %tile = arith.constant dense<0.000000e+00> : vector<4x16xf32>
    func.return
  }
}
