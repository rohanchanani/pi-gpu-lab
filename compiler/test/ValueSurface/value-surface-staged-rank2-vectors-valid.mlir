// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // Rank-2 fixed vectors are surface-admissible and staged for later tile/contract planning; Phase 5 V1 does not lower them.
  // CHECK-LABEL: func.func @kernel
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    // CHECK: vector<4x16xf32>
    %v4x16 = arith.constant dense<0.000000e+00> : vector<4x16xf32>
    // CHECK: vector<16x16xf32>
    %v16x16 = arith.constant dense<0.000000e+00> : vector<16x16xf32>
    // CHECK: vector<4x16xi8>
    %v4x16_i8 = arith.constant dense<0> : vector<4x16xi8>
    return
  }
}
