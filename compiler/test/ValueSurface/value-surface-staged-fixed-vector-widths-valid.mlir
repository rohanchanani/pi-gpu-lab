// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // Non-16 fixed rank-1 vectors are surface-admissible and staged for later splitting; Phase 5 V1 does not lower them.
  // CHECK-LABEL: func.func @kernel
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    // CHECK: vector<8xf32>
    %v8 = arith.constant dense<0.000000e+00> : vector<8xf32>
    // CHECK: vector<17xf32>
    %v17 = arith.constant dense<0.000000e+00> : vector<17xf32>
    // CHECK: vector<32xf32>
    %v32 = arith.constant dense<0.000000e+00> : vector<32xf32>
    // CHECK: vector<64xi32>
    %v64 = arith.constant dense<0> : vector<64xi32>
    return
  }
}
