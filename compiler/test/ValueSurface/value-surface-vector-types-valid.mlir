// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @kernel
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %v_i1 = arith.constant dense<false> : vector<16xi1>
    %v_index = vector.step : vector<16xindex>
    %v_i8 = arith.constant dense<0> : vector<16xi8>
    %v_i16 = arith.constant dense<0> : vector<16xi16>
    %v_i32 = arith.constant dense<0> : vector<16xi32>
    %v_f16 = arith.constant dense<0.000000e+00> : vector<16xf16>
    %v_f32 = arith.constant dense<0.000000e+00> : vector<16xf32>
    %v_non16_i32 = arith.constant dense<0> : vector<7xi32>
    %v_non16_f32 = arith.constant dense<0.000000e+00> : vector<17xf32>
    %v_rank2_i8 = arith.constant dense<0> : vector<2x8xi8>
    %v_rank2_f32 = arith.constant dense<0.000000e+00> : vector<4x4xf32>
    return
  }
}
