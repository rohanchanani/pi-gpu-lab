// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @kernel
  func.func @kernel(
      %v_i1: vector<16xi1>,
      %v_index: vector<16xindex>,
      %v_i8: vector<16xi8>,
      %v_i16: vector<16xi16>,
      %v_i32: vector<16xi32>,
      %v_f16: vector<16xf16>,
      %v_f32: vector<16xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    return
  }
}
