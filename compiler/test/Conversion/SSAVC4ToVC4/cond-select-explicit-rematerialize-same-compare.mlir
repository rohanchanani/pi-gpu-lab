// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @cond_select_explicit_rematerialize_same_compare
// CHECK: vc4.func @kernel
// CHECK: set_flags
// CHECK: cond_add = #vc4.cond<cs>
// CHECK: set_flags
// CHECK: cond_add = #vc4.cond<cs>
// CHECK-NOT: ssavc4.
ssavc4.module @cond_select_explicit_rematerialize_same_compare {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %all = ssavc4.load_imm <splat32> {value = -1 : i32} : vector<16xi32>
    %lanes = ssavc4.element_number : vector<16xi32>
    %threshold = ssavc4.load_imm <splat32> {value = 8 : i32} : vector<16xi32>
    %flags0 = ssavc4.make_flags %lanes, %threshold {kind = #ssavc4.flag_kind<sub>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
    %mask0 = ssavc4.cond_select %flags0, %all, %zero {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %flags1 = ssavc4.make_flags %lanes, %threshold {kind = #ssavc4.flag_kind<sub>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
    %mask1 = ssavc4.cond_select %flags1, %zero, %all {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %combined = ssavc4.alu.add %mask0, %mask1 {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
