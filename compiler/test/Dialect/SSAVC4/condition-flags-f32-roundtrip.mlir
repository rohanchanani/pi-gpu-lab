// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @condition_flags_f32_roundtrip
// CHECK: ssavc4.make_flags
// CHECK-SAME: kind = #ssavc4.flag_kind<fsub>
ssavc4.module @condition_flags_f32_roundtrip {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %a = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : vector<16xf32>
    %b = ssavc4.load_imm <splat32> {value = 2.000000e+00 : f32} : vector<16xf32>
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
    %flags = ssavc4.make_flags %a, %b {kind = #ssavc4.flag_kind<fsub>} : (vector<16xf32>, vector<16xf32>) -> !ssavc4.flags
    %selected = ssavc4.cond_select %flags, %one, %zero {cond = #vc4.cond<ns>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sink = ssavc4.alu.add %selected, %zero {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
