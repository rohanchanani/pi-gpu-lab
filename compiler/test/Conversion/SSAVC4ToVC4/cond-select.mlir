// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @cond_select_ssavc4
// CHECK: vc4.func @cond_select_kernel
// CHECK: vc4.qpu.bundle
// CHECK-SAME: cond_add = #vc4.cond<always>
// CHECK-SAME: op_add = #vc4.add_opcode<sub>
// CHECK-SAME: set_flags
// CHECK-SAME: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: small_imm = 0 : i32
// CHECK: vc4.qpu.bundle
// CHECK-SAME: cond_add = #vc4.cond<always>
// CHECK-SAME: op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.bundle
// CHECK-SAME: cond_add = #vc4.cond<zc>
// CHECK-SAME: op_add = #vc4.add_opcode<or>
// CHECK-NOT: ssavc4.
ssavc4.module @cond_select_ssavc4 {
  ssavc4.func @cond_select_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "cond_select_ssavc4",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    %mask = ssavc4.load_imm <per_elem_u2> {values = array<i32: 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0>} : vector<16xi32>
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %value = ssavc4.load_imm <splat32> {value = -7 : i32} : vector<16xi32>
    %flags = ssavc4.make_flags %mask {kind = #ssavc4.flag_kind<zero_test>} : (vector<16xi32>) -> !ssavc4.flags
    %selected = ssavc4.cond_select %flags, %value, %zero {cond = #vc4.cond<zc>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sink = ssavc4.alu.add %selected, %zero {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
