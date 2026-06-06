// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @make_flags_f32_ssavc4
// CHECK: vc4.qpu.bundle
// CHECK-SAME: cond_add = #vc4.cond<always>
// CHECK-SAME: op_add = #vc4.add_opcode<fsub>
// CHECK-SAME: set_flags
// CHECK: vc4.qpu.bundle {{.*}}cond_add = #vc4.cond<ns>{{.*}}op_add = #vc4.add_opcode<or>
// CHECK-NOT: ssavc4.
ssavc4.module @make_flags_f32_ssavc4 {
  ssavc4.func @make_flags_f32_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "make_flags_f32_ssavc4",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
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
