// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @tmu_direct_load_ssavc4
// CHECK: vc4.func @tmu_direct_kernel
// CHECK: vc4.qpu.bundle
// CHECK-SAME: op_add = #vc4.add_opcode<or>
// CHECK-SAME: waddr_add = 56 : i32
// CHECK: vc4.qpu.bundle {{.*}}sig = #vc4.qpu_signal<ldtmu0>
// CHECK: vc4.qpu.bundle
// CHECK-SAME: add_a = #vc4.qpu_mux<r4>
// CHECK-SAME: add_b = #vc4.qpu_mux<r4>
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @tmu_direct_load_ssavc4 {
  ssavc4.func @tmu_direct_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "tmu_direct_load_ssavc4",
      code_symbol = "tmu_direct_load_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %tok = ssavc4.tmu.request %addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %value = ssavc4.tmu.read %tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    ssavc4.thread_end
  }
}
