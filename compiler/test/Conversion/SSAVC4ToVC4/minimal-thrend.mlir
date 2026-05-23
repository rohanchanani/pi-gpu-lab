// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @minimal_thrend_ssavc4
// CHECK: vc4.func @minimal_thrend_ssavc4_kernel
// CHECK: domain = #vc4.execution_domain<qpu>
// CHECK: form = #vc4.function_form<scheduled>
// CHECK: kernel
// CHECK: vc4.launch_abi =
// CHECK: public_name = "minimal_thrend_ssavc4"
// CHECK: vc4.qpu.bundle
// CHECK-SAME: sig = #vc4.qpu_signal<thrend>
// CHECK: vc4.qpu.bundle
// CHECK: vc4.qpu.bundle
// CHECK-NOT: ssavc4.
ssavc4.module @minimal_thrend_ssavc4 {
  ssavc4.func @minimal_thrend_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "minimal_thrend_ssavc4",
      code_symbol = "minimal_thrend_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    ssavc4.thread_end
  }
}
