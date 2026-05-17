// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @metadata_copy_ssavc4
// CHECK: vc4.func @metadata_copy_kernel
// CHECK: domain = #vc4.execution_domain<qpu>
// CHECK: form = #vc4.function_form<scheduled>
// CHECK: threading = #vc4.threading_mode<single>
// CHECK: vc4.launch_abi =
// CHECK: code_symbol = "metadata_copy_ssavc4_shader"
// CHECK: public_name = "metadata_copy_ssavc4"
// CHECK: vc4.resource =
// CHECK: schedule_mode = "independent_vector"
// CHECK: vc4.qpu.bundle
// CHECK-SAME: sig = #vc4.qpu_signal<thrend>
ssavc4.module @metadata_copy_ssavc4 {
  ssavc4.func @metadata_copy_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "metadata_copy_ssavc4",
      code_symbol = "metadata_copy_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      uses_barrier = false,
      uses_shared_vpm = false,
      require_full_block_residency = false,
      semaphores_per_block = 0 : i32,
      warps_per_block_max = 1 : i32
    }
  } {
    ssavc4.thread_end
  }
}
