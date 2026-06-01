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
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 0 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 0 : i32,
      uses_tmu = false,
      uses_vpm = false,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = false,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = false,
      requires_semaphore_base_builtin = false
    }
  } {
    ssavc4.thread_end
  }
}
