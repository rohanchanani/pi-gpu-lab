// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @cooperative_barrier_invalid {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {public_name = "cooperative_barrier_invalid", code_symbol = "cooperative_barrier_invalid_shader", tail_policy = "tail_safe", uniform_words_per_qpu = 0 : i32, args = [], builtins = []},
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
    // CHECK: uses ssavc4.barrier and requires vc4.resource schedule_mode = "cooperative_block"
    ssavc4.barrier {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}
    ssavc4.thread_end
  }
}
