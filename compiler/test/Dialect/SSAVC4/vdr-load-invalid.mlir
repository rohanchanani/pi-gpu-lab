// RUN: not vc4-opt %s 2>&1 | FileCheck %s

ssavc4.module @bad_vdr_load {
  ssavc4.func @bad_vdr_load_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      uses_barrier = true,
      uses_shared_vpm = true,
      shared_vpm_bytes = 1024 : i32,
      require_full_block_residency = true,
      semaphores_per_block = 4 : i32,
      warps_per_block_max = 4 : i32
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    // CHECK: supports only 32-bit executable VDR loads; elem_bytes must be 4
    ssavc4.vdr.load %addr {
      elem_bytes = 2 : i32,
      row_len = 16 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 32 : i32,
      vpm_base_row = 0 : i32,
      vpm_base_col = 0 : i32,
      orientation = "horizontal",
      vpitch = 1 : i32,
      serialize = "mutex"
    } : i32
    ssavc4.thread_end
  }
}
