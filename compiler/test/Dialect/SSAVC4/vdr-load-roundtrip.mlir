// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @vdr_load_roundtrip
// CHECK: %[[ROW:.*]] = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
// CHECK: ssavc4.vdr.load
// CHECK-SAME: %[[ROW]]
// CHECK-SAME: elem_bytes = 4 : i32
// CHECK-SAME: memory_pitch_bytes = 64 : i32
// CHECK-SAME: nrows = 16 : i32
// CHECK-SAME: orientation = "horizontal"
// CHECK-SAME: row_len = 16 : i32
// CHECK-SAME: serialize = "mutex"
// CHECK-SAME: vpitch = 1 : i32
// CHECK-SAME: vpm_base_col = 0 : i32
ssavc4.module @vdr_load_roundtrip {
  ssavc4.func @vdr_load_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 4 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = true,
      uses_vdw = false,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    ssavc4.vdr.load %addr, %row {
      elem_bytes = 4 : i32,
      row_len = 16 : i32,
      nrows = 16 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_base_col = 0 : i32,
      orientation = "horizontal",
      vpitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32
    ssavc4.thread_end
  }
}
