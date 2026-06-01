// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 -o /dev/null 2>&1 | FileCheck %s

// CHECK: uses_vpm must equal VPM feature usage or nonzero total VPM rows
ssavc4.module @shared_vpm_invalid_ssavc4 {
  ssavc4.func @shared_vpm_bad_resource() attributes {
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
      uses_vpm = false,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = true,
      uses_vdr = false,
      uses_vdw = false,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
    }
  } {
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %tile = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    ssavc4.vpm.write %row0, %tile {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
