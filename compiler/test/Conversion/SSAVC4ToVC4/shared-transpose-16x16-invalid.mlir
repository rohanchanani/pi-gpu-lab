// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @bad_shared_vpm_resource {
  ssavc4.func @bad_shared_vpm_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 2 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
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
    %seed = ssavc4.load_imm <splat32> {value = 42 : i32} : vector<16xi32>
    // CHECK: sub-32 VPM QPU access requires subword = #ssavc4.vpm_subword<packed> or #ssavc4.vpm_subword<laned>
    ssavc4.vpm.write %row0, %seed {
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w16>,
      subword = #ssavc4.vpm_subword<none>,
      x = 0 : i32,
      stride = 1 : i32,
      lanes = 16 : i32
    } : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
