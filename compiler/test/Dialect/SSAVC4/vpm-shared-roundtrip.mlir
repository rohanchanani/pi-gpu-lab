// RUN: vc4-opt %s | FileCheck %s

// CHECK: ssavc4.vpm.write
// CHECK-SAME: lanes = 16 : i32
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK-SAME: stride = 1 : i32
// CHECK-SAME: subword = #ssavc4.vpm_subword<none>
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w32>
// CHECK-SAME: x = 0 : i32
// CHECK: ssavc4.vpm.read
// CHECK-SAME: lanes = 16 : i32
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK-SAME: stride = 1 : i32
// CHECK-SAME: subword = #ssavc4.vpm_subword<none>
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w32>
// CHECK-SAME: x = 0 : i32
ssavc4.module @vpm_shared_roundtrip {
  ssavc4.func @vpm_shared_kernel() attributes {
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
      uses_vpm_qpu_read = true,
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
    ssavc4.vpm.write %row0, %tile {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, vector<16xi32>
    %read = ssavc4.vpm.read %row0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32 -> vector<16xi32>
    ssavc4.vpm.write %row0, %read {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
