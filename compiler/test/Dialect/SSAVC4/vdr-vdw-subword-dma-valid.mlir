// RUN: vc4-opt %s | FileCheck %s

ssavc4.module @subword_dma_valid {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 4 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 4 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = true,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %pitch8 = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %pitch16 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    // CHECK: ssavc4.vdr.load
    // CHECK-SAME: subword = #ssavc4.vpm_subword<packed>
    // CHECK-SAME: width = #ssavc4.vpm_elem_width<w8>
    ssavc4.vdr.load %addr, %row {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, vpm_x = 3 : i32, subword_selector = 3 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32
    // CHECK: ssavc4.vdr.load_rect.dynamic
    // CHECK-SAME: elem_bytes = 2
    // CHECK-SAME: width = #ssavc4.vpm_elem_width<w16>
    ssavc4.vdr.load_rect.dynamic %addr, %row, %rows, %cols, %pitch16 {max_rows = 1 : i32, max_cols = 8 : i32, elem_bytes = 2 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, dst_x = 1 : i32, subword_selector = 1 : i32, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32, i32, i32, i32
    // CHECK: ssavc4.vdw.store_vpm
    // CHECK-SAME: subword = #ssavc4.vpm_subword<packed>
    // CHECK-SAME: width = #ssavc4.vpm_elem_width<w16>
    ssavc4.vdw.store_vpm %addr, %row, %row {width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, row_len = 8 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, subword_selector = 1 : i32, active_lanes = 8 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, i32, i32
    // CHECK: ssavc4.vdw.store_rect.dynamic
    // CHECK-SAME: elem_bytes = 1
    // CHECK-SAME: width = #ssavc4.vpm_elem_width<w8>
    ssavc4.vdw.store_rect.dynamic %addr, %row, %rows, %cols, %pitch8 {max_rows = 1 : i32, max_cols = 8 : i32, elem_bytes = 1 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, src_x = 0 : i32, subword_selector = 3 : i32, vpm_pitch = 1 : i32, preserve_inactive = true} : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}
