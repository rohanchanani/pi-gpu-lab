// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @subword_dma_setup
// VDR w8 packed x=3: ID | MODEW(7) | MPITCH(16 bytes) | NROWS(1) | VPITCH(4) | X(3)
// CHECK: value = -251576317 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// VDR w16 packed horizontal x=1: ID | MODEW(3) | MPITCH(16 bytes) | ROWLEN(4) | NROWS(1) | VPITCH(2) | X(1)
// CHECK: value = -1321132031 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// VDW w16 packed x=1: setup base includes MODEW(3).
// CHECK: value = -2139078653 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// VDW w8 packed horizontal x=3: setup base includes MODEW(7).
// CHECK: value = -2130690041 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-NOT: ssavc4.
// CHECK-LABEL: vc4.func @subword_vdw_rect_row_fallback
// CHECK: value = -2139078652 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK: value = 32 : i32
// CHECK: value = -2139078652 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-NOT: ssavc4.
ssavc4.module @subword_dma_setup {
  ssavc4.func @subword_dma_setup() attributes {
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
    %addr = ssavc4.load_imm <splat32> {value = 4096 : i32} : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %x3 = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    ssavc4.vdr.load %addr, %row0 {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, vpm_x = 3 : i32, subword_selector = 3 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32
    ssavc4.vdr.load %addr, %row0 {width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, row_len = 4 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, vpm_x = 1 : i32, subword_selector = 1 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32
    ssavc4.vdw.store_vpm %addr, %row0, %x1 {width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, row_len = 8 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, subword_selector = 1 : i32, active_lanes = 8 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, i32, i32
    ssavc4.vdw.store_vpm %addr, %row0, %x3 {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, row_len = 16 : i32, nrows = 2 : i32, memory_pitch_bytes = 16 : i32, subword_selector = 3 : i32, active_lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, i32, i32
    ssavc4.thread_end
  }
  ssavc4.func @subword_vdw_rect_row_fallback() attributes {
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
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 4096 : i32} : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 32 : i32} : i32
    ssavc4.vdw.store_rect.dynamic %addr, %row0, %rows, %cols, %stride {max_rows = 2 : i32, max_cols = 16 : i32, elem_bytes = 1 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, src_x = 0 : i32, subword_selector = 0 : i32, vpm_pitch = 1 : i32, preserve_inactive = true} : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}
