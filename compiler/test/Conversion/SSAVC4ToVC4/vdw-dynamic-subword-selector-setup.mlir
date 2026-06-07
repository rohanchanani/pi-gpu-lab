// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @vdw_fragment_selector
// CHECK: op_add = #vc4.add_opcode<and>
// CHECK-SAME: small_imm = 3 : i32
// CHECK: value = -1073741824 : i32
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}op_add = #vc4.add_opcode<or>
// CHECK-SAME: side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @vdw_rect_selector
// CHECK: op_add = #vc4.add_opcode<and>
// CHECK-SAME: small_imm = 1 : i32
// CHECK: value = -1073741824 : i32
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}op_add = #vc4.add_opcode<or>
// CHECK-SAME: side = #vc4.vpmvcd_side<write>

ssavc4.module @vdw_dynamic_subword_selector_setup {
  ssavc4.func @vdw_fragment_selector() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
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
      requires_vpm_base_row_builtin = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_semaphore_base_builtin = false,
      spill_frame_bytes = 0 : i32
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %sel = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    ssavc4.vdw.store_vpm %addr, %row, %x dynamic_subword_selector %sel {
      width = #ssavc4.vpm_elem_width<w8>,
      subword = #ssavc4.vpm_subword<packed>,
      row_len = 8 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 8 : i32,
      active_lanes = 8 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32 dynamic_subword_selector i32
    ssavc4.thread_end
  }

  ssavc4.func @vdw_rect_selector() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
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
      requires_vpm_base_row_builtin = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_semaphore_base_builtin = false,
      spill_frame_bytes = 0 : i32
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %sel = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    ssavc4.vdw.store_rect.dynamic %addr, %row dynamic_src_x %x dynamic_subword_selector %sel, %rows, %cols, %stride {
      max_rows = 1 : i32,
      max_cols = 8 : i32,
      elem_bytes = 2 : i32,
      orientation = #ssavc4.vpm_orientation<vertical>,
      width = #ssavc4.vpm_elem_width<w16>,
      subword = #ssavc4.vpm_subword<packed>,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32
    ssavc4.thread_end
  }
}
