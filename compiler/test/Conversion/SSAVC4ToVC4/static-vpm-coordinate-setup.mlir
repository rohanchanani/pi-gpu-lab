// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @static_vpm_coordinate_setup
// CHECK: vc4.func @kernel
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: ssavc4.
ssavc4.module @static_vpm_coordinate_setup {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "static_vpm_coordinate_setup",
      code_symbol = "static_vpm_coordinate_setup_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 1 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32}
      ],
      builtins = []
    },
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
      uses_vdr = true,
      uses_vdw = true,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
    }
  } {
    %addr = ssavc4.uniform.read 0 : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    ssavc4.vpm.write %row, %value {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>,
      x = 0 : i32, stride = 1 : i32, lanes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>
    } : i32, vector<16xi32>
    %read = ssavc4.vpm.read %row {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>,
      x = 0 : i32, stride = 1 : i32, lanes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>
    } : i32 -> vector<16xi32>
    ssavc4.vdr.load %addr, %row {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_x = 3 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      vpm_pitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32
    ssavc4.vdr.load_rect.dynamic %addr, %row, %rows, %cols, %pitch {
      max_rows = 1 : i32, max_cols = 8 : i32, elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      dst_x = 3 : i32,
      vpm_pitch = 1 : i32,
      zero_fill = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32
    ssavc4.vdw.store_vpm %addr, %row, %x {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 64 : i32,
      active_lanes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32
    ssavc4.vdw.store_rect.dynamic %addr, %row, %rows, %cols, %pitch {
      max_rows = 1 : i32, max_cols = 8 : i32, elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 3 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}
