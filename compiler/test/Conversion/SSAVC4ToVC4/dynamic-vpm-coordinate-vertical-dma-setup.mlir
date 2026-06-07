// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @dynamic_vpm_coordinate_vertical_dma_setup
// CHECK: vc4.func @kernel
// Vertical w32 VDR/VDW dynamic X is accepted in P12b and composed into DMA setup.
// CHECK: op_add = #vc4.add_opcode<and>
// CHECK: op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: ssavc4.
ssavc4.module @dynamic_vpm_coordinate_vertical_dma_setup {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_vpm_coordinate_vertical_dma_setup",
      code_symbol = "dynamic_vpm_coordinate_vertical_dma_setup_shader",
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
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
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
    ssavc4.vdr.load %addr, %row dynamic_x %x {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 64 : i32,
      orientation = #ssavc4.vpm_orientation<vertical>,
      vpm_pitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32 dynamic_x i32
    ssavc4.vdr.load_rect.dynamic %addr, %row dynamic_dst_x %x, %rows, %cols, %pitch {
      max_rows = 1 : i32, max_cols = 8 : i32, elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<vertical>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      vpm_pitch = 1 : i32,
      zero_fill = true,
      serialize = "mutex"
    } : i32, i32 dynamic_dst_x i32, i32, i32, i32
    ssavc4.vdw.store_vpm %addr, %row, %x {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 64 : i32,
      active_lanes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<vertical>,
      serialize = "mutex"
    } : i32, i32, i32
    ssavc4.vdw.store_rect.dynamic %addr, %row dynamic_src_x %x, %rows, %cols, %pitch {
      max_rows = 1 : i32, max_cols = 8 : i32, elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<vertical>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32 dynamic_src_x i32, i32, i32, i32
    ssavc4.thread_end
  }
}
