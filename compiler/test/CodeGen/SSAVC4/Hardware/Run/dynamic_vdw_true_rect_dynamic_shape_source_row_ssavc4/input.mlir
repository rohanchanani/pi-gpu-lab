ssavc4.module @dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4 {
  ssavc4.func @dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4",
      code_symbol = "dynamic_vdw_true_rect_dynamic_shape_source_row_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 10 : i32,
      args = [
        {name = "in0", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "in16", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 1 : i32},
        {name = "in32", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 2 : i32},
        {name = "in48", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 3 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "u8", uniform_index = 4 : i32},
        {name = "preserve", kind = "buffer", direction = "out", elem_type = "u8", uniform_index = 5 : i32},
        {name = "source_row", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 6 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 7 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 8 : i32},
        {name = "stride_bytes", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 9 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 64 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 64 : i32,
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
    %in0 = ssavc4.uniform.read 0 : i32
    %in16 = ssavc4.uniform.read 1 : i32
    %in32 = ssavc4.uniform.read 2 : i32
    %in48 = ssavc4.uniform.read 3 : i32
    %out = ssavc4.uniform.read 4 : i32
    %preserve = ssavc4.uniform.read 5 : i32
    %source_row = ssavc4.uniform.read 6 : i32
    %active_rows = ssavc4.uniform.read 7 : i32
    %active_cols = ssavc4.uniform.read 8 : i32
    %stride = ssavc4.uniform.read 9 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row16 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %row32 = ssavc4.load_imm <splat32> {value = 32 : i32} : i32
    %row48 = ssavc4.load_imm <splat32> {value = 48 : i32} : i32
    %rows1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %rows16 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %cols16 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %stride64 = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    ssavc4.vdr.load %in0, %row0 {
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      row_len = 16 : i32,
      nrows = 16 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_x = 0 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      vpm_pitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32

    ssavc4.vdr.load %in16, %row16 {
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      row_len = 16 : i32,
      nrows = 16 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_x = 0 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      vpm_pitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32

    ssavc4.vdr.load %in32, %row32 {
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      row_len = 16 : i32,
      nrows = 16 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_x = 0 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      vpm_pitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32

    ssavc4.vdr.load %in48, %row48 {
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      row_len = 16 : i32,
      nrows = 16 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_x = 0 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      vpm_pitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %active_rows, %active_cols, %stride {
      max_rows = 16 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 0 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.vdw.store_rect.dynamic %preserve, %row0, %rows1, %cols16, %stride64 {
      max_rows = 1 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 0 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.vdw.store_rect.dynamic %preserve, %source_row, %active_rows, %active_cols, %stride {
      max_rows = 16 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 0 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.thread_end
  }
}
