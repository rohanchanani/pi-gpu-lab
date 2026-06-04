ssavc4.module @dynamic_vdw_setup_word_ssavc4 {
  ssavc4.func @dynamic_vdw_setup_word_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_vdw_setup_word_ssavc4",
      code_symbol = "dynamic_vdw_setup_word_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
        {name = "stride_bytes", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 1 : i32,
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
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %in = ssavc4.uniform.read 0 : i32
    %out = ssavc4.uniform.read 1 : i32
    %active_rows = ssavc4.uniform.read 2 : i32
    %active_cols = ssavc4.uniform.read 3 : i32
    %stride = ssavc4.uniform.read 4 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32

    ssavc4.vdr.load %in, %row0 {
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

    ssavc4.vdw.store_rect.dynamic %out, %row0, %active_rows, %active_cols, %stride {
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
