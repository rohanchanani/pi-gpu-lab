ssavc4.module @dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4 {
  ssavc4.func @dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4",
      code_symbol = "dynamic_vdr_vdw_runtime_pitch_cols_sweep_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "out_zero", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
        {name = "out_preserve", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 2 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
        {name = "pitch_bytes", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32}
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
      uses_vpm_qpu_write = true,
      uses_vdr = true,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %in = ssavc4.uniform.read 0 : i32
    %out_zero = ssavc4.uniform.read 1 : i32
    %out_preserve = ssavc4.uniform.read 2 : i32
    %active_cols = ssavc4.uniform.read 3 : i32
    %pitch = ssavc4.uniform.read 4 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32

    ssavc4.vdr.load_rect.dynamic %in, %row0, %rows, %active_cols, %pitch {
      max_rows = 1 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      dst_x = 0 : i32,
      vpm_pitch = 1 : i32,
      zero_fill = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.vdw.store_vpm %out_zero, %row0, %row0 {
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      row_len = 16 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 64 : i32,
      active_lanes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32

    ssavc4.vdw.store_rect.dynamic %out_preserve, %row0, %rows, %active_cols, %pitch {
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

    ssavc4.thread_end
  }
}
