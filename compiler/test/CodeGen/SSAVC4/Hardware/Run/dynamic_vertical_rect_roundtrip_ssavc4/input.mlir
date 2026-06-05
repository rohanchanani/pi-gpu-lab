ssavc4.module @dynamic_vertical_rect_roundtrip_ssavc4 {
  ssavc4.func @dynamic_vertical_rect_roundtrip_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_vertical_rect_roundtrip_ssavc4",
      code_symbol = "dynamic_vertical_rect_roundtrip_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "out_vdr", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
        {name = "out_vdw", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 2 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
        {name = "pitch_stride_bytes", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 5 : i32}
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
    %out_vdr = ssavc4.uniform.read 1 : i32
    %out_vdw = ssavc4.uniform.read 2 : i32
    %active_rows = ssavc4.uniform.read 3 : i32
    %active_cols = ssavc4.uniform.read 4 : i32
    %pitch_stride = ssavc4.uniform.read 5 : i32
    %vpm_row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %vpm_row8 = ssavc4.load_imm <splat32> {value = 8 : i32} : i32

    ssavc4.vdr.load_rect.dynamic %in, %vpm_row0, %active_rows, %active_cols, %pitch_stride {
      max_rows = 4 : i32,
      max_cols = 8 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<vertical>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      dst_x = 2 : i32,
      vpm_pitch = 1 : i32,
      zero_fill = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.vdw.store_rect.dynamic %out_vdr, %vpm_row0, %active_cols, %active_rows, %pitch_stride {
      max_rows = 8 : i32,
      max_cols = 4 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 2 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.vdr.load_rect.dynamic %in, %vpm_row8, %active_cols, %active_rows, %pitch_stride {
      max_rows = 8 : i32,
      max_cols = 4 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      dst_x = 0 : i32,
      vpm_pitch = 1 : i32,
      zero_fill = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.vdw.store_rect.dynamic %out_vdw, %vpm_row8, %active_rows, %active_cols, %pitch_stride {
      max_rows = 4 : i32,
      max_cols = 8 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<vertical>,
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
