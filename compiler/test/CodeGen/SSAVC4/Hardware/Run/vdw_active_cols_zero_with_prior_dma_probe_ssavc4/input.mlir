ssavc4.module @vdw_active_cols_zero_with_prior_dma_probe_ssavc4 {
  ssavc4.func @vdw_active_cols_zero_with_prior_dma_probe_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_active_cols_zero_with_prior_dma_probe_ssavc4",
      code_symbol = "vdw_active_cols_zero_with_prior_dma_probe_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "first", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "second", kind = "buffer", direction = "out", elem_type = "u8", uniform_index = 1 : i32},
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
      uses_vpm_qpu_write = true,
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %first = ssavc4.uniform.read 0 : i32
    %second = ssavc4.uniform.read 1 : i32
    %active_rows = ssavc4.uniform.read 2 : i32
    %active_cols = ssavc4.uniform.read 3 : i32
    %stride = ssavc4.uniform.read 4 : i32
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %row2 = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %row3 = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %row4 = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %row5 = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %row6 = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %row7 = ssavc4.load_imm <splat32> {value = 7 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %base0 = ssavc4.load_imm <splat32> {value = 2147483648 : i32} : vector<16xi32>
    %base1 = ssavc4.load_imm <splat32> {value = 2147483904 : i32} : vector<16xi32>
    %value0 = ssavc4.alu.add %base0, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value1 = ssavc4.alu.add %base1, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.vpm.write %row0, %value0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row1, %value1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row2, %value1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row3, %value1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row4, %value1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row5, %value1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row6, %value1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row7, %value1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vdw.store_vpm %first, %row0, %zero {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 1 : i32, nrows = 1 : i32, memory_pitch_bytes = 4 : i32, active_lanes = 1 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, i32, i32
    ssavc4.vdw.store_rect.dynamic %second, %row1, %active_rows, %active_cols, %stride {
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
