ssavc4.module @vdw_fallback_high_rows_static_source_probe_ssavc4 {
  ssavc4.func @vdw_fallback_high_rows_static_source_probe_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_fallback_high_rows_static_source_probe_ssavc4",
      code_symbol = "vdw_fallback_high_rows_static_source_probe_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 1 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u8", uniform_index = 0 : i32}
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
    %out = ssavc4.uniform.read 0 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %row2 = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %row3 = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %row4 = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %row5 = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %row6 = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %row7 = ssavc4.load_imm <splat32> {value = 7 : i32} : i32
    %row8 = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %row9 = ssavc4.load_imm <splat32> {value = 9 : i32} : i32
    %row10 = ssavc4.load_imm <splat32> {value = 10 : i32} : i32
    %row11 = ssavc4.load_imm <splat32> {value = 11 : i32} : i32
    %row12 = ssavc4.load_imm <splat32> {value = 12 : i32} : i32
    %row13 = ssavc4.load_imm <splat32> {value = 13 : i32} : i32
    %row14 = ssavc4.load_imm <splat32> {value = 14 : i32} : i32
    %row15 = ssavc4.load_imm <splat32> {value = 15 : i32} : i32
    %rows16 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %cols1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %stride17000 = ssavc4.load_imm <splat32> {value = 17000 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %base0 = ssavc4.load_imm <splat32> {value = 2113929216 : i32} : vector<16xi32>
    %base1 = ssavc4.load_imm <splat32> {value = 2113929472 : i32} : vector<16xi32>
    %base2 = ssavc4.load_imm <splat32> {value = 2113929728 : i32} : vector<16xi32>
    %base3 = ssavc4.load_imm <splat32> {value = 2113929984 : i32} : vector<16xi32>
    %base4 = ssavc4.load_imm <splat32> {value = 2113930240 : i32} : vector<16xi32>
    %base5 = ssavc4.load_imm <splat32> {value = 2113930496 : i32} : vector<16xi32>
    %base6 = ssavc4.load_imm <splat32> {value = 2113930752 : i32} : vector<16xi32>
    %base7 = ssavc4.load_imm <splat32> {value = 2113931008 : i32} : vector<16xi32>
    %base8 = ssavc4.load_imm <splat32> {value = 2113931264 : i32} : vector<16xi32>
    %base9 = ssavc4.load_imm <splat32> {value = 2113931520 : i32} : vector<16xi32>
    %base10 = ssavc4.load_imm <splat32> {value = 2113931776 : i32} : vector<16xi32>
    %base11 = ssavc4.load_imm <splat32> {value = 2113932032 : i32} : vector<16xi32>
    %base12 = ssavc4.load_imm <splat32> {value = 2113932288 : i32} : vector<16xi32>
    %base13 = ssavc4.load_imm <splat32> {value = 2113932544 : i32} : vector<16xi32>
    %base14 = ssavc4.load_imm <splat32> {value = 2113932800 : i32} : vector<16xi32>
    %base15 = ssavc4.load_imm <splat32> {value = 2113933056 : i32} : vector<16xi32>
    %value0 = ssavc4.alu.add %base0, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value1 = ssavc4.alu.add %base1, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value2 = ssavc4.alu.add %base2, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value3 = ssavc4.alu.add %base3, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value4 = ssavc4.alu.add %base4, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value5 = ssavc4.alu.add %base5, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value6 = ssavc4.alu.add %base6, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value7 = ssavc4.alu.add %base7, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value8 = ssavc4.alu.add %base8, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value9 = ssavc4.alu.add %base9, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value10 = ssavc4.alu.add %base10, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value11 = ssavc4.alu.add %base11, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value12 = ssavc4.alu.add %base12, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value13 = ssavc4.alu.add %base13, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value14 = ssavc4.alu.add %base14, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value15 = ssavc4.alu.add %base15, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    ssavc4.vpm.write %row0, %value0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row1, %value1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row2, %value2 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row3, %value3 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row4, %value4 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row5, %value5 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row6, %value6 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row7, %value7 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row8, %value8 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row9, %value9 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row10, %value10 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row11, %value11 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row12, %value12 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row13, %value13 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row14, %value14 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row15, %value15 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>

    ssavc4.vdw.store_rect.dynamic %out, %row0, %rows16, %cols1, %stride17000 {
      max_rows = 16 : i32,
      max_cols = 1 : i32,
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
