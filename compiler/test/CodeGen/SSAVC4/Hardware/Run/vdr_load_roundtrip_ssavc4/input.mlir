ssavc4.module @vdr_load_roundtrip_ssavc4 {
  ssavc4.func @vdr_load_roundtrip_ssavc4() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdr_load_roundtrip_ssavc4",
      code_symbol = "vdr_load_roundtrip_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = true,
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
    %vdr_row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    ssavc4.vdr.load %in, %vdr_row {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32,
      nrows = 16 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_x = 0 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      vpm_pitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %tile0 = ssavc4.vpm.read %row0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    ssavc4.vdw.store %out, %tile0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %tile1 = ssavc4.vpm.read %row1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off1 = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %addr1 = ssavc4.alu.add %out, %off1 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr1, %tile1 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row2 = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %tile2 = ssavc4.vpm.read %row2 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off2 = ssavc4.load_imm <splat32> {value = 128 : i32} : i32
    %addr2 = ssavc4.alu.add %out, %off2 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr2, %tile2 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row3 = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %tile3 = ssavc4.vpm.read %row3 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off3 = ssavc4.load_imm <splat32> {value = 192 : i32} : i32
    %addr3 = ssavc4.alu.add %out, %off3 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr3, %tile3 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row4 = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %tile4 = ssavc4.vpm.read %row4 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off4 = ssavc4.load_imm <splat32> {value = 256 : i32} : i32
    %addr4 = ssavc4.alu.add %out, %off4 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr4, %tile4 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row5 = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %tile5 = ssavc4.vpm.read %row5 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off5 = ssavc4.load_imm <splat32> {value = 320 : i32} : i32
    %addr5 = ssavc4.alu.add %out, %off5 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr5, %tile5 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row6 = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %tile6 = ssavc4.vpm.read %row6 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off6 = ssavc4.load_imm <splat32> {value = 384 : i32} : i32
    %addr6 = ssavc4.alu.add %out, %off6 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr6, %tile6 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row7 = ssavc4.load_imm <splat32> {value = 7 : i32} : i32
    %tile7 = ssavc4.vpm.read %row7 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off7 = ssavc4.load_imm <splat32> {value = 448 : i32} : i32
    %addr7 = ssavc4.alu.add %out, %off7 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr7, %tile7 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row8 = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %tile8 = ssavc4.vpm.read %row8 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off8 = ssavc4.load_imm <splat32> {value = 512 : i32} : i32
    %addr8 = ssavc4.alu.add %out, %off8 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr8, %tile8 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row9 = ssavc4.load_imm <splat32> {value = 9 : i32} : i32
    %tile9 = ssavc4.vpm.read %row9 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off9 = ssavc4.load_imm <splat32> {value = 576 : i32} : i32
    %addr9 = ssavc4.alu.add %out, %off9 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr9, %tile9 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row10 = ssavc4.load_imm <splat32> {value = 10 : i32} : i32
    %tile10 = ssavc4.vpm.read %row10 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off10 = ssavc4.load_imm <splat32> {value = 640 : i32} : i32
    %addr10 = ssavc4.alu.add %out, %off10 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr10, %tile10 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row11 = ssavc4.load_imm <splat32> {value = 11 : i32} : i32
    %tile11 = ssavc4.vpm.read %row11 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off11 = ssavc4.load_imm <splat32> {value = 704 : i32} : i32
    %addr11 = ssavc4.alu.add %out, %off11 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr11, %tile11 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row12 = ssavc4.load_imm <splat32> {value = 12 : i32} : i32
    %tile12 = ssavc4.vpm.read %row12 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off12 = ssavc4.load_imm <splat32> {value = 768 : i32} : i32
    %addr12 = ssavc4.alu.add %out, %off12 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr12, %tile12 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row13 = ssavc4.load_imm <splat32> {value = 13 : i32} : i32
    %tile13 = ssavc4.vpm.read %row13 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off13 = ssavc4.load_imm <splat32> {value = 832 : i32} : i32
    %addr13 = ssavc4.alu.add %out, %off13 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr13, %tile13 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row14 = ssavc4.load_imm <splat32> {value = 14 : i32} : i32
    %tile14 = ssavc4.vpm.read %row14 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off14 = ssavc4.load_imm <splat32> {value = 896 : i32} : i32
    %addr14 = ssavc4.alu.add %out, %off14 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr14, %tile14 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    %row15 = ssavc4.load_imm <splat32> {value = 15 : i32} : i32
    %tile15 = ssavc4.vpm.read %row15 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %off15 = ssavc4.load_imm <splat32> {value = 960 : i32} : i32
    %addr15 = ssavc4.alu.add %out, %off15 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr15, %tile15 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
