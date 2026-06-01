ssavc4.module @vpm_horizontal_32_stride_probe_ssavc4 {
  ssavc4.func @vpm_horizontal_32_stride_probe_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vpm_horizontal_32_stride_probe_ssavc4",
      code_symbol = "vpm_horizontal_32_stride_probe_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 1 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 4 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 4 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = true,
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
    %row2 = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %off64 = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %active = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %base0 = ssavc4.load_imm <splat32> {value = 1627389952 : i32} : vector<16xi32>
    %base2 = ssavc4.load_imm <splat32> {value = 1644167168 : i32} : vector<16xi32>
    %value0 = ssavc4.alu.add %base0, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value2 = ssavc4.alu.add %base2, %lane {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    ssavc4.vpm.write %row0, %value0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.vpm.write %row2, %value2 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32, vector<16xi32>
    %read0 = ssavc4.vpm.read %row0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>
    %read2 = ssavc4.vpm.read %row2 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xi32>

    ssavc4.vdw.store %out, %read0, %active, %row0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32
    %out1 = ssavc4.alu.add %out, %off64 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store %out1, %read2, %active, %row0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32
    ssavc4.thread_end
  }
}
