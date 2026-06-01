ssavc4.module @vdw_store_vpm_dynamic_row4_probe_ssavc4 {
  ssavc4.func @vdw_store_vpm_dynamic_row4_probe_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_store_vpm_dynamic_row4_probe_ssavc4",
      code_symbol = "vdw_store_vpm_dynamic_row4_probe_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 4 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 4 : i32,
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
    %in = ssavc4.uniform.read 0 : i32
    %out = ssavc4.uniform.read 1 : i32

    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %three = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %four = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %off1 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %off2 = ssavc4.load_imm <splat32> {value = 32 : i32} : i32
    %off3 = ssavc4.load_imm <splat32> {value = 48 : i32} : i32
    %off4 = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    ssavc4.vdr.load %in, %zero {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32,
      nrows = 4 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_x = 0 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      vpm_pitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32

    ssavc4.vdw.store_vpm %out, %zero, %zero, %zero {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 4 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32, i32

    %addr1 = ssavc4.alu.add %out, %off1 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store_vpm %addr1, %zero, %zero, %one {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 4 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32, i32

    %addr2 = ssavc4.alu.add %out, %off2 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store_vpm %addr2, %one, %zero, %two {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 4 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32, i32

    %addr3 = ssavc4.alu.add %out, %off3 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store_vpm %addr3, %two, %zero, %three {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 4 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32, i32

    %addr4 = ssavc4.alu.add %out, %off4 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdw.store_vpm %addr4, %three, %zero, %four {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 4 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32, i32

    ssavc4.thread_end
  }
}
