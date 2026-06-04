ssavc4.module @f32_tail_vpmread_loop_backedge_edgecopy_ssavc4 {
  ssavc4.func @f32_tail_vpmread_loop_backedge_edgecopy_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "f32_tail_vpmread_loop_backedge_edgecopy_ssavc4",
      code_symbol = "f32_tail_vpmread_loop_backedge_edgecopy_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 1 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "pitch_bytes", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
      ],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 4 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 5 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 1 : i32,
      compiler_vpm_staging_rows_per_warp = 1 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 2 : i32,
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
    %active_cols = ssavc4.uniform.read 2 : i32
    %pitch = ssavc4.uniform.read 3 : i32
    %qpu_id = ssavc4.uniform.read 4 : i32

    %zero_i = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one_i = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %iters = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %zero_v = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xf32>
    %seed_v = ssavc4.load_imm <splat32> {value = 1056964608 : i32} : vector<16xf32>
    %carry0 = ssavc4.load_imm <splat32> {value = 1048576000 : i32} : vector<16xf32>
    %carry_step = ssavc4.load_imm <splat32> {value = 1040187392 : i32} : vector<16xf32>
    %lane = ssavc4.element_number : vector<16xi32>
    %active_vec = ssavc4.splat %active_cols : i32 -> vector<16xi32>
    ssavc4.br ^loop(%zero_i, %seed_v, %carry0 : i32, vector<16xf32>, vector<16xf32>)

  ^loop(%iter: i32, %acc: vector<16xf32>, %carry: vector<16xf32>):
    %done_flags = ssavc4.make_flags %iter, %iters {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^store(%acc : vector<16xf32>), ^step(%iter, %acc, %carry : i32, vector<16xf32>, vector<16xf32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^step(%iter_step: i32, %acc_step: vector<16xf32>, %carry_in: vector<16xf32>):
    %src_offset = ssavc4.alu.mul %iter_step, %pitch {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
    %src = ssavc4.alu.add %in, %src_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdr.load %src, %row0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, vpm_x = 0 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32, serialize = "mutex"} : i32, i32
    %read = ssavc4.vpm.read %row0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xf32>
    %tail_flags = ssavc4.make_flags %lane, %active_vec {kind = #ssavc4.flag_kind<sub>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
    %masked = ssavc4.cond_select %tail_flags, %read, %zero_v {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %with_read = ssavc4.alu.add %acc_step, %masked {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc_next = ssavc4.alu.add %with_read, %carry_in {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %carry_next = ssavc4.alu.add %carry_in, %carry_step {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %iter_next = ssavc4.alu.add %iter_step, %one_i {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^backedge(%iter_next, %acc_next, %carry_next : i32, vector<16xf32>, vector<16xf32>)

  ^backedge(%iter_edge: i32, %acc_edge: vector<16xf32>, %carry_edge: vector<16xf32>):
    ssavc4.br ^loop(%iter_edge, %acc_edge, %carry_edge : i32, vector<16xf32>, vector<16xf32>)

  ^store(%final: vector<16xf32>):
    ssavc4.vdw.store %out, %final, %active_full, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.thread_end
  }
}
