ssavc4.module @two_tmu_f32_loop_tail_b_ssavc4 {
  ssavc4.func @two_tmu_f32_loop_tail_b_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "two_tmu_f32_loop_tail_b_ssavc4",
      code_symbol = "two_tmu_f32_loop_tail_b_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 7 : i32,
      args = [
        {name = "a", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
        {name = "b", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 2 : i32},
        {name = "k", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32}
      ],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 5 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 6 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 0 : i32,
      compiler_vpm_staging_rows_per_warp = 1 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 1 : i32,
      uses_tmu = true,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %a_base = ssavc4.uniform.read 0 : i32
    %b_base = ssavc4.uniform.read 1 : i32
    %out_base = ssavc4.uniform.read 2 : i32
    %k = ssavc4.uniform.read 3 : i32
    %active_cols = ssavc4.uniform.read 4 : i32
    %qpu_id = ssavc4.uniform.read 5 : i32
    %zero_i = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one_i = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %zero_f = ssavc4.load_imm <splat32> {value = 0.000000e+00 : f32} : f32
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_two_vec = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %lane_bytes = ssavc4.alu.add %lane, %shift_two_vec {opcode = #vc4.add_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %zero_v = ssavc4.splat %zero_f : f32 -> vector<16xf32>
    %active_cols_v = ssavc4.splat %active_cols : i32 -> vector<16xi32>
    ssavc4.br ^loop(%zero_i, %zero_v : i32, vector<16xf32>)

  ^loop(%iter: i32, %acc: vector<16xf32>):
    %done_flags = ssavc4.make_flags %iter, %k {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^step(%iter, %acc : i32, vector<16xf32>), ^after(%acc : vector<16xf32>) {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags

  ^step(%iter_step: i32, %acc_step: vector<16xf32>):
    %a_offset = ssavc4.alu.add %iter_step, %shift_two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %b_row_offset = ssavc4.alu.add %iter_step, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %a_offset_v = ssavc4.splat %a_offset : i32 -> vector<16xi32>
    %b_row_offset_v = ssavc4.splat %b_row_offset : i32 -> vector<16xi32>
    %a_base_v = ssavc4.splat %a_base : i32 -> vector<16xi32>
    %b_base_v = ssavc4.splat %b_base : i32 -> vector<16xi32>
    %a_addr = ssavc4.alu.add %a_base_v, %a_offset_v {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b_raw_offset = ssavc4.alu.add %b_row_offset_v, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tail_flags = ssavc4.make_flags %lane, %active_cols_v {kind = #ssavc4.flag_kind<sub>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
    %b_safe_offset = ssavc4.cond_select %tail_flags, %b_raw_offset, %b_row_offset_v {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %b_addr = ssavc4.alu.add %b_base_v, %b_safe_offset {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a_tok = ssavc4.tmu.request %a_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %a_val = ssavc4.tmu.read %a_tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    %b_tok = ssavc4.tmu.request %b_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %b_loaded = ssavc4.tmu.read %b_tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    %tail_flags_result = ssavc4.make_flags %lane, %active_cols_v {kind = #ssavc4.flag_kind<sub>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
    %b_val = ssavc4.cond_select %tail_flags_result, %b_loaded, %zero_v {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %prod = ssavc4.alu.mul %a_val, %b_val {opcode = #vc4.mul_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc_next = ssavc4.alu.add %acc_step, %prod {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %iter_next = ssavc4.alu.add %iter_step, %one_i {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%iter_next, %acc_next : i32, vector<16xf32>)

  ^after(%result: vector<16xf32>):
    ssavc4.vdw.store %out_base, %result, %active_full, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.thread_end
  }
}
