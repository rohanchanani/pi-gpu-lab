ssavc4.module @natural_loop_vector_accumulator_ssavc4 {
  ssavc4.func @natural_loop_vector_accumulator_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "natural_loop_vector_accumulator_ssavc4",
      code_symbol = "natural_loop_vector_accumulator_ssavc4_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 8 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 0 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "iterations", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "seed", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 3 : i32},
        {name = "step", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 4 : i32},
        {name = "case_id", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 5 : i32}
      ],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 6 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 7 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 0 : i32,
      compiler_vpm_staging_rows_per_warp = 1 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 1 : i32,
      uses_tmu = false,
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
    %out = ssavc4.uniform.read 0 : i32
    %n = ssavc4.uniform.read 1 : i32
    %iterations = ssavc4.uniform.read 2 : i32
    %seed = ssavc4.uniform.read 3 : f32
    %step = ssavc4.uniform.read 4 : f32
    %case_id = ssavc4.uniform.read 5 : i32
    %qpu_id = ssavc4.uniform.read 6 : i32

    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %full_bytes = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    %seed_vec = ssavc4.splat %seed : f32 -> vector<16xf32>
    %step_vec = ssavc4.splat %step : f32 -> vector<16xf32>
    %case_vec = ssavc4.splat %case_id : i32 -> vector<16xi32>
    %case_fudge = ssavc4.alu.add %case_vec, %case_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %byte_offset = ssavc4.alu.add %qpu_id, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %logical_bytes = ssavc4.alu.add %n, %shift_two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %skip_flags = ssavc4.make_flags %byte_offset, %logical_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %skip_flags, ^done, ^preheader {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^preheader:
    %addr = ssavc4.alu.add %out, %byte_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %remaining = ssavc4.alu.add %logical_bytes, %byte_offset {opcode = #vc4.add_opcode<sub>} : (i32, i32) -> i32
    ssavc4.br ^loop(%zero, %seed_vec : i32, vector<16xf32>)

  ^loop(%iter: i32, %acc: vector<16xf32>):
    %done_flags = ssavc4.make_flags %iter, %iterations {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^after_loop(%acc : vector<16xf32>), ^step(%iter, %acc : i32, vector<16xf32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^step(%iter_step: i32, %acc_step: vector<16xf32>):
    %next_acc = ssavc4.alu.add %acc_step, %step_vec {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %iter_next = ssavc4.alu.add %iter_step, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%iter_next, %next_acc : i32, vector<16xf32>)

  ^after_loop(%final: vector<16xf32>):
    %use_case = ssavc4.alu.add %case_fudge, %case_fudge {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %full_flags = ssavc4.make_flags %remaining, %full_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %full_flags, ^full(%final, %use_case : vector<16xf32>, vector<16xi32>), ^tail(%final, %use_case : vector<16xf32>, vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^tail(%tail_value: vector<16xf32>, %tail_case: vector<16xi32>):
    %active = ssavc4.alu.add %remaining, %shift_two {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr, %tail_value, %active, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.br ^done

  ^full(%full_value: vector<16xf32>, %full_case: vector<16xi32>):
    ssavc4.vdw.store %addr, %full_value, %active_full, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.br ^done

  ^done:
    ssavc4.thread_end
  }
}
