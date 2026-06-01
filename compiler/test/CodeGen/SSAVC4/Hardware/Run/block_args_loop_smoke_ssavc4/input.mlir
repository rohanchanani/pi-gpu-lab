ssavc4.module @block_args_loop_smoke_ssavc4 {
  ssavc4.func @block_args_loop_smoke_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "block_args_loop_smoke_ssavc4",
      code_symbol = "block_args_loop_smoke_ssavc4_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "case_id", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
      ],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 3 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 4 : i32}
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
    %case_id = ssavc4.uniform.read 2 : i32
    %qpu_id = ssavc4.uniform.read 3 : i32

    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %limit = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_four = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %shift_eight = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %full_bytes = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>

    %byte_offset = ssavc4.alu.add %qpu_id, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %logical_bytes = ssavc4.alu.add %n, %shift_two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %skip_flags = ssavc4.make_flags %byte_offset, %logical_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %skip_flags, ^done, ^body {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^body:
    %addr = ssavc4.alu.add %out, %byte_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %remaining = ssavc4.alu.add %logical_bytes, %byte_offset {opcode = #vc4.add_opcode<sub>} : (i32, i32) -> i32
    %base_elem = ssavc4.alu.add %qpu_id, %shift_four {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %base_vec = ssavc4.splat %base_elem : i32 -> vector<16xi32>
    %index = ssavc4.alu.add %base_vec, %lane {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %case_shift8 = ssavc4.alu.add %case_id, %shift_eight {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %case_base_vec = ssavc4.splat %case_shift8 : i32 -> vector<16xi32>
    %acc0 = ssavc4.alu.add %index, %case_base_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %case_step_scalar = ssavc4.alu.add %case_id, %shift_two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %case_step_vec = ssavc4.splat %case_step_scalar : i32 -> vector<16xi32>
    ssavc4.br ^loop(%zero, %acc0 : i32, vector<16xi32>)

  ^loop(%iter: i32, %acc: vector<16xi32>):
    %done_flags = ssavc4.make_flags %iter, %limit {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^after_loop(%acc : vector<16xi32>), ^step(%iter, %acc : i32, vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^step(%iter_step: i32, %acc_step: vector<16xi32>):
    %iter_vec = ssavc4.splat %iter_step : i32 -> vector<16xi32>
    %lane_iter = ssavc4.alu.add %lane, %iter_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %term = ssavc4.alu.add %lane_iter, %case_step_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %acc_next = ssavc4.alu.add %acc_step, %term {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %iter_next = ssavc4.alu.add %iter_step, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%iter_next, %acc_next : i32, vector<16xi32>)

  ^after_loop(%final: vector<16xi32>):
    %full_flags = ssavc4.make_flags %remaining, %full_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %full_flags, ^full(%final : vector<16xi32>), ^tail(%final : vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^tail(%tail_value: vector<16xi32>):
    %active = ssavc4.alu.add %remaining, %shift_two {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr, %tail_value, %active, %qpu_id {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32
    ssavc4.br ^done

  ^full(%full_value: vector<16xi32>):
    ssavc4.vdw.store %addr, %full_value, %active_full, %qpu_id {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32
    ssavc4.br ^done

  ^done:
    ssavc4.thread_end
  }
}
