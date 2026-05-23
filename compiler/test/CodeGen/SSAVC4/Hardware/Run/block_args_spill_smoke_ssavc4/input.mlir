ssavc4.module @block_args_spill_smoke_ssavc4 {
  ssavc4.func @block_args_spill_smoke_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "block_args_spill_smoke_ssavc4",
      code_symbol = "block_args_spill_smoke_ssavc4_shader",
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
      warps_per_block_max = 1 : i32,
      uses_shared_vpm = false,
      uses_barrier = false,
      semaphores_per_block = 0 : i32,
      require_full_block_residency = false
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
    %zero_vec = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %one_vec = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
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

    %p01 = ssavc4.alu.add %index, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p02 = ssavc4.alu.add %p01, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p03 = ssavc4.alu.add %p02, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p04 = ssavc4.alu.add %p03, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p05 = ssavc4.alu.add %p04, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p06 = ssavc4.alu.add %p05, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p07 = ssavc4.alu.add %p06, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p08 = ssavc4.alu.add %p07, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p09 = ssavc4.alu.add %p08, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p10 = ssavc4.alu.add %p09, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p11 = ssavc4.alu.add %p10, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p12 = ssavc4.alu.add %p11, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p13 = ssavc4.alu.add %p12, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p14 = ssavc4.alu.add %p13, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p15 = ssavc4.alu.add %p14, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p16 = ssavc4.alu.add %p15, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^loop(%zero, %acc0, %p01 : i32, vector<16xi32>, vector<16xi32>)

  ^loop(%iter: i32, %acc: vector<16xi32>, %carry: vector<16xi32>):
    %done_flags = ssavc4.make_flags %iter, %limit {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^after_loop(%acc, %carry : vector<16xi32>, vector<16xi32>), ^step(%iter, %acc, %carry : i32, vector<16xi32>, vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^step(%iter_step: i32, %acc_step: vector<16xi32>, %carry_step: vector<16xi32>):
    %iter_vec = ssavc4.splat %iter_step : i32 -> vector<16xi32>
    %lane_iter = ssavc4.alu.add %lane, %iter_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %term_base = ssavc4.alu.add %lane_iter, %case_step_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %term = ssavc4.alu.add %term_base, %carry_step {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %acc_next = ssavc4.alu.add %acc_step, %term {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %carry_next = ssavc4.alu.add %carry_step, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %iter_next = ssavc4.alu.add %iter_step, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%iter_next, %acc_next, %carry_next : i32, vector<16xi32>, vector<16xi32>)

  ^after_loop(%final_acc: vector<16xi32>, %final_carry: vector<16xi32>):
    %s00 = ssavc4.alu.add %final_acc, %final_carry {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s01 = ssavc4.alu.add %s00, %p01 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s02 = ssavc4.alu.add %s01, %p02 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s03 = ssavc4.alu.add %s02, %p03 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s04 = ssavc4.alu.add %s03, %p04 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s05 = ssavc4.alu.add %s04, %p05 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s06 = ssavc4.alu.add %s05, %p06 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s07 = ssavc4.alu.add %s06, %p07 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s08 = ssavc4.alu.add %s07, %p08 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %z09 = ssavc4.alu.add %p09, %zero_vec {opcode = #vc4.add_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n09 = ssavc4.alu.add %s08, %z09 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %z10 = ssavc4.alu.add %p10, %zero_vec {opcode = #vc4.add_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n10 = ssavc4.alu.add %n09, %z10 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %z11 = ssavc4.alu.add %p11, %zero_vec {opcode = #vc4.add_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n11 = ssavc4.alu.add %n10, %z11 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %z12 = ssavc4.alu.add %p12, %zero_vec {opcode = #vc4.add_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n12 = ssavc4.alu.add %n11, %z12 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %z13 = ssavc4.alu.add %p13, %zero_vec {opcode = #vc4.add_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n13 = ssavc4.alu.add %n12, %z13 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %z14 = ssavc4.alu.add %p14, %zero_vec {opcode = #vc4.add_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n14 = ssavc4.alu.add %n13, %z14 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %z15 = ssavc4.alu.add %p15, %zero_vec {opcode = #vc4.add_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n15 = ssavc4.alu.add %n14, %z15 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %z16 = ssavc4.alu.add %p16, %zero_vec {opcode = #vc4.add_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %final = ssavc4.alu.add %n15, %z16 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
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
