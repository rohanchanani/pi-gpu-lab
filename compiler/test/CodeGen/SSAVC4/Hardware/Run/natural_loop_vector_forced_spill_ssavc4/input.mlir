ssavc4.module @natural_loop_vector_forced_spill_ssavc4 {
  ssavc4.func @natural_loop_vector_forced_spill_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "natural_loop_vector_forced_spill_ssavc4",
      code_symbol = "natural_loop_vector_forced_spill_ssavc4_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 9 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 0 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "iterations", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "seed", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 3 : i32},
        {name = "alpha", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 4 : i32},
        {name = "beta", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 5 : i32},
        {name = "case_id", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 6 : i32}
      ],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 7 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 8 : i32}
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
    %alpha = ssavc4.uniform.read 4 : f32
    %beta = ssavc4.uniform.read 5 : f32
    %case_id = ssavc4.uniform.read 6 : i32
    %qpu_id = ssavc4.uniform.read 7 : i32

    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %full_bytes = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    %seed_v = ssavc4.splat %seed : f32 -> vector<16xf32>
    %alpha_v = ssavc4.splat %alpha : f32 -> vector<16xf32>
    %beta_v = ssavc4.splat %beta : f32 -> vector<16xf32>
    %case_v = ssavc4.splat %case_id : i32 -> vector<16xi32>

    %byte_offset = ssavc4.alu.add %qpu_id, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %logical_bytes = ssavc4.alu.add %n, %shift_two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %skip_flags = ssavc4.make_flags %byte_offset, %logical_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %skip_flags, ^done, ^preheader {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^preheader:
    %addr = ssavc4.alu.add %out, %byte_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %remaining = ssavc4.alu.add %logical_bytes, %byte_offset {opcode = #vc4.add_opcode<sub>} : (i32, i32) -> i32
    %p01 = ssavc4.alu.add %seed_v, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p02 = ssavc4.alu.add %p01, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p03 = ssavc4.alu.add %p02, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p04 = ssavc4.alu.add %p03, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p05 = ssavc4.alu.add %p04, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p06 = ssavc4.alu.add %p05, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p07 = ssavc4.alu.add %p06, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p08 = ssavc4.alu.add %p07, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p09 = ssavc4.alu.add %p08, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p10 = ssavc4.alu.add %p09, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p11 = ssavc4.alu.add %p10, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p12 = ssavc4.alu.add %p11, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p13 = ssavc4.alu.add %p12, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p14 = ssavc4.alu.add %p13, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p15 = ssavc4.alu.add %p14, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p16 = ssavc4.alu.add %p15, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p17 = ssavc4.alu.add %p16, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p18 = ssavc4.alu.add %p17, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p19 = ssavc4.alu.add %p18, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p20 = ssavc4.alu.add %p19, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.br ^loop(%zero, %seed_v, %beta_v : i32, vector<16xf32>, vector<16xf32>)

  ^loop(%iter: i32, %acc: vector<16xf32>, %carry: vector<16xf32>):
    %done_flags = ssavc4.make_flags %iter, %iterations {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^after_loop(%acc, %carry : vector<16xf32>, vector<16xf32>), ^step(%iter, %acc, %carry : i32, vector<16xf32>, vector<16xf32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^step(%iter_step: i32, %acc_step: vector<16xf32>, %carry_step: vector<16xf32>):
    %acc_next = ssavc4.alu.add %acc_step, %carry_step {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %carry_next = ssavc4.alu.add %carry_step, %alpha_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %iter_next = ssavc4.alu.add %iter_step, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%iter_next, %acc_next, %carry_next : i32, vector<16xf32>, vector<16xf32>)

  ^after_loop(%final_acc: vector<16xf32>, %final_carry: vector<16xf32>):
    %s00 = ssavc4.alu.add %final_acc, %final_carry {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s01 = ssavc4.alu.add %s00, %p01 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s02 = ssavc4.alu.add %s01, %p02 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s03 = ssavc4.alu.add %s02, %p03 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s04 = ssavc4.alu.add %s03, %p04 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s05 = ssavc4.alu.add %s04, %p05 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s06 = ssavc4.alu.add %s05, %p06 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s07 = ssavc4.alu.add %s06, %p07 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s08 = ssavc4.alu.add %s07, %p08 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s09 = ssavc4.alu.add %s08, %p09 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s10 = ssavc4.alu.add %s09, %p10 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s11 = ssavc4.alu.add %s10, %p11 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s12 = ssavc4.alu.add %s11, %p12 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s13 = ssavc4.alu.add %s12, %p13 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s14 = ssavc4.alu.add %s13, %p14 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s15 = ssavc4.alu.add %s14, %p15 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s16 = ssavc4.alu.add %s15, %p16 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s17 = ssavc4.alu.add %s16, %p17 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s18 = ssavc4.alu.add %s17, %p18 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s19 = ssavc4.alu.add %s18, %p19 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s20 = ssavc4.alu.add %s19, %p20 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %use_case = ssavc4.alu.add %case_v, %case_v {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %full_flags = ssavc4.make_flags %remaining, %full_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %full_flags, ^full(%s20, %use_case : vector<16xf32>, vector<16xi32>), ^tail(%s20, %use_case : vector<16xf32>, vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

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
