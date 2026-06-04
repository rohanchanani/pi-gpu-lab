ssavc4.module @loop_branch_pingpong_vector_args_forced_spill_ssavc4 {
  ssavc4.func @loop_branch_pingpong_vector_args_forced_spill_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "loop_branch_pingpong_vector_args_forced_spill_ssavc4",
      code_symbol = "loop_branch_pingpong_vector_args_forced_spill_ssavc4_shader",
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
    %branch0_bonus = ssavc4.load_imm <splat32> {value = 10 : i32} : i32
    %branch1_bonus = ssavc4.load_imm <splat32> {value = 20 : i32} : i32
    %k01 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %k02 = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %k03 = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %k04 = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %k05 = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %k06 = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %k07 = ssavc4.load_imm <splat32> {value = 7 : i32} : i32
    %k08 = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %k09 = ssavc4.load_imm <splat32> {value = 9 : i32} : i32
    %k10 = ssavc4.load_imm <splat32> {value = 10 : i32} : i32
    %k11 = ssavc4.load_imm <splat32> {value = 11 : i32} : i32
    %k12 = ssavc4.load_imm <splat32> {value = 12 : i32} : i32
    %k13 = ssavc4.load_imm <splat32> {value = 13 : i32} : i32
    %k14 = ssavc4.load_imm <splat32> {value = 14 : i32} : i32
    %k15 = ssavc4.load_imm <splat32> {value = 15 : i32} : i32
    %k16 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %k17 = ssavc4.load_imm <splat32> {value = 17 : i32} : i32
    %k18 = ssavc4.load_imm <splat32> {value = 18 : i32} : i32
    %k19 = ssavc4.load_imm <splat32> {value = 19 : i32} : i32
    %k20 = ssavc4.load_imm <splat32> {value = 20 : i32} : i32
    %k21 = ssavc4.load_imm <splat32> {value = 21 : i32} : i32
    %k22 = ssavc4.load_imm <splat32> {value = 22 : i32} : i32
    %k23 = ssavc4.load_imm <splat32> {value = 23 : i32} : i32
    %k24 = ssavc4.load_imm <splat32> {value = 24 : i32} : i32
    %k25 = ssavc4.load_imm <splat32> {value = 25 : i32} : i32
    %k26 = ssavc4.load_imm <splat32> {value = 26 : i32} : i32
    %k27 = ssavc4.load_imm <splat32> {value = 27 : i32} : i32
    %k28 = ssavc4.load_imm <splat32> {value = 28 : i32} : i32
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_four = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %shift_eight = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %full_bytes = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %v01 = ssavc4.splat %k01 : i32 -> vector<16xi32>
    %v02 = ssavc4.splat %k02 : i32 -> vector<16xi32>
    %v03 = ssavc4.splat %k03 : i32 -> vector<16xi32>
    %v04 = ssavc4.splat %k04 : i32 -> vector<16xi32>
    %v05 = ssavc4.splat %k05 : i32 -> vector<16xi32>
    %v06 = ssavc4.splat %k06 : i32 -> vector<16xi32>
    %v07 = ssavc4.splat %k07 : i32 -> vector<16xi32>
    %v08 = ssavc4.splat %k08 : i32 -> vector<16xi32>
    %v09 = ssavc4.splat %k09 : i32 -> vector<16xi32>
    %v10 = ssavc4.splat %k10 : i32 -> vector<16xi32>
    %v11 = ssavc4.splat %k11 : i32 -> vector<16xi32>
    %v12 = ssavc4.splat %k12 : i32 -> vector<16xi32>
    %v13 = ssavc4.splat %k13 : i32 -> vector<16xi32>
    %v14 = ssavc4.splat %k14 : i32 -> vector<16xi32>
    %v15 = ssavc4.splat %k15 : i32 -> vector<16xi32>
    %v16 = ssavc4.splat %k16 : i32 -> vector<16xi32>
    %v17 = ssavc4.splat %k17 : i32 -> vector<16xi32>
    %v18 = ssavc4.splat %k18 : i32 -> vector<16xi32>
    %v19 = ssavc4.splat %k19 : i32 -> vector<16xi32>
    %v20 = ssavc4.splat %k20 : i32 -> vector<16xi32>
    %v21 = ssavc4.splat %k21 : i32 -> vector<16xi32>
    %v22 = ssavc4.splat %k22 : i32 -> vector<16xi32>
    %v23 = ssavc4.splat %k23 : i32 -> vector<16xi32>
    %v24 = ssavc4.splat %k24 : i32 -> vector<16xi32>
    %v25 = ssavc4.splat %k25 : i32 -> vector<16xi32>
    %v26 = ssavc4.splat %k26 : i32 -> vector<16xi32>
    %v27 = ssavc4.splat %k27 : i32 -> vector<16xi32>
    %v28 = ssavc4.splat %k28 : i32 -> vector<16xi32>

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
    ssavc4.cond_br %done_flags, ^after_loop(%acc : vector<16xi32>), ^select(%iter, %acc : i32, vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^select(%iter_select: i32, %acc_select: vector<16xi32>):
    %first_flags = ssavc4.make_flags %iter_select, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %first_flags, ^first_iter(%iter_select, %acc_select : i32, vector<16xi32>), ^later_iter(%iter_select, %acc_select : i32, vector<16xi32>) {cond = #vc4.branch_cond<any_z_set>} : !ssavc4.flags

  ^first_iter(%iter_first: i32, %acc_first: vector<16xi32>):
    %iter_vec0 = ssavc4.splat %iter_first : i32 -> vector<16xi32>
    %bonus_vec0 = ssavc4.splat %branch0_bonus : i32 -> vector<16xi32>
    %lane_iter0 = ssavc4.alu.add %lane, %iter_vec0 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %term_base0 = ssavc4.alu.add %lane_iter0, %case_step_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %term0 = ssavc4.alu.add %term_base0, %bonus_vec0 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %next0 = ssavc4.alu.add %acc_first, %term0 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^merge(%iter_first, %next0 : i32, vector<16xi32>)

  ^later_iter(%iter_later: i32, %acc_later: vector<16xi32>):
    %iter_vec1 = ssavc4.splat %iter_later : i32 -> vector<16xi32>
    %bonus_vec1 = ssavc4.splat %branch1_bonus : i32 -> vector<16xi32>
    %lane_iter1 = ssavc4.alu.add %lane, %iter_vec1 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %term_base1 = ssavc4.alu.add %lane_iter1, %case_step_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %term1 = ssavc4.alu.add %term_base1, %bonus_vec1 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %next1 = ssavc4.alu.add %acc_later, %term1 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^merge(%iter_later, %next1 : i32, vector<16xi32>)

  ^merge(%iter_merge: i32, %acc_next: vector<16xi32>):
    %iter_next = ssavc4.alu.add %iter_merge, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%iter_next, %acc_next : i32, vector<16xi32>)

  ^after_loop(%final: vector<16xi32>):
    %s01 = ssavc4.alu.add %final, %v01 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s02 = ssavc4.alu.add %s01, %v02 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s03 = ssavc4.alu.add %s02, %v03 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s04 = ssavc4.alu.add %s03, %v04 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s05 = ssavc4.alu.add %s04, %v05 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s06 = ssavc4.alu.add %s05, %v06 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s07 = ssavc4.alu.add %s06, %v07 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s08 = ssavc4.alu.add %s07, %v08 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s09 = ssavc4.alu.add %s08, %v09 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s10 = ssavc4.alu.add %s09, %v10 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s11 = ssavc4.alu.add %s10, %v11 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s12 = ssavc4.alu.add %s11, %v12 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s13 = ssavc4.alu.add %s12, %v13 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s14 = ssavc4.alu.add %s13, %v14 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s15 = ssavc4.alu.add %s14, %v15 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s16 = ssavc4.alu.add %s15, %v16 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s17 = ssavc4.alu.add %s16, %v17 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s18 = ssavc4.alu.add %s17, %v18 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s19 = ssavc4.alu.add %s18, %v19 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s20 = ssavc4.alu.add %s19, %v20 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s21 = ssavc4.alu.add %s20, %v21 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s22 = ssavc4.alu.add %s21, %v22 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s23 = ssavc4.alu.add %s22, %v23 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s24 = ssavc4.alu.add %s23, %v24 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s25 = ssavc4.alu.add %s24, %v25 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s26 = ssavc4.alu.add %s25, %v26 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s27 = ssavc4.alu.add %s26, %v27 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %adjusted = ssavc4.alu.add %s27, %v28 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %full_flags = ssavc4.make_flags %remaining, %full_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %full_flags, ^full(%adjusted : vector<16xi32>), ^tail(%adjusted : vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^tail(%tail_value: vector<16xi32>):
    %active = ssavc4.alu.add %remaining, %shift_two {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr, %tail_value, %active, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32
    ssavc4.br ^done

  ^full(%full_value: vector<16xi32>):
    ssavc4.vdw.store %addr, %full_value, %active_full, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32
    ssavc4.br ^done

  ^done:
    ssavc4.thread_end
  }
}
