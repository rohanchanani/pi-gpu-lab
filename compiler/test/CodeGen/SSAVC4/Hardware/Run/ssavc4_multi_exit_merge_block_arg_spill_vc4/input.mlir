ssavc4.module @ssavc4_multi_exit_merge_block_arg_spill_vc4 {
  ssavc4.func @ssavc4_multi_exit_merge_block_arg_spill_vc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "ssavc4_multi_exit_merge_block_arg_spill_vc4",
      code_symbol = "ssavc4_multi_exit_merge_block_arg_spill_vc4_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "selector", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
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
    %selector = ssavc4.uniform.read 2 : i32
    %qpu_id = ssavc4.uniform.read 3 : i32

    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_four = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %full_bytes = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %done_bias = ssavc4.load_imm <splat32> {value = 1000 : i32} : vector<16xi32>
    %early_bias = ssavc4.load_imm <splat32> {value = 2000 : i32} : vector<16xi32>
    %p04 = ssavc4.load_imm <splat32> {value = 4 : i32} : vector<16xi32>
    %p05 = ssavc4.load_imm <splat32> {value = 5 : i32} : vector<16xi32>
    %p06 = ssavc4.load_imm <splat32> {value = 6 : i32} : vector<16xi32>
    %p07 = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    %p08 = ssavc4.load_imm <splat32> {value = 8 : i32} : vector<16xi32>
    %p09 = ssavc4.load_imm <splat32> {value = 9 : i32} : vector<16xi32>
    %p10 = ssavc4.load_imm <splat32> {value = 10 : i32} : vector<16xi32>
    %p11 = ssavc4.load_imm <splat32> {value = 11 : i32} : vector<16xi32>
    %p12 = ssavc4.load_imm <splat32> {value = 12 : i32} : vector<16xi32>
    %p13 = ssavc4.load_imm <splat32> {value = 13 : i32} : vector<16xi32>
    %p14 = ssavc4.load_imm <splat32> {value = 14 : i32} : vector<16xi32>
    %p15 = ssavc4.load_imm <splat32> {value = 15 : i32} : vector<16xi32>
    %p16 = ssavc4.load_imm <splat32> {value = 16 : i32} : vector<16xi32>
    %p17 = ssavc4.load_imm <splat32> {value = 17 : i32} : vector<16xi32>
    %p18 = ssavc4.load_imm <splat32> {value = 18 : i32} : vector<16xi32>
    %p19 = ssavc4.load_imm <splat32> {value = 19 : i32} : vector<16xi32>
    %p20 = ssavc4.load_imm <splat32> {value = 20 : i32} : vector<16xi32>
    %p21 = ssavc4.load_imm <splat32> {value = 21 : i32} : vector<16xi32>
    %p22 = ssavc4.load_imm <splat32> {value = 22 : i32} : vector<16xi32>
    %p23 = ssavc4.load_imm <splat32> {value = 23 : i32} : vector<16xi32>
    %p24 = ssavc4.load_imm <splat32> {value = 24 : i32} : vector<16xi32>
    %p25 = ssavc4.load_imm <splat32> {value = 25 : i32} : vector<16xi32>
    %p26 = ssavc4.load_imm <splat32> {value = 26 : i32} : vector<16xi32>
    %p27 = ssavc4.load_imm <splat32> {value = 27 : i32} : vector<16xi32>
    %p28 = ssavc4.load_imm <splat32> {value = 28 : i32} : vector<16xi32>
    %p29 = ssavc4.load_imm <splat32> {value = 29 : i32} : vector<16xi32>
    %p30 = ssavc4.load_imm <splat32> {value = 30 : i32} : vector<16xi32>
    %p31 = ssavc4.load_imm <splat32> {value = 31 : i32} : vector<16xi32>
    %p32 = ssavc4.load_imm <splat32> {value = 32 : i32} : vector<16xi32>
    %p33 = ssavc4.load_imm <splat32> {value = 33 : i32} : vector<16xi32>
    %p34 = ssavc4.load_imm <splat32> {value = 34 : i32} : vector<16xi32>
    %p35 = ssavc4.load_imm <splat32> {value = 35 : i32} : vector<16xi32>
    %p36 = ssavc4.load_imm <splat32> {value = 36 : i32} : vector<16xi32>
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
    %done_seed = ssavc4.alu.add %index, %done_bias {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %early_seed = ssavc4.alu.add %index, %early_bias {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %select_flags = ssavc4.make_flags %selector, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %select_flags, ^exit_done(%done_seed : vector<16xi32>), ^exit_early(%early_seed : vector<16xi32>) {cond = #vc4.branch_cond<any_z_set>} : !ssavc4.flags

  ^exit_done(%done_in: vector<16xi32>):
    %done_value = ssavc4.alu.add %done_in, %p04 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^merge(%done_value : vector<16xi32>)

  ^exit_early(%early_in: vector<16xi32>):
    %early_value = ssavc4.alu.add %early_in, %p05 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^merge(%early_value : vector<16xi32>)

  ^merge(%merged: vector<16xi32>):
    %s06 = ssavc4.alu.add %merged, %p06 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s07 = ssavc4.alu.add %s06, %p07 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s08 = ssavc4.alu.add %s07, %p08 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s09 = ssavc4.alu.add %s08, %p09 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s10 = ssavc4.alu.add %s09, %p10 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s11 = ssavc4.alu.add %s10, %p11 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s12 = ssavc4.alu.add %s11, %p12 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s13 = ssavc4.alu.add %s12, %p13 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s14 = ssavc4.alu.add %s13, %p14 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s15 = ssavc4.alu.add %s14, %p15 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s16 = ssavc4.alu.add %s15, %p16 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s17 = ssavc4.alu.add %s16, %p17 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s18 = ssavc4.alu.add %s17, %p18 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s19 = ssavc4.alu.add %s18, %p19 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s20 = ssavc4.alu.add %s19, %p20 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s21 = ssavc4.alu.add %s20, %p21 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s22 = ssavc4.alu.add %s21, %p22 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s23 = ssavc4.alu.add %s22, %p23 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s24 = ssavc4.alu.add %s23, %p24 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s25 = ssavc4.alu.add %s24, %p25 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s26 = ssavc4.alu.add %s25, %p26 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s27 = ssavc4.alu.add %s26, %p27 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s28 = ssavc4.alu.add %s27, %p28 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s29 = ssavc4.alu.add %s28, %p29 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s30 = ssavc4.alu.add %s29, %p30 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s31 = ssavc4.alu.add %s30, %p31 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s32 = ssavc4.alu.add %s31, %p32 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s33 = ssavc4.alu.add %s32, %p33 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s34 = ssavc4.alu.add %s33, %p34 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s35 = ssavc4.alu.add %s34, %p35 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %final = ssavc4.alu.add %s35, %p36 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %full_flags = ssavc4.make_flags %remaining, %full_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %full_flags, ^full(%final : vector<16xi32>), ^tail(%final : vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

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
