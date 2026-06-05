ssavc4.module @spill_reload_between_user_tmu_reads_ssavc4 {
  ssavc4.func @spill_reload_between_user_tmu_reads_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "spill_reload_between_user_tmu_reads_ssavc4",
      code_symbol = "spill_reload_between_user_tmu_reads_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 7 : i32,
      args = [
        {name = "a", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
        {name = "b", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 2 : i32},
        {name = "beta", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 3 : i32}
      ],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 4 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 5 : i32},
        {name = "vpm_base_row", kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", uniform_index = 6 : i32}
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
    %beta = ssavc4.uniform.read 3 : f32
    %qpu_id = ssavc4.uniform.read 4 : i32
    %shift_two_vec = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %lane_bytes = ssavc4.alu.add %lane, %shift_two_vec {opcode = #vc4.add_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a_base_v = ssavc4.splat %a_base : i32 -> vector<16xi32>
    %b_base_v = ssavc4.splat %b_base : i32 -> vector<16xi32>
    %a_addr = ssavc4.alu.add %a_base_v, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b_addr = ssavc4.alu.add %b_base_v, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %beta_v = ssavc4.splat %beta : f32 -> vector<16xf32>
    %p01 = ssavc4.alu.add %beta_v, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
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
    %p21 = ssavc4.alu.add %p20, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p22 = ssavc4.alu.add %p21, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p23 = ssavc4.alu.add %p22, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p24 = ssavc4.alu.add %p23, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p25 = ssavc4.alu.add %p24, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p26 = ssavc4.alu.add %p25, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p27 = ssavc4.alu.add %p26, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p28 = ssavc4.alu.add %p27, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p29 = ssavc4.alu.add %p28, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p30 = ssavc4.alu.add %p29, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p31 = ssavc4.alu.add %p30, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p32 = ssavc4.alu.add %p31, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p33 = ssavc4.alu.add %p32, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p34 = ssavc4.alu.add %p33, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p35 = ssavc4.alu.add %p34, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p36 = ssavc4.alu.add %p35, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p37 = ssavc4.alu.add %p36, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p38 = ssavc4.alu.add %p37, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p39 = ssavc4.alu.add %p38, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p40 = ssavc4.alu.add %p39, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p41 = ssavc4.alu.add %p40, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p42 = ssavc4.alu.add %p41, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p43 = ssavc4.alu.add %p42, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p44 = ssavc4.alu.add %p43, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p45 = ssavc4.alu.add %p44, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p46 = ssavc4.alu.add %p45, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p47 = ssavc4.alu.add %p46, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p48 = ssavc4.alu.add %p47, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %a_tok = ssavc4.tmu.request %a_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %a_val = ssavc4.tmu.read %a_tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    %mid = ssavc4.alu.add %a_val, %p01 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s02 = ssavc4.alu.add %mid, %p02 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
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
    %s21 = ssavc4.alu.add %s20, %p21 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s22 = ssavc4.alu.add %s21, %p22 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s23 = ssavc4.alu.add %s22, %p23 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s24 = ssavc4.alu.add %s23, %p24 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %b_tok = ssavc4.tmu.request %b_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %b_val = ssavc4.tmu.read %b_tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    %with_b = ssavc4.alu.add %s24, %b_val {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s25 = ssavc4.alu.add %with_b, %p25 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s26 = ssavc4.alu.add %s25, %p26 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s27 = ssavc4.alu.add %s26, %p27 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s28 = ssavc4.alu.add %s27, %p28 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s29 = ssavc4.alu.add %s28, %p29 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s30 = ssavc4.alu.add %s29, %p30 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s31 = ssavc4.alu.add %s30, %p31 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s32 = ssavc4.alu.add %s31, %p32 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s33 = ssavc4.alu.add %s32, %p33 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s34 = ssavc4.alu.add %s33, %p34 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s35 = ssavc4.alu.add %s34, %p35 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s36 = ssavc4.alu.add %s35, %p36 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s37 = ssavc4.alu.add %s36, %p37 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s38 = ssavc4.alu.add %s37, %p38 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s39 = ssavc4.alu.add %s38, %p39 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s40 = ssavc4.alu.add %s39, %p40 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s41 = ssavc4.alu.add %s40, %p41 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s42 = ssavc4.alu.add %s41, %p42 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s43 = ssavc4.alu.add %s42, %p43 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s44 = ssavc4.alu.add %s43, %p44 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s45 = ssavc4.alu.add %s44, %p45 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s46 = ssavc4.alu.add %s45, %p46 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s47 = ssavc4.alu.add %s46, %p47 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s48 = ssavc4.alu.add %s47, %p48 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.vdw.store %out_base, %s48, %active_full, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.thread_end
  }
}
