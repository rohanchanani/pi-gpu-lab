ssavc4.module @vdw_user_store_between_spill_copies_ssavc4 {
  ssavc4.func @vdw_user_store_between_spill_copies_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_user_store_between_spill_copies_ssavc4",
      code_symbol = "vdw_user_store_between_spill_copies_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 0 : i32},
        {name = "beta", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 1 : i32}
      ],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 2 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 3 : i32},
        {name = "vpm_base_row", kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", uniform_index = 4 : i32}
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
    %out_base = ssavc4.uniform.read 0 : i32
    %beta = ssavc4.uniform.read 1 : f32
    %qpu_id = ssavc4.uniform.read 2 : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %sixty_four = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
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
    %first01 = ssavc4.alu.add %p01, %p02 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first02 = ssavc4.alu.add %first01, %p03 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first03 = ssavc4.alu.add %first02, %p04 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first04 = ssavc4.alu.add %first03, %p05 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first05 = ssavc4.alu.add %first04, %p06 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first06 = ssavc4.alu.add %first05, %p07 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first07 = ssavc4.alu.add %first06, %p08 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first08 = ssavc4.alu.add %first07, %p09 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first09 = ssavc4.alu.add %first08, %p10 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first10 = ssavc4.alu.add %first09, %p11 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first11 = ssavc4.alu.add %first10, %p12 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first12 = ssavc4.alu.add %first11, %p13 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first13 = ssavc4.alu.add %first12, %p14 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first14 = ssavc4.alu.add %first13, %p15 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %first_value = ssavc4.alu.add %first14, %p16 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.vdw.store %out_base, %first_value, %active_full, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    %out_second = ssavc4.alu.add %out_base, %sixty_four {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %second01 = ssavc4.alu.add %p17, %p18 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second02 = ssavc4.alu.add %second01, %p19 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second03 = ssavc4.alu.add %second02, %p20 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second04 = ssavc4.alu.add %second03, %p21 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second05 = ssavc4.alu.add %second04, %p22 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second06 = ssavc4.alu.add %second05, %p23 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second07 = ssavc4.alu.add %second06, %p24 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second08 = ssavc4.alu.add %second07, %p25 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second09 = ssavc4.alu.add %second08, %p26 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second10 = ssavc4.alu.add %second09, %p27 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second11 = ssavc4.alu.add %second10, %p28 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second12 = ssavc4.alu.add %second11, %p29 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second13 = ssavc4.alu.add %second12, %p30 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second14 = ssavc4.alu.add %second13, %p31 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %second_value = ssavc4.alu.add %second14, %p32 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.vdw.store %out_second, %second_value, %active_full, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.thread_end
  }
}
