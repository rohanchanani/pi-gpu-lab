ssavc4.module @mixed_lower_half_spill_dma_branch_ssavc4 {
  ssavc4.func @mixed_lower_half_spill_dma_branch_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "mixed_lower_half_spill_dma_branch_ssavc4",
      code_symbol = "mixed_lower_half_spill_dma_branch_ssavc4_shader",
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
    %v01 = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : vector<16xf32>
    %v02 = ssavc4.load_imm <splat32> {value = 2.000000e+00 : f32} : vector<16xf32>
    %v03 = ssavc4.load_imm <splat32> {value = 3.000000e+00 : f32} : vector<16xf32>
    %v04 = ssavc4.load_imm <splat32> {value = 4.000000e+00 : f32} : vector<16xf32>
    %v05 = ssavc4.load_imm <splat32> {value = 5.000000e+00 : f32} : vector<16xf32>
    %v06 = ssavc4.load_imm <splat32> {value = 6.000000e+00 : f32} : vector<16xf32>
    %v07 = ssavc4.load_imm <splat32> {value = 7.000000e+00 : f32} : vector<16xf32>
    %v08 = ssavc4.load_imm <splat32> {value = 8.000000e+00 : f32} : vector<16xf32>
    %v09 = ssavc4.load_imm <splat32> {value = 9.000000e+00 : f32} : vector<16xf32>
    %v10 = ssavc4.load_imm <splat32> {value = 1.000000e+01 : f32} : vector<16xf32>
    %v11 = ssavc4.load_imm <splat32> {value = 1.100000e+01 : f32} : vector<16xf32>
    %v12 = ssavc4.load_imm <splat32> {value = 1.200000e+01 : f32} : vector<16xf32>
    %v13 = ssavc4.load_imm <splat32> {value = 1.300000e+01 : f32} : vector<16xf32>
    %v14 = ssavc4.load_imm <splat32> {value = 1.400000e+01 : f32} : vector<16xf32>
    %v15 = ssavc4.load_imm <splat32> {value = 1.500000e+01 : f32} : vector<16xf32>
    %v16 = ssavc4.load_imm <splat32> {value = 1.600000e+01 : f32} : vector<16xf32>
    %v17 = ssavc4.load_imm <splat32> {value = 1.700000e+01 : f32} : vector<16xf32>
    %v18 = ssavc4.load_imm <splat32> {value = 1.800000e+01 : f32} : vector<16xf32>
    %v19 = ssavc4.load_imm <splat32> {value = 1.900000e+01 : f32} : vector<16xf32>
    %v20 = ssavc4.load_imm <splat32> {value = 2.000000e+01 : f32} : vector<16xf32>
    %v21 = ssavc4.load_imm <splat32> {value = 2.100000e+01 : f32} : vector<16xf32>
    %v22 = ssavc4.load_imm <splat32> {value = 2.200000e+01 : f32} : vector<16xf32>
    %v23 = ssavc4.load_imm <splat32> {value = 2.300000e+01 : f32} : vector<16xf32>
    %v24 = ssavc4.load_imm <splat32> {value = 2.400000e+01 : f32} : vector<16xf32>
    %v25 = ssavc4.load_imm <splat32> {value = 2.500000e+01 : f32} : vector<16xf32>
    %v26 = ssavc4.load_imm <splat32> {value = 2.600000e+01 : f32} : vector<16xf32>
    %v27 = ssavc4.load_imm <splat32> {value = 2.700000e+01 : f32} : vector<16xf32>
    %v28 = ssavc4.load_imm <splat32> {value = 2.800000e+01 : f32} : vector<16xf32>
    %lane = ssavc4.element_number : vector<16xi32>
    %active_vec = ssavc4.splat %active_cols : i32 -> vector<16xi32>
    ssavc4.br ^loop(%zero_i, %seed_v : i32, vector<16xf32>)

  ^loop(%iter: i32, %acc: vector<16xf32>):
    %done_flags = ssavc4.make_flags %iter, %iters {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^after_loop(%acc : vector<16xf32>), ^step(%iter, %acc : i32, vector<16xf32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^step(%iter_step: i32, %acc_step: vector<16xf32>):
    %src_offset = ssavc4.alu.mul %iter_step, %pitch {opcode = #vc4.mul_opcode<mul24>} : (i32, i32) -> i32
    %src = ssavc4.alu.add %in, %src_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vdr.load %src, %row0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, vpm_x = 0 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32, serialize = "mutex"} : i32, i32
    %read = ssavc4.vpm.read %row0 {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, serialize = "mutex"} : i32 -> vector<16xf32>
    %tail_flags = ssavc4.make_flags %lane, %active_vec {kind = #ssavc4.flag_kind<sub>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
    %masked = ssavc4.cond_select %tail_flags, %read, %zero_v {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %acc_next = ssavc4.alu.add %acc_step, %masked {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %iter_next = ssavc4.alu.add %iter_step, %one_i {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%iter_next, %acc_next : i32, vector<16xf32>)

  ^after_loop(%final: vector<16xf32>):
    %s01 = ssavc4.alu.add %final, %v01 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s02 = ssavc4.alu.add %s01, %v02 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s03 = ssavc4.alu.add %s02, %v03 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s04 = ssavc4.alu.add %s03, %v04 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s05 = ssavc4.alu.add %s04, %v05 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s06 = ssavc4.alu.add %s05, %v06 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s07 = ssavc4.alu.add %s06, %v07 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s08 = ssavc4.alu.add %s07, %v08 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s09 = ssavc4.alu.add %s08, %v09 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s10 = ssavc4.alu.add %s09, %v10 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s11 = ssavc4.alu.add %s10, %v11 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s12 = ssavc4.alu.add %s11, %v12 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s13 = ssavc4.alu.add %s12, %v13 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s14 = ssavc4.alu.add %s13, %v14 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s15 = ssavc4.alu.add %s14, %v15 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s16 = ssavc4.alu.add %s15, %v16 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s17 = ssavc4.alu.add %s16, %v17 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s18 = ssavc4.alu.add %s17, %v18 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s19 = ssavc4.alu.add %s18, %v19 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s20 = ssavc4.alu.add %s19, %v20 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s21 = ssavc4.alu.add %s20, %v21 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s22 = ssavc4.alu.add %s21, %v22 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s23 = ssavc4.alu.add %s22, %v23 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s24 = ssavc4.alu.add %s23, %v24 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s25 = ssavc4.alu.add %s24, %v25 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s26 = ssavc4.alu.add %s25, %v26 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s27 = ssavc4.alu.add %s26, %v27 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %adjusted = ssavc4.alu.add %s27, %v28 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.vdw.store %out, %adjusted, %active_full, %qpu_id {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.thread_end
  }
}
