ssavc4.module @scalar_f32_uniform_order_canary_ssavc4 {
  ssavc4.func @scalar_f32_uniform_order_canary_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "scalar_f32_uniform_order_canary_ssavc4",
      code_symbol = "scalar_f32_uniform_order_canary_ssavc4_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "f32", uniform_index = 0 : i32},
        {name = "alpha", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 1 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "beta", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 3 : i32}
      ],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 4 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 5 : i32}
      ],
      work_distribution = {
        base_element = "qpu_id * 16",
        stride_elements = "num_qpus * 16",
        tail_store = "dynamic_vdw_depth"
      }
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
    %alpha = ssavc4.uniform.read 1 : f32
    %n = ssavc4.uniform.read 2 : i32
    %beta = ssavc4.uniform.read 3 : f32
    %qpu_id = ssavc4.uniform.read 4 : i32
    %num_qpus = ssavc4.uniform.read 5 : i32

    %alpha_v = ssavc4.splat %alpha : f32 -> vector<16xf32>
    %beta_v = ssavc4.splat %beta : f32 -> vector<16xf32>
    %three = ssavc4.load_imm <splat32> {value = 1077936128 : i32} : vector<16xf32>
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %full_bytes = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    %byte_offset = ssavc4.alu.add %qpu_id, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %logical_bytes = ssavc4.alu.add %n, %shift_two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %skip_flags = ssavc4.make_flags %byte_offset, %logical_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %skip_flags, ^done, ^body {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^body:
    %out_chunk = ssavc4.alu.add %out_base, %byte_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %remaining = ssavc4.alu.add %logical_bytes, %byte_offset {opcode = #vc4.add_opcode<sub>} : (i32, i32) -> i32
    %full_flags = ssavc4.make_flags %remaining, %full_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    %scaled = ssavc4.alu.mul %alpha_v, %three {opcode = #vc4.mul_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %sum = ssavc4.alu.add %scaled, %beta_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.cond_br %full_flags, ^full, ^tail {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^tail:
    %active = ssavc4.alu.add %remaining, %shift_two {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
    ssavc4.vdw.store %out_chunk, %sum, %active, %qpu_id {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.br ^done

  ^full:
    ssavc4.vdw.store %out_chunk, %sum, %active_full, %qpu_id {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.br ^done

  ^done:
    ssavc4.thread_end
  }
}
