ssavc4.module @saxpy_full_ssavc4 {
  ssavc4.func @saxpy_full_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "saxpy_full_ssavc4",
      code_symbol = "saxpy_full_ssavc4_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
        {name = "y", kind = "buffer", direction = "inout", elem_type = "f32", uniform_index = 1 : i32},
        {name = "alpha", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 2 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
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
    %x_base = ssavc4.uniform.read 0 : i32
    %y_base = ssavc4.uniform.read 1 : i32
    %alpha = ssavc4.uniform.read 2 : vector<16xf32>
    %n = ssavc4.uniform.read 3 : i32
    %qpu_id = ssavc4.uniform.read 4 : i32
    %num_qpus = ssavc4.uniform.read 5 : i32

    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_two_vec = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %full_bytes = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>

    %byte_offset = ssavc4.alu.add %qpu_id, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %logical_bytes = ssavc4.alu.add %n, %shift_two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %skip_flags = ssavc4.make_flags %byte_offset, %logical_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %skip_flags, ^done, ^body {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^body:
    %x_chunk = ssavc4.alu.add %x_base, %byte_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %y_chunk = ssavc4.alu.add %y_base, %byte_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %remaining = ssavc4.alu.add %logical_bytes, %byte_offset {opcode = #vc4.add_opcode<sub>} : (i32, i32) -> i32
    %full_flags = ssavc4.make_flags %remaining, %full_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    %lane_bytes = ssavc4.alu.add %lane, %shift_two_vec {opcode = #vc4.add_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %x_base_vec = ssavc4.splat %x_chunk : i32 -> vector<16xi32>
    %y_base_vec = ssavc4.splat %y_chunk : i32 -> vector<16xi32>
    %x_addr = ssavc4.alu.add %x_base_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %y_addr = ssavc4.alu.add %y_base_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %x_tok = ssavc4.tmu.request %x_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %x = ssavc4.tmu.read %x_tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    %y_tok = ssavc4.tmu.request %y_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %y = ssavc4.tmu.read %y_tok {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xf32>
    %scaled = ssavc4.alu.mul %alpha, %x {opcode = #vc4.mul_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %sum = ssavc4.alu.add %scaled, %y {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.cond_br %full_flags, ^full, ^tail {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^tail:
    %active = ssavc4.alu.add %remaining, %shift_two {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
    ssavc4.vdw.store %y_chunk, %sum, %active, %qpu_id {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.br ^done

  ^full:
    ssavc4.vdw.store %y_chunk, %sum, %active_full, %qpu_id {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xf32>, i32, i32
    ssavc4.br ^done

  ^done:
    ssavc4.thread_end
  }
}
