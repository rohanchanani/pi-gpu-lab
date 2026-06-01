ssavc4.module @resource_limited_semaphore_residency_ssavc4 {
  ssavc4.func @resource_limited_semaphore_residency_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "resource_limited_semaphore_residency_ssavc4",
      code_symbol = "resource_limited_semaphore_residency_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 7 : i32,
args = [
        {name = "input", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "output", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32}
      ],
      builtins = [
        {name = "logical_warp_id", kind = #vc4.builtin_kind<logical_warp_id>, materialization = "uniform_suffix", uniform_index = 5 : i32},
        {name = "warps_per_block", kind = #vc4.builtin_kind<warps_per_block>, materialization = "uniform_suffix", uniform_index = 6 : i32},
        {name = "logical_warp_id", kind = #vc4.builtin_kind<logical_warp_id>, materialization = "uniform_suffix", uniform_index = 2 : i32},
        {name = "warps_per_block", kind = #vc4.builtin_kind<warps_per_block>, materialization = "uniform_suffix", uniform_index = 3 : i32},
        {name = "vpm_base_row", kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", uniform_index = 4 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 4 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = true,
      uses_vpm = true,
      uses_vpm_qpu_read = true,
      uses_vpm_qpu_write = true,
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = true,
      semaphore_count_per_block = 8 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
    }
  } {
    %input_base = ssavc4.uniform.read 0 : i32
    %out_base = ssavc4.uniform.read 1 : i32
    %logical_warp_id = ssavc4.uniform.read 2 : i32
    %warps_per_block = ssavc4.uniform.read 3 : i32
    %vpm_base_row = ssavc4.uniform.read 4 : i32
    %qpu_id = ssavc4.uniform.read 5 : i32
    %num_qpus = ssavc4.uniform.read 6 : i32

    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %three = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %store_row = ssavc4.load_imm <splat32> {value = 63 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %lane_bytes = ssavc4.alu.add %lane, %shift_two {opcode = #vc4.add_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %row_base = ssavc4.alu.add %logical_warp_id, %two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32

    %row0 = ssavc4.mov %row_base : i32 -> i32
    %row1 = ssavc4.alu.add %row_base, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row2 = ssavc4.alu.add %row_base, %two {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row3 = ssavc4.alu.add %row_base, %three {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32

    %row0_bytes = ssavc4.alu.add %row0, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %row0_input = ssavc4.alu.add %input_base, %row0_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row0_input_vec = ssavc4.splat %row0_input : i32 -> vector<16xi32>
    %row0_addr = ssavc4.alu.add %row0_input_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tok0 = ssavc4.tmu.request %row0_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %tile0 = ssavc4.tmu.read %tok0 {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    %vpm_row0 = ssavc4.alu.add %vpm_base_row, %row0 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vpm.write %vpm_row0, %tile0 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, vector<16xi32>

    %row1_bytes = ssavc4.alu.add %row1, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %row1_input = ssavc4.alu.add %input_base, %row1_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row1_input_vec = ssavc4.splat %row1_input : i32 -> vector<16xi32>
    %row1_addr = ssavc4.alu.add %row1_input_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tok1 = ssavc4.tmu.request %row1_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %tile1 = ssavc4.tmu.read %tok1 {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    %vpm_row1 = ssavc4.alu.add %vpm_base_row, %row1 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vpm.write %vpm_row1, %tile1 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, vector<16xi32>

    %row2_bytes = ssavc4.alu.add %row2, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %row2_input = ssavc4.alu.add %input_base, %row2_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row2_input_vec = ssavc4.splat %row2_input : i32 -> vector<16xi32>
    %row2_addr = ssavc4.alu.add %row2_input_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tok2 = ssavc4.tmu.request %row2_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %tile2 = ssavc4.tmu.read %tok2 {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    %vpm_row2 = ssavc4.alu.add %vpm_base_row, %row2 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vpm.write %vpm_row2, %tile2 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, vector<16xi32>

    %row3_bytes = ssavc4.alu.add %row3, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %row3_input = ssavc4.alu.add %input_base, %row3_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %row3_input_vec = ssavc4.splat %row3_input : i32 -> vector<16xi32>
    %row3_addr = ssavc4.alu.add %row3_input_vec, %lane_bytes {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tok3 = ssavc4.tmu.request %row3_addr {unit = "tmu0", mode = "direct"} : vector<16xi32> -> !ssavc4.async.token
    %tile3 = ssavc4.tmu.read %tok3 {unit = "tmu0", part = "raw32"} : !ssavc4.async.token -> vector<16xi32>
    %vpm_row3 = ssavc4.alu.add %vpm_base_row, %row3 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vpm.write %vpm_row3, %tile3 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, vector<16xi32>

    ssavc4.barrier %logical_warp_id, %warps_per_block : i32, i32 {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}

    %out0 = ssavc4.alu.add %out_base, %row0_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %read0 = ssavc4.vpm.read %vpm_row0 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical", serialize = "mutex"} : i32 -> vector<16xi32>
    ssavc4.vdw.store %out0, %read0, %active_full, %store_row {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32

    %out1 = ssavc4.alu.add %out_base, %row1_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %read1 = ssavc4.vpm.read %vpm_row1 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical", serialize = "mutex"} : i32 -> vector<16xi32>
    ssavc4.vdw.store %out1, %read1, %active_full, %store_row {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32

    %out2 = ssavc4.alu.add %out_base, %row2_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %read2 = ssavc4.vpm.read %vpm_row2 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical", serialize = "mutex"} : i32 -> vector<16xi32>
    ssavc4.vdw.store %out2, %read2, %active_full, %store_row {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32

    %out3 = ssavc4.alu.add %out_base, %row3_bytes {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %read3 = ssavc4.vpm.read %vpm_row3 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical", serialize = "mutex"} : i32 -> vector<16xi32>
    ssavc4.vdw.store %out3, %read3, %active_full, %store_row {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32

    ssavc4.thread_end
  }
}
