
ssavc4.module @spill_cooperative_vpm_smoke_ssavc4 {
  ssavc4.func @spill_cooperative_vpm_smoke_ssavc4_smoke_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "spill_cooperative_vpm_smoke_ssavc4",
      code_symbol = "spill_cooperative_vpm_smoke_ssavc4_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 5 : i32,
args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32}
      ],
      builtins = [
        {name = "logical_warp_id", kind = #vc4.builtin_kind<logical_warp_id>, materialization = "uniform_suffix", uniform_index = 4 : i32},
        {name = "logical_warp_id", kind = #vc4.builtin_kind<logical_warp_id>, materialization = "uniform_suffix", uniform_index = 1 : i32},
        {name = "warps_per_block", kind = #vc4.builtin_kind<warps_per_block>, materialization = "uniform_suffix", uniform_index = 2 : i32},
        {name = "vpm_base_row", kind = #vc4.builtin_kind<vpm_base_row>, materialization = "uniform_suffix", uniform_index = 3 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 4 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = true,
      uses_vpm_qpu_write = true,
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
    }
  } {
    %out = ssavc4.uniform.read 0 : i32
    %logical_warp_id = ssavc4.uniform.read 1 : i32
    %warps_per_block = ssavc4.uniform.read 2 : i32
    %vpm_base_row = ssavc4.uniform.read 3 : i32
    %qpu_id = ssavc4.uniform.read 4 : i32
    %shift_four = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %byte_offset = ssavc4.alu.add %qpu_id, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %addr = ssavc4.alu.add %out, %byte_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %base_elem = ssavc4.alu.add %qpu_id, %shift_four {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %base_vec = ssavc4.splat %base_elem : i32 -> vector<16xi32>
    %index = ssavc4.alu.add %base_vec, %lane {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %user_row = ssavc4.alu.add %vpm_base_row, %logical_warp_id {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.vpm.write %user_row, %index {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, vector<16xi32>
    %v01 = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
    %v02 = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    %v03 = ssavc4.load_imm <splat32> {value = 3 : i32} : vector<16xi32>
    %v04 = ssavc4.load_imm <splat32> {value = 4 : i32} : vector<16xi32>
    %v05 = ssavc4.load_imm <splat32> {value = 5 : i32} : vector<16xi32>
    %v06 = ssavc4.load_imm <splat32> {value = 6 : i32} : vector<16xi32>
    %v07 = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    %v08 = ssavc4.load_imm <splat32> {value = 8 : i32} : vector<16xi32>
    %v09 = ssavc4.load_imm <splat32> {value = 9 : i32} : vector<16xi32>
    %v10 = ssavc4.load_imm <splat32> {value = 10 : i32} : vector<16xi32>
    %v11 = ssavc4.load_imm <splat32> {value = 11 : i32} : vector<16xi32>
    %v12 = ssavc4.load_imm <splat32> {value = 12 : i32} : vector<16xi32>
    %v13 = ssavc4.load_imm <splat32> {value = 13 : i32} : vector<16xi32>
    %v14 = ssavc4.load_imm <splat32> {value = 14 : i32} : vector<16xi32>
    %v15 = ssavc4.load_imm <splat32> {value = 15 : i32} : vector<16xi32>
    %v16 = ssavc4.load_imm <splat32> {value = 16 : i32} : vector<16xi32>
    %v17 = ssavc4.load_imm <splat32> {value = 17 : i32} : vector<16xi32>
    %v18 = ssavc4.load_imm <splat32> {value = 18 : i32} : vector<16xi32>
    %v19 = ssavc4.load_imm <splat32> {value = 19 : i32} : vector<16xi32>
    %v20 = ssavc4.load_imm <splat32> {value = 20 : i32} : vector<16xi32>
    %v21 = ssavc4.load_imm <splat32> {value = 21 : i32} : vector<16xi32>
    %v22 = ssavc4.load_imm <splat32> {value = 22 : i32} : vector<16xi32>
    %v23 = ssavc4.load_imm <splat32> {value = 23 : i32} : vector<16xi32>
    %v24 = ssavc4.load_imm <splat32> {value = 24 : i32} : vector<16xi32>
    %v25 = ssavc4.load_imm <splat32> {value = 25 : i32} : vector<16xi32>
    %v26 = ssavc4.load_imm <splat32> {value = 26 : i32} : vector<16xi32>
    %v27 = ssavc4.load_imm <splat32> {value = 27 : i32} : vector<16xi32>
    %v28 = ssavc4.load_imm <splat32> {value = 28 : i32} : vector<16xi32>
    %s01 = ssavc4.alu.add %index, %v01 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
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
    %value = ssavc4.alu.add %s27, %v28 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.barrier %logical_warp_id, %warps_per_block : i32, i32 {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}
    %shared = ssavc4.vpm.read %user_row {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32 -> vector<16xi32>
    %final = ssavc4.alu.add %value, %shared {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.vdw.store %addr, %final, %active_full, %qpu_id {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32
    ssavc4.thread_end
  }
}
