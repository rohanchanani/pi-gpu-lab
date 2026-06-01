// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @reserved_scratch_reg31
// CHECK: spill_frame_bytes = 384 : i32
// CHECK: value = 28 : i32, waddr_add = 25 : i32
// CHECK: sig = #vc4.qpu_signal<ldtmu0>

// This keeps enough vector values live across a cooperative barrier to force
// the allocator to confront ra31. The barrier lowering uses ra31 as fixed
// scratch for flag-setting operations, so user SSA values must not be homed
// there across the barrier.
ssavc4.module @reserved_scratch_reg31 {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "reserved_scratch_reg31",
      code_symbol = "reserved_scratch_reg31_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_warp_id", kind = #vc4.builtin_kind<logical_warp_id>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "warps_per_block", kind = #vc4.builtin_kind<warps_per_block>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 12 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = true,
      uses_vdr = false,
      uses_vdw = false,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
    }
  } {
    %logical = ssavc4.uniform.read 0 : i32
    %warps = ssavc4.uniform.read 1 : i32
    %v00 = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
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
    ssavc4.barrier %logical, %warps : i32, i32 {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}
    %s01 = ssavc4.alu.add %v00, %v01 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
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
    %sum = ssavc4.alu.add %s27, %v28 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.vpm.write %logical, %sum {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
