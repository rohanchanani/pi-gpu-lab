// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @spill_branch_layout_invalid {
  ssavc4.func @spill_branch_layout_invalid_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block_max = 1 : i32,
      uses_shared_vpm = false,
      uses_barrier = false,
      semaphores_per_block = 0 : i32,
      require_full_block_residency = false
    }
  } {
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
    %v29 = ssavc4.load_imm <splat32> {value = 29 : i32} : vector<16xi32>
    %v30 = ssavc4.load_imm <splat32> {value = 30 : i32} : vector<16xi32>
    %v31 = ssavc4.load_imm <splat32> {value = 31 : i32} : vector<16xi32>
    %v32 = ssavc4.load_imm <splat32> {value = 32 : i32} : vector<16xi32>
    %v33 = ssavc4.load_imm <splat32> {value = 33 : i32} : vector<16xi32>
    %v34 = ssavc4.load_imm <splat32> {value = 34 : i32} : vector<16xi32>
    %v35 = ssavc4.load_imm <splat32> {value = 35 : i32} : vector<16xi32>
    %v36 = ssavc4.load_imm <splat32> {value = 36 : i32} : vector<16xi32>
    ssavc4.br ^loop
  ^loop:
    %s01 = ssavc4.alu.add %v01, %v02 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s02 = ssavc4.alu.add %s01, %v03 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s03 = ssavc4.alu.add %s02, %v04 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s04 = ssavc4.alu.add %s03, %v05 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s05 = ssavc4.alu.add %s04, %v06 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s06 = ssavc4.alu.add %s05, %v07 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s07 = ssavc4.alu.add %s06, %v08 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s08 = ssavc4.alu.add %s07, %v09 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s09 = ssavc4.alu.add %s08, %v10 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s10 = ssavc4.alu.add %s09, %v11 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s11 = ssavc4.alu.add %s10, %v12 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s12 = ssavc4.alu.add %s11, %v13 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s13 = ssavc4.alu.add %s12, %v14 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s14 = ssavc4.alu.add %s13, %v15 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s15 = ssavc4.alu.add %s14, %v16 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s16 = ssavc4.alu.add %s15, %v17 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s17 = ssavc4.alu.add %s16, %v18 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s18 = ssavc4.alu.add %s17, %v19 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s19 = ssavc4.alu.add %s18, %v20 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s20 = ssavc4.alu.add %s19, %v21 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s21 = ssavc4.alu.add %s20, %v22 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s22 = ssavc4.alu.add %s21, %v23 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s23 = ssavc4.alu.add %s22, %v24 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s24 = ssavc4.alu.add %s23, %v25 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s25 = ssavc4.alu.add %s24, %v26 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s26 = ssavc4.alu.add %s25, %v27 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s27 = ssavc4.alu.add %s26, %v28 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s28 = ssavc4.alu.add %s27, %v29 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s29 = ssavc4.alu.add %s28, %v30 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s30 = ssavc4.alu.add %s29, %v31 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s31 = ssavc4.alu.add %s30, %v32 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s32 = ssavc4.alu.add %s31, %v33 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s33 = ssavc4.alu.add %s32, %v34 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s34 = ssavc4.alu.add %s33, %v35 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s35 = ssavc4.alu.add %s34, %v36 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    // CHECK: S3 spilling does not support loop/backedge branch layouts requiring path-sensitive liveness
    ssavc4.br ^loop
  }
}
