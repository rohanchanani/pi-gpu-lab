// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 2>&1 | FileCheck %s

// CHECK: SSAVC4 block-argument lowering could not reserve register homes for loop-carried spilled block arguments
ssavc4.module @block_args_spill_loop {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "block_args_spill_loop",
      code_symbol = "block_args_spill_loop_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 0 : i32,
      args = [],
      builtins = []
    }
  } {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %limit = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %one_vec = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
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
    ssavc4.br ^loop(%zero, %v01, %v02, %v03, %v04, %v05, %v06, %v07, %v08, %v09, %v10, %v11, %v12, %v13, %v14, %v15, %v16, %v17, %v18, %v19, %v20 : i32, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>)

  ^loop(%i: i32, %p01: vector<16xi32>, %p02: vector<16xi32>, %p03: vector<16xi32>, %p04: vector<16xi32>, %p05: vector<16xi32>, %p06: vector<16xi32>, %p07: vector<16xi32>, %p08: vector<16xi32>, %p09: vector<16xi32>, %p10: vector<16xi32>, %p11: vector<16xi32>, %p12: vector<16xi32>, %p13: vector<16xi32>, %p14: vector<16xi32>, %p15: vector<16xi32>, %p16: vector<16xi32>, %p17: vector<16xi32>, %p18: vector<16xi32>, %p19: vector<16xi32>, %p20: vector<16xi32>):
    %done_flags = ssavc4.make_flags %i, %limit {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags,
      ^done(%p01, %p02, %p03, %p04, %p05, %p06, %p07, %p08, %p09, %p10, %p11, %p12, %p13, %p14, %p15, %p16, %p17, %p18, %p19, %p20 : vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>),
      ^body(%i, %p01, %p02, %p03, %p04, %p05, %p06, %p07, %p08, %p09, %p10, %p11, %p12, %p13, %p14, %p15, %p16, %p17, %p18, %p19, %p20 : i32, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>)
      {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^body(%i_body: i32, %b01: vector<16xi32>, %b02: vector<16xi32>, %b03: vector<16xi32>, %b04: vector<16xi32>, %b05: vector<16xi32>, %b06: vector<16xi32>, %b07: vector<16xi32>, %b08: vector<16xi32>, %b09: vector<16xi32>, %b10: vector<16xi32>, %b11: vector<16xi32>, %b12: vector<16xi32>, %b13: vector<16xi32>, %b14: vector<16xi32>, %b15: vector<16xi32>, %b16: vector<16xi32>, %b17: vector<16xi32>, %b18: vector<16xi32>, %b19: vector<16xi32>, %b20: vector<16xi32>):
    %i_next = ssavc4.alu.add %i_body, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %n01 = ssavc4.alu.add %b01, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n02 = ssavc4.alu.add %b02, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n03 = ssavc4.alu.add %b03, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n04 = ssavc4.alu.add %b04, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n05 = ssavc4.alu.add %b05, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n06 = ssavc4.alu.add %b06, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n07 = ssavc4.alu.add %b07, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n08 = ssavc4.alu.add %b08, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n09 = ssavc4.alu.add %b09, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n10 = ssavc4.alu.add %b10, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n11 = ssavc4.alu.add %b11, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n12 = ssavc4.alu.add %b12, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n13 = ssavc4.alu.add %b13, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n14 = ssavc4.alu.add %b14, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n15 = ssavc4.alu.add %b15, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n16 = ssavc4.alu.add %b16, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n17 = ssavc4.alu.add %b17, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n18 = ssavc4.alu.add %b18, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n19 = ssavc4.alu.add %b19, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %n20 = ssavc4.alu.add %b20, %one_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^loop(%i_next, %n01, %n02, %n03, %n04, %n05, %n06, %n07, %n08, %n09, %n10, %n11, %n12, %n13, %n14, %n15, %n16, %n17, %n18, %n19, %n20 : i32, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>)

  ^done(%d01: vector<16xi32>, %d02: vector<16xi32>, %d03: vector<16xi32>, %d04: vector<16xi32>, %d05: vector<16xi32>, %d06: vector<16xi32>, %d07: vector<16xi32>, %d08: vector<16xi32>, %d09: vector<16xi32>, %d10: vector<16xi32>, %d11: vector<16xi32>, %d12: vector<16xi32>, %d13: vector<16xi32>, %d14: vector<16xi32>, %d15: vector<16xi32>, %d16: vector<16xi32>, %d17: vector<16xi32>, %d18: vector<16xi32>, %d19: vector<16xi32>, %d20: vector<16xi32>):
    %s01 = ssavc4.alu.add %d01, %d02 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s02 = ssavc4.alu.add %s01, %d03 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s03 = ssavc4.alu.add %s02, %d04 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s04 = ssavc4.alu.add %s03, %d05 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s05 = ssavc4.alu.add %s04, %d06 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s06 = ssavc4.alu.add %s05, %d07 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s07 = ssavc4.alu.add %s06, %d08 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s08 = ssavc4.alu.add %s07, %d09 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s09 = ssavc4.alu.add %s08, %d10 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s10 = ssavc4.alu.add %s09, %d11 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s11 = ssavc4.alu.add %s10, %d12 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s12 = ssavc4.alu.add %s11, %d13 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s13 = ssavc4.alu.add %s12, %d14 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s14 = ssavc4.alu.add %s13, %d15 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s15 = ssavc4.alu.add %s14, %d16 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s16 = ssavc4.alu.add %s15, %d17 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s17 = ssavc4.alu.add %s16, %d18 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s18 = ssavc4.alu.add %s17, %d19 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sink = ssavc4.alu.add %s18, %d20 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
