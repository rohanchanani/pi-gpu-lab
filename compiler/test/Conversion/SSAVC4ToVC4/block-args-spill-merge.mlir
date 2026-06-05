// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck --implicit-check-not="#vc4.qpu_signal<ldtmu0>" %s

// CHECK-LABEL: vc4.func @block_args_spill_merge_kernel
// CHECK-SAME: spill_frame_bytes = {{[1-9][0-9]*}} : i32
// CHECK-SAME: spill_frame_stride_bytes = {{[1-9][0-9]*}} : i32
// CHECK-SAME: spill_frame_base
// CHECK-NOT: ssavc4.phi
// CHECK-DAG: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<any_c_clear>
// CHECK-DAG: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<always>
// CHECK-DAG: vc4.qpu.vpmvcd_addr
// CHECK-DAG: side = #vc4.vpmvcd_side<read>
ssavc4.module @block_args_spill_merge {
  ssavc4.func @block_args_spill_merge_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "block_args_spill_merge",
      code_symbol = "block_args_spill_merge_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32}
      ],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
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
    %v37 = ssavc4.load_imm <splat32> {value = 37 : i32} : vector<16xi32>
    %v38 = ssavc4.load_imm <splat32> {value = 38 : i32} : vector<16xi32>
    %v39 = ssavc4.load_imm <splat32> {value = 39 : i32} : vector<16xi32>
    %v40 = ssavc4.load_imm <splat32> {value = 40 : i32} : vector<16xi32>
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags,
      ^merge(%v01, %v02, %v03, %v04, %v05, %v06, %v07, %v08, %v09, %v10, %v11, %v12, %v13, %v14, %v15, %v16, %v17, %v18, %v19, %v20 : vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>),
      ^merge(%v21, %v22, %v23, %v24, %v25, %v26, %v27, %v28, %v29, %v30, %v31, %v32, %v33, %v34, %v35, %v36, %v37, %v38, %v39, %v40 : vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>, vector<16xi32>)
      {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^merge(%m01: vector<16xi32>, %m02: vector<16xi32>, %m03: vector<16xi32>, %m04: vector<16xi32>, %m05: vector<16xi32>, %m06: vector<16xi32>, %m07: vector<16xi32>, %m08: vector<16xi32>, %m09: vector<16xi32>, %m10: vector<16xi32>, %m11: vector<16xi32>, %m12: vector<16xi32>, %m13: vector<16xi32>, %m14: vector<16xi32>, %m15: vector<16xi32>, %m16: vector<16xi32>, %m17: vector<16xi32>, %m18: vector<16xi32>, %m19: vector<16xi32>, %m20: vector<16xi32>):
    %s01 = ssavc4.alu.add %m01, %m02 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s02 = ssavc4.alu.add %s01, %m03 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s03 = ssavc4.alu.add %s02, %m04 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s04 = ssavc4.alu.add %s03, %m05 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s05 = ssavc4.alu.add %s04, %m06 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s06 = ssavc4.alu.add %s05, %m07 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s07 = ssavc4.alu.add %s06, %m08 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s08 = ssavc4.alu.add %s07, %m09 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s09 = ssavc4.alu.add %s08, %m10 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s10 = ssavc4.alu.add %s09, %m11 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s11 = ssavc4.alu.add %s10, %m12 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s12 = ssavc4.alu.add %s11, %m13 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s13 = ssavc4.alu.add %s12, %m14 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s14 = ssavc4.alu.add %s13, %m15 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s15 = ssavc4.alu.add %s14, %m16 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s16 = ssavc4.alu.add %s15, %m17 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s17 = ssavc4.alu.add %s16, %m18 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s18 = ssavc4.alu.add %s17, %m19 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s19 = ssavc4.alu.add %s18, %m20 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
