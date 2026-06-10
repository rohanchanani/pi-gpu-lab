// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @multi_exit_merge_block_arg_spill_kernel
// CHECK-SAME: spill_frame_bytes = {{[1-9][0-9]*}} : i32
// CHECK: vc4.qpu.bundle
// CHECK-SAME: raddr_a = 32 : i32
// CHECK-SAME: waddr_add = 28 : i32
// CHECK-NEXT: vc4.qpu.ldi
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: raddr_a = 32 : i32
// CHECK-SAME: waddr_add = 27 : i32
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<any_c_clear>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @multi_exit_merge_block_arg_spill {
  ssavc4.func @multi_exit_merge_block_arg_spill_kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %v02 = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %v03 = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %v04 = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %v05 = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %v06 = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %v07 = ssavc4.load_imm <splat32> {value = 7 : i32} : i32
    %v08 = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %v09 = ssavc4.load_imm <splat32> {value = 9 : i32} : i32
    %v10 = ssavc4.load_imm <splat32> {value = 10 : i32} : i32
    %v11 = ssavc4.load_imm <splat32> {value = 11 : i32} : i32
    %v12 = ssavc4.load_imm <splat32> {value = 12 : i32} : i32
    %v13 = ssavc4.load_imm <splat32> {value = 13 : i32} : i32
    %v14 = ssavc4.load_imm <splat32> {value = 14 : i32} : i32
    %v15 = ssavc4.load_imm <splat32> {value = 15 : i32} : i32
    %v16 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %v17 = ssavc4.load_imm <splat32> {value = 17 : i32} : i32
    %v18 = ssavc4.load_imm <splat32> {value = 18 : i32} : i32
    %v19 = ssavc4.load_imm <splat32> {value = 19 : i32} : i32
    %v20 = ssavc4.load_imm <splat32> {value = 20 : i32} : i32
    %v21 = ssavc4.load_imm <splat32> {value = 21 : i32} : i32
    %v22 = ssavc4.load_imm <splat32> {value = 22 : i32} : i32
    %v23 = ssavc4.load_imm <splat32> {value = 23 : i32} : i32
    %v24 = ssavc4.load_imm <splat32> {value = 24 : i32} : i32
    %v25 = ssavc4.load_imm <splat32> {value = 25 : i32} : i32
    %v26 = ssavc4.load_imm <splat32> {value = 26 : i32} : i32
    %v27 = ssavc4.load_imm <splat32> {value = 27 : i32} : i32
    %v28 = ssavc4.load_imm <splat32> {value = 28 : i32} : i32
    %v29 = ssavc4.load_imm <splat32> {value = 29 : i32} : i32
    %v30 = ssavc4.load_imm <splat32> {value = 30 : i32} : i32
    %v31 = ssavc4.load_imm <splat32> {value = 31 : i32} : i32
    %v32 = ssavc4.load_imm <splat32> {value = 32 : i32} : i32
    %v33 = ssavc4.load_imm <splat32> {value = 33 : i32} : i32
    %v34 = ssavc4.load_imm <splat32> {value = 34 : i32} : i32
    %v35 = ssavc4.load_imm <splat32> {value = 35 : i32} : i32
    %v36 = ssavc4.load_imm <splat32> {value = 36 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^exit_done(%v10 : i32), ^exit_early(%v20 : i32) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^exit_done(%done_in: i32):
    %done = ssavc4.alu.add %done_in, %v02 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^merge(%done : i32)

  ^exit_early(%early_in: i32):
    %early = ssavc4.alu.add %early_in, %v03 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^merge(%early : i32)

  ^merge(%m: i32):
    %s04 = ssavc4.alu.add %m, %v04 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s05 = ssavc4.alu.add %s04, %v05 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s06 = ssavc4.alu.add %s05, %v06 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s07 = ssavc4.alu.add %s06, %v07 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s08 = ssavc4.alu.add %s07, %v08 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s09 = ssavc4.alu.add %s08, %v09 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s10 = ssavc4.alu.add %s09, %v10 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s11 = ssavc4.alu.add %s10, %v11 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s12 = ssavc4.alu.add %s11, %v12 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s13 = ssavc4.alu.add %s12, %v13 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s14 = ssavc4.alu.add %s13, %v14 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s15 = ssavc4.alu.add %s14, %v15 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s16 = ssavc4.alu.add %s15, %v16 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s17 = ssavc4.alu.add %s16, %v17 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s18 = ssavc4.alu.add %s17, %v18 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s19 = ssavc4.alu.add %s18, %v19 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s20 = ssavc4.alu.add %s19, %v20 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s21 = ssavc4.alu.add %s20, %v21 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s22 = ssavc4.alu.add %s21, %v22 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s23 = ssavc4.alu.add %s22, %v23 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s24 = ssavc4.alu.add %s23, %v24 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s25 = ssavc4.alu.add %s24, %v25 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s26 = ssavc4.alu.add %s25, %v26 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s27 = ssavc4.alu.add %s26, %v27 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s28 = ssavc4.alu.add %s27, %v28 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s29 = ssavc4.alu.add %s28, %v29 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s30 = ssavc4.alu.add %s29, %v30 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s31 = ssavc4.alu.add %s30, %v31 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s32 = ssavc4.alu.add %s31, %v32 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s33 = ssavc4.alu.add %s32, %v33 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s34 = ssavc4.alu.add %s33, %v34 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %s35 = ssavc4.alu.add %s34, %v35 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %sink = ssavc4.alu.add %s35, %v36 {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
