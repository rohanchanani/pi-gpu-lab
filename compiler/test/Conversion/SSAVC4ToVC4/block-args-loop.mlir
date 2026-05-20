// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @block_args_loop
// CHECK-NOT: ssavc4.phi
// Entry edge copies initial values to loop header args.
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// Loop exit and body edges use path-specific copy blocks.
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<any_c_clear>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// Backedge copies next values before a backward branch.
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK-SAME: immediate = -{{[0-9]+}} : i32
ssavc4.module @block_args_loop {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %limit = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %acc0 = ssavc4.load_imm <splat32> {value = 10 : i32} : i32
    ssavc4.br ^loop(%zero, %acc0 : i32, i32)

  ^loop(%i: i32, %acc: i32):
    %done_flags = ssavc4.make_flags %i, %limit {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^done(%acc : i32), ^body(%i, %acc : i32, i32) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^body(%i_body: i32, %acc_body: i32):
    %i_next = ssavc4.alu.add %i_body, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %acc_next = ssavc4.alu.add %acc_body, %i_next {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%i_next, %acc_next : i32, i32)

  ^done(%final: i32):
    %sink = ssavc4.alu.add %final, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
