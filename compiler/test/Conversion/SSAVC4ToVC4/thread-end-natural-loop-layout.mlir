// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards | FileCheck %s

// CHECK-LABEL: vc4.module @thread_end_natural_loop_layout
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<any_c_clear>
// Exit block appears before the body in source order, but it must branch to a
// final epilogue instead of emitting thread_end before the body/backedge.
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<always>{{.*}}immediate = -{{[0-9]+}} : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK: vc4.qpu.bundle
// CHECK: vc4.qpu.bundle
// CHECK-NOT: vc4.qpu.
ssavc4.module @thread_end_natural_loop_layout {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %limit = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    ssavc4.br ^loop(%zero, %zero : i32, i32)

  ^loop(%i: i32, %acc: i32):
    %done_flags = ssavc4.make_flags %i, %limit {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^done(%acc : i32), ^body(%i, %acc : i32, i32) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^done(%final: i32):
    %sink = ssavc4.alu.add %final, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end

  ^body(%i_body: i32, %acc_body: i32):
    %i_next = ssavc4.alu.add %i_body, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %acc_next = ssavc4.alu.add %acc_body, %i_next {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%i_next, %acc_next : i32, i32)
  }
}
