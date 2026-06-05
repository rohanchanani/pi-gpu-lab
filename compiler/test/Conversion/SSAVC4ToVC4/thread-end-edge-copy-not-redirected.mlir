// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @thread_end_edge_copy_not_redirected
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK-COUNT-1: sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: sig = #vc4.qpu_signal<thrend>
ssavc4.module @thread_end_edge_copy_not_redirected {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    ssavc4.br ^done(%one : i32)

  ^done(%unused: i32):
    ssavc4.thread_end
  }
}
