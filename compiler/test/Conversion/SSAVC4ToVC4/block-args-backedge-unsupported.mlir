// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @block_args_backedge_unsupported
// CHECK-NOT: ssavc4.phi
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<always>{{.*}}immediate = -{{[0-9]+}} : i32
ssavc4.module @block_args_backedge_unsupported {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    ssavc4.br ^loop(%zero : i32)
  ^loop(%i: i32):
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %next = ssavc4.alu.add %i, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%next : i32)
  }
}
