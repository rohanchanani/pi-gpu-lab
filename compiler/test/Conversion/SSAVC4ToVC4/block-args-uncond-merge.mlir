// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @block_args_uncond_merge
// CHECK-NOT: ssavc4.phi
// CHECK: vc4.qpu.ldi
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<add>
ssavc4.module @block_args_uncond_merge {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    ssavc4.br ^merge(%one : i32)
  ^merge(%m: i32):
    %two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %sum = ssavc4.alu.add %m, %two {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
