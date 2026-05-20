// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @block_args_cond_merge
// CHECK-NOT: ssavc4.phi
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<any_c_clear>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<add>
ssavc4.module @block_args_cond_merge {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %left = ssavc4.load_imm <splat32> {value = 11 : i32} : i32
    %right = ssavc4.load_imm <splat32> {value = 29 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^merge(%left : i32), ^merge(%right : i32) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags
  ^merge(%m: i32):
    %sum = ssavc4.alu.add %m, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
