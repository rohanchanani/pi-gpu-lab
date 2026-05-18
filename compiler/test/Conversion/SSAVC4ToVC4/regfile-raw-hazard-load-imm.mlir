// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @raw_hazard_load_imm
// CHECK: vc4.qpu.ldi <splat32>
// CHECK-SAME: value = 99 : i32
// CHECK-SAME: waddr_add = 0 : i32
// CHECK: vc4.qpu.ldi <splat32>
// CHECK-SAME: value = 4 : i32
// CHECK-SAME: waddr_add = 1 : i32
// CHECK-NEXT: vc4.qpu.ldi <splat32>
// CHECK-SAME: cond_add = #vc4.cond<never>
// CHECK-SAME: waddr_add = 32 : i32
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK-SAME: raddr_a = 1 : i32
// CHECK-NOT: lowering_template
// CHECK-NOT: ssavc4.
ssavc4.module @raw_hazard_load_imm {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %dead = ssavc4.load_imm <splat32> {value = 99 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %sum = ssavc4.alu.add %x, %x {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
