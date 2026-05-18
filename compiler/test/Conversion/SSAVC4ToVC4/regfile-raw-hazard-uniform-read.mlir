// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @raw_hazard_uniform_read
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK-SAME: raddr_a = 32 : i32
// CHECK-SAME: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: waddr_add = 0 : i32
// CHECK-NEXT: vc4.qpu.ldi <splat32>
// CHECK-SAME: cond_add = #vc4.cond<never>
// CHECK-SAME: waddr_add = 32 : i32
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK-SAME: raddr_a = 0 : i32
// CHECK-NOT: lowering_template
// CHECK-NOT: ssavc4.
ssavc4.module @raw_hazard_uniform_read {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %u = ssavc4.uniform.read 0 : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %sum = ssavc4.alu.add %u, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
