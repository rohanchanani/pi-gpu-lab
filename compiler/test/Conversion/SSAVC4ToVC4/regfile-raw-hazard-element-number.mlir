// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @raw_hazard_element_number
// CHECK: op_add = #vc4.add_opcode<or>
// CHECK-SAME: raddr_a = 38 : i32
// CHECK-SAME: waddr_add = 0 : i32
// CHECK-NEXT: vc4.qpu.ldi <splat32>
// CHECK-SAME: cond_add = #vc4.cond<never>
// CHECK-SAME: waddr_add = 32 : i32
// CHECK: op_add = #vc4.add_opcode<shl>
// CHECK-SAME: raddr_a = 0 : i32
// CHECK-NOT: lowering_template
// CHECK-NOT: ssavc4.
ssavc4.module @raw_hazard_element_number {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %lane = ssavc4.element_number : vector<16xi32>
    %shift = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    %bytes = ssavc4.alu.add %lane, %shift {opcode = #vc4.add_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
