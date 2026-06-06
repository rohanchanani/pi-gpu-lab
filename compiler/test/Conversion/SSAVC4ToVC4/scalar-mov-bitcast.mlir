// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @scalar_mov_bitcast
// CHECK: vc4.func @kernel
// CHECK-NOT: #vc4.add_opcode<itof>
// CHECK-NOT: #vc4.add_opcode<ftoi>
// CHECK-NOT: ssavc4.
ssavc4.module @scalar_mov_bitcast {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %raw = ssavc4.load_imm <splat32> {value = 1065353216 : i32} : i32
    %f = ssavc4.mov %raw : i32 -> f32
    %bits = ssavc4.mov %f : f32 -> i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %sum = ssavc4.alu.add %bits, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
