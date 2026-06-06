// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @mov_bitcast_lowering
// CHECK: vc4.qpu.bundle
// CHECK-SAME: op_add = #vc4.add_opcode<or>
// CHECK-NOT: #vc4.add_opcode<itof>
// CHECK-NOT: #vc4.add_opcode<ftoi>
ssavc4.module @m {
  ssavc4.func @mov_bitcast_lowering() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %bits = ssavc4.load_imm <splat32> {value = 1065353216 : i32} : vector<16xi32>
    %as_f = ssavc4.mov %bits : vector<16xi32> -> vector<16xf32>
    %as_i = ssavc4.mov %as_f : vector<16xf32> -> vector<16xi32>
    ssavc4.thread_end
  }
}
