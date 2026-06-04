// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-hardware-rules | FileCheck %s

// CHECK-LABEL: vc4.module @f32_load_imm_ssavc4
// CHECK: vc4.qpu.ldi <splat32> {{.*}}value = 0 : i32
// CHECK: vc4.qpu.ldi <splat32> {{.*}}value = 1065353216 : i32
// CHECK: op_add = #vc4.add_opcode<fadd>
// CHECK-NOT: ssavc4.
ssavc4.module @f32_load_imm_ssavc4 {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0.000000e+00 : f32} : vector<16xf32>
    %one = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : vector<16xf32>
    %sum = ssavc4.alu.add %zero, %one {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.thread_end
  }
}
