// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @arith
// CHECK: ssavc4.load_imm
// CHECK: ssavc4.alu.add
// CHECK-SAME: #vc4.add_opcode<add>
// CHECK: ssavc4.alu.add
// CHECK-SAME: #vc4.add_opcode<sub>
// CHECK: ssavc4.alu.mul
// CHECK: ssavc4.rotate
// CHECK: ssavc4.splat
// CHECK: ssavc4.alu.add
// CHECK-SAME: #vc4.add_opcode<fadd>
// CHECK: ssavc4.alu.mul
// CHECK-SAME: #vc4.mul_opcode<fmul>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @arith(%f : f32) attributes {
    public_name = "arith",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "f", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %add = vc4kernel.fragment_alu.add %v, %v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sub = vc4kernel.fragment_alu.add %add, %v {opcode = #vc4kernel.add_alu_opcode<sub>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %mul = vc4kernel.fragment_alu.mul %sub, %v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %rot = vc4kernel.fragment_rotate %mul {amount = 1 : i32} : vector<16xi32> -> vector<16xi32>
    %fv = vc4kernel.splat %f : f32 -> vector<16xf32>
    %fadd = vc4kernel.fragment_alu.add %fv, %fv {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %fmul = vc4kernel.fragment_alu.mul %fadd, %fv {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    vc4kernel.return
  }
}
