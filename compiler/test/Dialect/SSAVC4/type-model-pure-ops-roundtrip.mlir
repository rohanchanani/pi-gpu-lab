// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: module
// CHECK: ssavc4.load_imm <splat32>
// CHECK: ssavc4.uniform.read 2 : f32
// CHECK: ssavc4.load_imm <per_elem_u2>
// CHECK: ssavc4.element_number
// CHECK: ssavc4.splat
// CHECK: ssavc4.splat {{.*}} : f32 -> vector<16xf32>
// CHECK: ssavc4.mov
// CHECK: ssavc4.alu.add
// CHECK-SAME: opcode = #vc4.add_opcode<add>
// CHECK: ssavc4.alu.mul
// CHECK-SAME: opcode = #vc4.mul_opcode<mul24>
// CHECK: ssavc4.rotate
// CHECK-SAME: amount = 4 : i32
// CHECK: ssavc4.pack
// CHECK: ssavc4.unpack
// CHECK: ssavc4.make_flags
// CHECK-SAME: kind = #ssavc4.flag_kind<sub>
module {
  %i = ssavc4.load_imm <splat32> {value = 42 : i32} : i32
  %a = ssavc4.uniform.read 2 : f32
  %v = ssavc4.load_imm <per_elem_u2> {values = array<i32: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3>} : vector<16xi32>
  %lane = ssavc4.element_number : vector<16xi32>
  %splat = ssavc4.splat %i : i32 -> vector<16xi32>
  %av = ssavc4.splat %a : f32 -> vector<16xf32>
  %moved = ssavc4.mov %splat : vector<16xi32> -> vector<16xi32>
  %sum = ssavc4.alu.add %moved, %lane {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  %prod = ssavc4.alu.mul %sum, %v {opcode = #vc4.mul_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  %rot = ssavc4.rotate %prod {amount = 4 : i32} : vector<16xi32> -> vector<16xi32>
  %packed = ssavc4.pack %rot {mode = #vc4.regfile_a_pack_mode<to_16a>} : vector<16xi32> -> vector<16xi32>
  %unpacked = ssavc4.unpack %packed {mode = #vc4.regfile_a_unpack_mode<f16a_or_i16a>} : vector<16xi32> -> vector<16xi32>
  %flags = ssavc4.make_flags %unpacked, %lane {kind = #ssavc4.flag_kind<sub>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
}
