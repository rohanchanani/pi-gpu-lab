// RUN: vc4-opt %s | FileCheck %s

// This slice intentionally adds only pure value and descriptor/type-model
// constructs.  Hardware-state operations such as uniform/TMU request/VPM/VDW
// store/sema/barrier/thread_end are not introduced as pure placeholders here.

// CHECK: ssavc4.load_imm
// CHECK: ssavc4.element_number
// CHECK: ssavc4.alu.add
module {
  %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  %lane = ssavc4.element_number : vector<16xi32>
  %vone = ssavc4.splat %one : i32 -> vector<16xi32>
  %sum = ssavc4.alu.add %vone, %lane {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
}
