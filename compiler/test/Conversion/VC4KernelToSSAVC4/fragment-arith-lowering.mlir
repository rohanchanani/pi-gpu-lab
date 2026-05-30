// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @arith
// CHECK: ssavc4.load_imm
// CHECK: ssavc4.splat
// CHECK: ssavc4.alu.add
// CHECK-SAME: #vc4.add_opcode<add>
// CHECK: ssavc4.alu.add
// CHECK-SAME: #vc4.add_opcode<sub>
// CHECK: ssavc4.alu.mul
// CHECK: ssavc4.rotate
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @arith attributes {
    public_name = "arith",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {
      uses_vpm = false, uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32, vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32, semaphores_per_block = 0 : i32
    }
  } {
    %c1 = arith.constant 1 : i32
    %v = vc4kernel.splat %c1 : i32 -> vector<16xi32>
    %add = vc4kernel.fragment_add %v, %v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sub = vc4kernel.fragment_sub %add, %v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %mul = vc4kernel.fragment_mul %sub, %v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %rot = vc4kernel.fragment_rotate %mul {amount = 1 : i32} : vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}
