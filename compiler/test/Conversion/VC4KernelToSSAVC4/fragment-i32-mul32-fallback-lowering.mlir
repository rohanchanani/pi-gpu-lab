// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_i32_mul32_fallback
// CHECK: ssavc4.uniform.read 1 : i32
// CHECK: ssavc4.uniform.read 2 : i32
// CHECK: #vc4.add_opcode<add>
// CHECK: #vc4.add_opcode<add>
// CHECK: #vc4.add_opcode<and>
// CHECK: #vc4.add_opcode<shr>
// CHECK-COUNT-3: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<mul24>}
// CHECK: #vc4.add_opcode<shl>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_i32_mul32_fallback(%out : i32, %lhs_base : i32, %rhs_base : i32) attributes {
    public_name = "fragment_i32_mul32_fallback",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "lhs_base", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "rhs_base", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %lhs_base_v = vc4kernel.splat %lhs_base : i32 -> vector<16xi32>
    %rhs_base_v = vc4kernel.splat %rhs_base : i32 -> vector<16xi32>
    %lhs = vc4kernel.fragment_alu.add %lhs_base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %rhs = vc4kernel.fragment_alu.add %rhs_base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_mask = vc4kernel.fragment_const {value = dense<65535> : vector<16xi32>} : vector<16xi32>
    %value_shift = vc4kernel.fragment_const {value = dense<16> : vector<16xi32>} : vector<16xi32>
    %value_lhs_lo = vc4kernel.fragment_alu.add %lhs, %value_mask {opcode = #vc4kernel.add_alu_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_rhs_lo = vc4kernel.fragment_alu.add %rhs, %value_mask {opcode = #vc4kernel.add_alu_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_lhs_hi = vc4kernel.fragment_alu.add %lhs, %value_shift {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_rhs_hi = vc4kernel.fragment_alu.add %rhs, %value_shift {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_lo_lo = vc4kernel.fragment_alu.mul %value_lhs_lo, %value_rhs_lo {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_lo_hi = vc4kernel.fragment_alu.mul %value_lhs_lo, %value_rhs_hi {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_hi_lo = vc4kernel.fragment_alu.mul %value_lhs_hi, %value_rhs_lo {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_cross = vc4kernel.fragment_alu.add %value_lo_hi, %value_hi_lo {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_cross_shifted = vc4kernel.fragment_alu.add %value_cross, %value_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %value_lo_lo, %value_cross_shifted {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
