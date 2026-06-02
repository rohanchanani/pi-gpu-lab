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
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %lhs_base_v = vc4kernel.splat %lhs_base : i32 -> vector<16xi32>
    %rhs_base_v = vc4kernel.splat %rhs_base : i32 -> vector<16xi32>
    %lhs = vc4kernel.fragment_add %lhs_base_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %rhs = vc4kernel.fragment_add %rhs_base_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value = vc4kernel.fragment_mul %lhs, %rhs : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
