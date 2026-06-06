// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_i32_mul24_fastpath
// CHECK: ssavc4.element_number : vector<16xi32>
// CHECK: ssavc4.splat {{.*}} : i32 -> vector<16xi32>
// CHECK-COUNT-1: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<mul24>}
// CHECK-NOT: #vc4.add_opcode<bit_and>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_i32_mul24_fastpath(%out : i32) attributes {
    public_name = "fragment_i32_mul24_fastpath",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %scale = arith.constant 257 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %lanes, %byte_offsets_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %scale_v = vc4kernel.splat %scale : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_alu.mul %lanes, %scale_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
