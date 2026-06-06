// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_i32_mul24_fastpath
// CHECK: ssavc4.element_number : vector<16xi32>
// CHECK: ssavc4.load_imm <splat32> {value = 257 : i32} : vector<16xi32>
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
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %scale_v = vc4kernel.fragment_const {value = dense<257> : vector<16xi32>} : vector<16xi32>
    %value = vc4kernel.fragment_alu.mul %lanes, %scale_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
