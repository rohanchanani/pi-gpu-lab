// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @tmu
// CHECK: ssavc4.tmu.request
// CHECK: ssavc4.tmu.read
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @tmu(%ptr : i32) attributes {
    public_name = "tmu",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %offs = vc4kernel.fragment_alu.add %lanes, %offs_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
