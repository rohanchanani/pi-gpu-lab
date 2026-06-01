// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @tmu_empty
// CHECK: ssavc4.load_imm
// CHECK-NOT: ssavc4.tmu.request
// CHECK-NOT: ssavc4.tmu.read
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @tmu_empty(%ptr : i32) attributes {
    public_name = "tmu_empty",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "in", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %empty = vc4kernel.pred.empty : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %empty : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
