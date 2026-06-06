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
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %p7_safe0 = arith.constant 0 : i32
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    vc4kernel.return
  }
}
