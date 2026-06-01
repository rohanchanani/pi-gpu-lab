// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @store
// CHECK-SAME: uses_shared_vpm = true
// CHECK-SAME: vpm_bytes_per_block = 64
// CHECK-SAME: vpm_rows_per_block = 1
// CHECK: ssavc4.vdw.store
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @store(%ptr : i32) attributes {
    public_name = "store",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "out", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %v = vc4kernel.splat %c1 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %ptr, %offs, %v, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
