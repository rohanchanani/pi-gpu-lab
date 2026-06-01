// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @tmu_vdw(%ptr : i32, %base : i32, %limit : i32) attributes {
    public_name = "tmu_vdw",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "base", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "limit", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    // CHECK: vc4kernel.pred.full
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.tail
    %tail = vc4kernel.pred.tail %base, %limit : i32, i32 -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.lane_range
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    // CHECK: vc4kernel.tmu_load_fragment
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %tail : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: vc4kernel.vdw_store_fragment
    vc4kernel.vdw_store_fragment %ptr, %offs, %v, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
