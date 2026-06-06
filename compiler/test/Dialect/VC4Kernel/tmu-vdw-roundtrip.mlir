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
    // CHECK: vc4kernel.pred.full
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    // CHECK: vc4kernel.pred.tail
    %tail = vc4kernel.pred.tail %base, %limit : i32, i32 -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.lane_range
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %basev = vc4kernel.splat %base : i32 -> vector<16xi32>
    %offs = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    // CHECK: vc4kernel.fragment_cmp
    %cmp = vc4kernel.fragment_cmp %lanes, %basev {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.tmu_load_fragment
    %p7_safe0 = arith.constant 0 : i32
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %cmp, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    // CHECK: vc4kernel.vdw_store_fragment
    vc4kernel.vdw_store_fragment %ptr, %offs, %v, %cmp {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
