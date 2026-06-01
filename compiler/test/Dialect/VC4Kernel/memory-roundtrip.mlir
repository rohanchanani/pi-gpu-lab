// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @mem(%ptr : i32) attributes {
    public_name = "mem",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offs = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    // CHECK: vc4kernel.fragment_cmp
    %cmp = vc4kernel.fragment_cmp %lanes, %offs {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.tmu_load_fragment
    %v = vc4kernel.tmu_load_fragment %ptr, %offs, %cmp : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: vc4kernel.vdw_store_fragment
    vc4kernel.vdw_store_fragment %ptr, %offs, %v, %cmp : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    // CHECK: vc4kernel.vpm_alloc
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: vc4kernel.vpm_write_fragment
    vc4kernel.vpm_write_fragment %tile, %c0, %v, %cmp {orientation = #vc4kernel.vpm_orientation<row>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    // CHECK: vc4kernel.vpm_read_fragment
    %r = vc4kernel.vpm_read_fragment %tile, %c0, %cmp {orientation = #vc4kernel.vpm_orientation<row>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: vc4kernel.vdr_load_to_vpm
    vc4kernel.vdr_load_to_vpm %ptr, %c0, %tile, %c0 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32} : i32, i32, !vc4kernel.vpm_tile, i32
    // CHECK: vc4kernel.vdw_store_vpm_fragment
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %ptr, %c0, %full {elem_bytes = 4 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
