// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @vpm(%value : i32) attributes {
    public_name = "vpm",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "value", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %frag = vc4kernel.splat %value : i32 -> vector<16xi32>
    // CHECK: vc4kernel.fragment_cmp
    %cmp = vc4kernel.fragment_cmp %lanes, %frag {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    // CHECK: vc4kernel.vpm_alloc
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: vc4kernel.vpm_write_fragment
    // CHECK: #vc4kernel.vpm_orientation<row>
    vc4kernel.vpm_write_fragment %tile, %c0, %frag, %cmp {orientation = #vc4kernel.vpm_orientation<row>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    // CHECK: vc4kernel.vpm_read_fragment
    // CHECK: #vc4kernel.vpm_orientation<row>
    %read = vc4kernel.vpm_read_fragment %tile, %c0, %cmp {orientation = #vc4kernel.vpm_orientation<row>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
