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
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<horizontal>
    // CHECK-SAME: subword = #vc4kernel.vpm_subword<none>
    // CHECK-SAME: width = #vc4kernel.vpm_width<w32>
    vc4kernel.vpm_write_fragment %tile, %c0, %frag, %cmp {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    // CHECK: vc4kernel.vpm_read_fragment
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<vertical>
    %read = vc4kernel.vpm_read_fragment %tile, %c0, %cmp {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 3 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
