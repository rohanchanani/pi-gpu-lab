// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @vpm_subword(%value : i32) attributes {
    public_name = "vpm_subword",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "value", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %frag = vc4kernel.splat %value : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: vc4kernel.vpm_write_fragment
    // CHECK-SAME: subword = #vc4kernel.vpm_subword<packed>
    // CHECK-SAME: width = #vc4kernel.vpm_width<w8>
    vc4kernel.vpm_write_fragment %tile, %c0, %frag, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, x = 3 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    // CHECK: vc4kernel.vpm_write_fragment
    // CHECK-SAME: subword = #vc4kernel.vpm_subword<laned>
    // CHECK-SAME: width = #vc4kernel.vpm_width<w16>
    vc4kernel.vpm_write_fragment %tile, %c0, %frag, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<laned>, x = 1 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    // CHECK: vc4kernel.vpm_read_fragment
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<vertical>
    // CHECK-SAME: subword = #vc4kernel.vpm_subword<packed>
    // CHECK-SAME: width = #vc4kernel.vpm_width<w16>
    %r0 = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, x = 15 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: vc4kernel.vpm_read_fragment
    // CHECK-SAME: subword = #vc4kernel.vpm_subword<laned>
    // CHECK-SAME: width = #vc4kernel.vpm_width<w8>
    %r1 = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<laned>, x = 7 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.return
  }
}
