// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  // CHECK-LABEL: vc4kernel.kernel @vpm_dynamic_row
  // CHECK: vc4kernel.vpm_write_fragment
  vc4kernel.kernel @vpm_dynamic_row(%row : i32) attributes {
    public_name = "vpm_dynamic_row",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %v = vc4kernel.splat %c0 : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile, %row, %v, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
