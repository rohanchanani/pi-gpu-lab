// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad(%ptr : i32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %tile = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: max_rows must be in range [1, 16]
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %c0, %c4, %c4, %c4 {max_rows = 17 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.return
  }
}
