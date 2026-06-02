// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @vdr_horizontal_vertical
// CHECK: ssavc4.vdr.load
// CHECK-SAME: memory_pitch_bytes = 64 : i32
// CHECK-SAME: nrows = 4 : i32
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK-SAME: row_len = 16 : i32
// CHECK-SAME: serialize = "mutex"
// CHECK-SAME: vpm_pitch = 1 : i32
// CHECK-SAME: vpm_x = 0 : i32
// CHECK: ssavc4.vdr.load
// CHECK-SAME: memory_pitch_bytes = 128 : i32
// CHECK-SAME: nrows = 3 : i32
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK-SAME: row_len = 8 : i32
// CHECK-SAME: vpm_pitch = 2 : i32
// CHECK-SAME: vpm_x = 2 : i32
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @vdr_horizontal_vertical(%in : i32) attributes {
    public_name = "vdr_horizontal_vertical",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c256 = arith.constant 256 : i32
    %tile = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c0 {rows = 4 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32
    vc4kernel.vdr_load_to_vpm %in, %c256, %tile, %c0 {rows = 3 : i32, cols = 8 : i32, global_stride_bytes = 128 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 2 : i32, vpm_pitch = 2 : i32} : i32, i32, !vc4kernel.vpm_tile, i32
    vc4kernel.return
  }
}
