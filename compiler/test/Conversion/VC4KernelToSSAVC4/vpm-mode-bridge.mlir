// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @vpm_modes
// CHECK: ssavc4.vpm.write
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK-SAME: subword = #ssavc4.vpm_subword<none>
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w32>
// CHECK: ssavc4.vpm.read
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK-SAME: x = 3 : i32
// CHECK: ssavc4.vdr.load
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK-SAME: vpm_pitch = 1 : i32
// CHECK-SAME: vpm_x = 0 : i32
// CHECK: ssavc4.vdw.store_vpm
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK-NOT: orientation = "horizontal"
// CHECK-NOT: orientation = "vertical"
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @vpm_modes(%in : i32, %out : i32) attributes {
    public_name = "vpm_modes",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %value = vc4kernel.splat %one : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 2 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile, %zero, %value, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %read = vc4kernel.vpm_read_fragment %tile, %zero, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 3 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdr_load_to_vpm %in, %zero, %tile, %zero {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32
    vc4kernel.vdw_store_vpm_fragment %tile, %zero, %out, %zero, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 3 : i32, vpm_pitch = 1 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
