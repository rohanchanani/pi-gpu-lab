// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @vdr_vdw_subword(%in : i32, %out : i32) attributes {
    public_name = "vdr_vdw_subword",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "u8"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "u8"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c8 = arith.constant 8 : i32
    %c16 = arith.constant 16 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: vc4kernel.vdr_load_to_vpm
    // CHECK-SAME: elem_bytes = 1
    // CHECK-SAME: subword = #vc4kernel.vpm_subword<packed>
    // CHECK-SAME: width = #vc4kernel.vpm_width<w8>
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c0 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 16 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, dst_x = 3 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32
    // CHECK: vc4kernel.vdr_load_rect_to_vpm
    // CHECK-SAME: elem_bytes = 2
    // CHECK-SAME: width = #vc4kernel.vpm_width<w16>
    vc4kernel.vdr_load_rect_to_vpm %in, %c0, %tile, %c0, %c1, %c8, %c16 {max_rows = 1 : i32, max_cols = 8 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, dst_x = 1 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    // CHECK: vc4kernel.vdw_store_vpm_fragment
    // CHECK-SAME: elem_bytes = 2
    // CHECK-SAME: subword = #vc4kernel.vpm_subword<packed>
    // CHECK-SAME: width = #vc4kernel.vpm_width<w16>
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %out, %c0, %full {elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, src_x = 1 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    // CHECK: vc4kernel.vdw_store_rect_from_vpm
    // CHECK-SAME: elem_bytes = 1
    // CHECK-SAME: width = #vc4kernel.vpm_width<w8>
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %out, %c0, %c1, %c16, %c16 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, src_x = 3 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
