// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @dynamic_rect_transfer(%ptr : i32, %active_rows : i32, %active_cols : i32, %pitch : i32) attributes {
    public_name = "dynamic_rect_transfer",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "active_rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "pitch", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: vc4kernel.vdr_load_rect_to_vpm
    // CHECK-SAME: elem_bytes = 4
    // CHECK-SAME: max_cols = 8
    // CHECK-SAME: max_rows = 4
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %c0, %active_rows, %active_cols, %pitch {max_rows = 4 : i32, max_cols = 8 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    // CHECK: vc4kernel.vdw_store_rect_from_vpm
    // CHECK-SAME: elem_bytes = 4
    // CHECK-SAME: max_cols = 8
    // CHECK-SAME: max_rows = 4
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %ptr, %c0, %active_rows, %active_cols, %pitch {max_rows = 4 : i32, max_cols = 8 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 3 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
