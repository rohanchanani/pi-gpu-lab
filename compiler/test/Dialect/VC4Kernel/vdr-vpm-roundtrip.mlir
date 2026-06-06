// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @vdr_vpm(%ptr : i32) attributes {
    public_name = "vdr_vpm",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "ptr", kind = "buffer", direction = "inout", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: vc4kernel.vdr_load_to_vpm
    // CHECK-SAME: cols = 16
    // CHECK-SAME: elem_bytes = 4
    // CHECK-SAME: global_stride_bytes = 64
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<horizontal>
    // CHECK-SAME: rows = 1
    vc4kernel.vdr_load_to_vpm %ptr, %c0, %tile, %c0 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32
    // CHECK: vc4kernel.vdw_store_vpm_fragment
    // CHECK-SAME: elem_bytes = 4
    // CHECK-SAME: orientation = #vc4kernel.vpm_orientation<vertical>
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %ptr, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 2 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
