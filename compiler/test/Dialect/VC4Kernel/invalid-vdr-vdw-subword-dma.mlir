// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad(%ptr : i32) attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "inout", elem_type = "u8"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: VDR DMA laned subword mode is not supported by VC4 hardware
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %c0, %c1, %c4, %c4 {max_rows = 1 : i32, max_cols = 4 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<laned>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_elem(%ptr : i32) attributes {
    public_name = "bad_elem",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "inout", elem_type = "u8"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: elem_bytes must match VPM width (2 for this mode)
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %ptr, %c0, %c1, %c4, %c4 {max_rows = 1 : i32, max_cols = 4 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_vertical(%ptr : i32) attributes {
    public_name = "bad_vertical",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "ptr", kind = "buffer", direction = "inout", elem_type = "u8"}],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    // CHECK: vertical subword VDR DMA is unproven/deferred in P12
    vc4kernel.vdr_load_rect_to_vpm %ptr, %c0, %tile, %c0, %c1, %c4, %c4 {max_rows = 1 : i32, max_cols = 4 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.return
  }
}
