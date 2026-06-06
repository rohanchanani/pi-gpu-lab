module {
  vc4kernel.kernel @vdr_vdw_horizontal_32_roundtrip_vc4kernel(%in : i32, %out : i32) attributes {
    public_name = "vdr_vdw_horizontal_32_roundtrip_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %off64 = arith.constant 64 : i32
    %off128 = arith.constant 128 : i32
    %off192 = arith.constant 192 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %c0 {rows = 4 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %out, %c0, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.vdw_store_vpm_fragment %tile, %c1, %out, %off64, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.vdw_store_vpm_fragment %tile, %c2, %out, %off128, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.vdw_store_vpm_fragment %tile, %c3, %out, %off192, %full {elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
