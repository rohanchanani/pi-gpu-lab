module {
  vc4kernel.kernel @dynamic_vpm_64_capacity_rect_tiles_vc4kernel(%in0 : i32, %in16 : i32, %in32 : i32, %in48 : i32, %out0 : i32, %out16 : i32, %out32 : i32, %out48 : i32, %active_rows : i32, %active_cols : i32, %pitch : i32, %stride : i32) attributes {
    public_name = "dynamic_vpm_64_capacity_rect_tiles_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in0", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "in16", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "in32", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "in48", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out0", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "out16", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "out32", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "out48", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "active_rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "pitch", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "stride", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %tile0 = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %tile16 = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %tile32 = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %tile48 = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdr_load_rect_to_vpm %in0, %c0, %tile0, %c0, %active_rows, %active_cols, %pitch {max_rows = 16 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.vdr_load_rect_to_vpm %in16, %c0, %tile16, %c0, %active_rows, %active_cols, %pitch {max_rows = 16 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.vdr_load_rect_to_vpm %in32, %c0, %tile32, %c0, %active_rows, %active_cols, %pitch {max_rows = 16 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.vdr_load_rect_to_vpm %in48, %c0, %tile48, %c0, %active_rows, %active_cols, %pitch {max_rows = 16 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.vdw_store_rect_from_vpm %tile0, %c0, %out0, %c0, %active_rows, %active_cols, %stride {max_rows = 16 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.vdw_store_rect_from_vpm %tile16, %c0, %out16, %c0, %active_rows, %active_cols, %stride {max_rows = 16 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.vdw_store_rect_from_vpm %tile32, %c0, %out32, %c0, %active_rows, %active_cols, %stride {max_rows = 16 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.vdw_store_rect_from_vpm %tile48, %c0, %out48, %c0, %active_rows, %active_cols, %stride {max_rows = 16 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
