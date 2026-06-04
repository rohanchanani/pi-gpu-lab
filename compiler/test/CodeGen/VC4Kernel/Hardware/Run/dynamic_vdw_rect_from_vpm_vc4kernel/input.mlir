module {
  vc4kernel.kernel @dynamic_vdw_rect_from_vpm_vc4kernel(%out : i32, %active_rows : i32, %active_cols : i32, %stride : i32) attributes {
    public_name = "dynamic_vdw_rect_from_vpm_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "active_rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "stride", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %base0 = arith.constant 1660944384 : i32
    %base1 = arith.constant 1660944640 : i32
    %base2 = arith.constant 1660944896 : i32
    %base3 = arith.constant 1660945152 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %base0v = vc4kernel.splat %base0 : i32 -> vector<16xi32>
    %base1v = vc4kernel.splat %base1 : i32 -> vector<16xi32>
    %base2v = vc4kernel.splat %base2 : i32 -> vector<16xi32>
    %base3v = vc4kernel.splat %base3 : i32 -> vector<16xi32>
    %value0 = vc4kernel.fragment_add %base0v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value1 = vc4kernel.fragment_add %base1v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value2 = vc4kernel.fragment_add %base2v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value3 = vc4kernel.fragment_add %base3v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile, %c0, %value0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c1, %value1, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c2, %value2, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c3, %value3, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_rect_from_vpm %tile, %c0, %out, %c0, %active_rows, %active_cols, %stride {max_rows = 4 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
