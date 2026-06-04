module {
  vc4kernel.kernel @dynamic_vdr_rect_runtime_dst_row_vc4kernel(%in : i32, %out : i32, %dst_row : i32, %active_rows : i32, %active_cols : i32, %pitch : i32) attributes {
    public_name = "dynamic_vdr_rect_runtime_dst_row_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "dst_row", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "pitch", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c6 = arith.constant 6 : i32
    %c7 = arith.constant 7 : i32
    %shift2 = arith.constant 2 : i32
    %row64 = arith.constant 64 : i32
    %row128 = arith.constant 128 : i32
    %row192 = arith.constant 192 : i32
    %row256 = arith.constant 256 : i32
    %row320 = arith.constant 320 : i32
    %row384 = arith.constant 384 : i32
    %row448 = arith.constant 448 : i32
    %base0 = arith.constant 1895825408 : i32
    %base1 = arith.constant 1895825664 : i32
    %base2 = arith.constant 1895825920 : i32
    %base3 = arith.constant 1895826176 : i32
    %base4 = arith.constant 1895826432 : i32
    %base5 = arith.constant 1895826688 : i32
    %base6 = arith.constant 1895826944 : i32
    %base7 = arith.constant 1895827200 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %shift2 : vector<16xi32>, i32 -> vector<16xi32>
    %row64v = vc4kernel.splat %row64 : i32 -> vector<16xi32>
    %row128v = vc4kernel.splat %row128 : i32 -> vector<16xi32>
    %row192v = vc4kernel.splat %row192 : i32 -> vector<16xi32>
    %row256v = vc4kernel.splat %row256 : i32 -> vector<16xi32>
    %row320v = vc4kernel.splat %row320 : i32 -> vector<16xi32>
    %row384v = vc4kernel.splat %row384 : i32 -> vector<16xi32>
    %row448v = vc4kernel.splat %row448 : i32 -> vector<16xi32>
    %offs1 = vc4kernel.fragment_add %row64v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs2 = vc4kernel.fragment_add %row128v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs3 = vc4kernel.fragment_add %row192v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs4 = vc4kernel.fragment_add %row256v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs5 = vc4kernel.fragment_add %row320v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs6 = vc4kernel.fragment_add %row384v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %offs7 = vc4kernel.fragment_add %row448v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %base0v = vc4kernel.splat %base0 : i32 -> vector<16xi32>
    %base1v = vc4kernel.splat %base1 : i32 -> vector<16xi32>
    %base2v = vc4kernel.splat %base2 : i32 -> vector<16xi32>
    %base3v = vc4kernel.splat %base3 : i32 -> vector<16xi32>
    %base4v = vc4kernel.splat %base4 : i32 -> vector<16xi32>
    %base5v = vc4kernel.splat %base5 : i32 -> vector<16xi32>
    %base6v = vc4kernel.splat %base6 : i32 -> vector<16xi32>
    %base7v = vc4kernel.splat %base7 : i32 -> vector<16xi32>
    %value0 = vc4kernel.fragment_add %base0v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value1 = vc4kernel.fragment_add %base1v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value2 = vc4kernel.fragment_add %base2v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value3 = vc4kernel.fragment_add %base3v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value4 = vc4kernel.fragment_add %base4v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value5 = vc4kernel.fragment_add %base5v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value6 = vc4kernel.fragment_add %base6v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value7 = vc4kernel.fragment_add %base7v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 8 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile, %c0, %value0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c1, %value1, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c2, %value2, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c3, %value3, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c4, %value4, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c5, %value5, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c6, %value6, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %c7, %value7, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdr_load_rect_to_vpm %in, %c0, %tile, %dst_row, %active_rows, %active_cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    %r0 = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r1 = vc4kernel.vpm_read_fragment %tile, %c1, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r2 = vc4kernel.vpm_read_fragment %tile, %c2, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r3 = vc4kernel.vpm_read_fragment %tile, %c3, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r4 = vc4kernel.vpm_read_fragment %tile, %c4, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r5 = vc4kernel.vpm_read_fragment %tile, %c5, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r6 = vc4kernel.vpm_read_fragment %tile, %c6, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r7 = vc4kernel.vpm_read_fragment %tile, %c7, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %r0, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs1, %r1, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs2, %r2, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs3, %r3, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs4, %r4, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs5, %r5, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs6, %r6, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs7, %r7, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
