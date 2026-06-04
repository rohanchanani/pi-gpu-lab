module {
  vc4kernel.kernel @dynamic_vpm_read_runtime_row_vc4kernel(%out : i32, %row : i32) attributes {
    public_name = "dynamic_vpm_read_runtime_row_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"}
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
    %base0 = arith.constant 1929379840 : i32
    %base1 = arith.constant 1929380096 : i32
    %base2 = arith.constant 1929380352 : i32
    %base3 = arith.constant 1929380608 : i32
    %base4 = arith.constant 1929380864 : i32
    %base5 = arith.constant 1929381120 : i32
    %base6 = arith.constant 1929381376 : i32
    %base7 = arith.constant 1929381632 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %shift2 : vector<16xi32>, i32 -> vector<16xi32>
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
    %read = vc4kernel.vpm_read_fragment %tile, %row, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %read, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
