module {
  vc4kernel.kernel @vpm_multi_alloc_rows_vc4kernel(%out : i32) attributes {
    public_name = "vpm_multi_alloc_rows_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %off64 = arith.constant 64 : i32
    %off128 = arith.constant 128 : i32
    %base0 = arith.constant 1962934272 : i32
    %base1 = arith.constant 1979711488 : i32
    %base2 = arith.constant 1996488704 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %base0_v = vc4kernel.splat %base0 : i32 -> vector<16xi32>
    %base1_v = vc4kernel.splat %base1 : i32 -> vector<16xi32>
    %base2_v = vc4kernel.splat %base2 : i32 -> vector<16xi32>
    %value0 = vc4kernel.fragment_add %base0_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value1 = vc4kernel.fragment_add %base1_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %value2 = vc4kernel.fragment_add %base2_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %tile0 = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %tile1 = vc4kernel.vpm_alloc {rows = 2 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vpm_write_fragment %tile0, %c0, %value0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile1, %c0, %value1, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile1, %c1, %value2, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %read0 = vc4kernel.vpm_read_fragment %tile0, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %read1 = vc4kernel.vpm_read_fragment %tile1, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %read2 = vc4kernel.vpm_read_fragment %tile1, %c1, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %row1_base = vc4kernel.splat %off64 : i32 -> vector<16xi32>
    %row2_base = vc4kernel.splat %off128 : i32 -> vector<16xi32>
    %row1_offsets = vc4kernel.fragment_add %row1_base, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %row2_offsets = vc4kernel.fragment_add %row2_base, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %read0, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %row1_offsets, %read1, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %row2_offsets, %read2, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
