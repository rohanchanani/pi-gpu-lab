module {
  vc4kernel.kernel @blocked_gemm_vpm_tile_compute_micro_vc4kernel(%a : i32, %b : i32, %c : i32) attributes {
    public_name = "blocked_gemm_vpm_tile_compute_micro_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "c", kind = "buffer", direction = "out", elem_type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %a_pitch = arith.constant 16 : i32
    %b_pitch = arith.constant 64 : i32
    %zero_scalar = arith.constant 0.000000e+00 : f32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %zero = vc4kernel.splat %zero_scalar : f32 -> vector<16xf32>
    %a_tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %b_tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdr_load_rect_to_vpm %a, %c0, %a_tile, %c0, %c1, %c4, %a_pitch {max_rows = 1 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.vdr_load_rect_to_vpm %b, %c0, %b_tile, %c0, %c4, %c16, %b_pitch {max_rows = 4 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32

    %a_vec0 = vc4kernel.vpm_read_fragment %a_tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %k0v = vc4kernel.splat %c0 : i32 -> vector<16xi32>
    %k0p = vc4kernel.fragment_cmp %lanes, %k0v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %a0 = vc4kernel.fragment_reduce %a_vec0, %k0p {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b0 = vc4kernel.vpm_read_fragment %b_tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %p0 = vc4kernel.fragment_mul %a0, %b0 : vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %acc0 = vc4kernel.fragment_add %zero, %p0 : vector<16xf32>, vector<16xf32> -> vector<16xf32>

    %a_vec1 = vc4kernel.vpm_read_fragment %a_tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %k1v = vc4kernel.splat %c1 : i32 -> vector<16xi32>
    %k1p = vc4kernel.fragment_cmp %lanes, %k1v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %a1 = vc4kernel.fragment_reduce %a_vec1, %k1p {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b1 = vc4kernel.vpm_read_fragment %b_tile, %c1, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %p1 = vc4kernel.fragment_mul %a1, %b1 : vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %acc1 = vc4kernel.fragment_add %acc0, %p1 : vector<16xf32>, vector<16xf32> -> vector<16xf32>

    %a_vec2 = vc4kernel.vpm_read_fragment %a_tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %k2v = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %k2p = vc4kernel.fragment_cmp %lanes, %k2v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %a2 = vc4kernel.fragment_reduce %a_vec2, %k2p {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b2 = vc4kernel.vpm_read_fragment %b_tile, %c2, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %p2 = vc4kernel.fragment_mul %a2, %b2 : vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %acc2 = vc4kernel.fragment_add %acc1, %p2 : vector<16xf32>, vector<16xf32> -> vector<16xf32>

    %a_vec3 = vc4kernel.vpm_read_fragment %a_tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %k3v = vc4kernel.splat %c3 : i32 -> vector<16xi32>
    %k3p = vc4kernel.fragment_cmp %lanes, %k3v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %a3 = vc4kernel.fragment_reduce %a_vec3, %k3p {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b3 = vc4kernel.vpm_read_fragment %b_tile, %c3, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %p3 = vc4kernel.fragment_mul %a3, %b3 : vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %acc3 = vc4kernel.fragment_add %acc2, %p3 : vector<16xf32>, vector<16xf32> -> vector<16xf32>

    vc4kernel.vdw_store_fragment %c, %lane_bytes, %acc3, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
