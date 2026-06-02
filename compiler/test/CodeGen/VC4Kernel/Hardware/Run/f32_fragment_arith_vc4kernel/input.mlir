module {
  vc4kernel.kernel @f32_fragment_arith_vc4kernel(%out : i32, %alpha : f32, %beta : f32, %offset_elems : i32) attributes {
    public_name = "f32_fragment_arith_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "alpha", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "beta", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %offset_bytes = arith.shli %offset_elems, %c2 : i32
    %offset_vec = vc4kernel.splat %offset_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_add %offset_vec, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a = vc4kernel.splat %alpha : f32 -> vector<16xf32>
    %b = vc4kernel.splat %beta : f32 -> vector<16xf32>
    %sum = vc4kernel.fragment_add %a, %b : vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %diff = vc4kernel.fragment_sub %sum, %b : vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %value = vc4kernel.fragment_mul %diff, %sum : vector<16xf32>, vector<16xf32> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
