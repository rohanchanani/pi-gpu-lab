module {
  vc4kernel.kernel @fragment_bitcast_roundtrip_vc4kernel(%bits_in : i32, %f_in : i32, %out_i32 : i32, %out_f32 : i32) attributes {
    public_name = "fragment_bitcast_roundtrip_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "bits_in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "f_in", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out_i32", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "out_f32", kind = "buffer", direction = "out", elem_type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %c2v = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_alu.add %lanes, %c2v {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %bits = vc4kernel.tmu_load_fragment %bits_in, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %as_f = vc4kernel.fragment_bitcast %bits : vector<16xi32> -> vector<16xf32>
    %round_i = vc4kernel.fragment_bitcast %as_f : vector<16xf32> -> vector<16xi32>
    %f = vc4kernel.tmu_load_fragment %f_in, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %as_i = vc4kernel.fragment_bitcast %f : vector<16xf32> -> vector<16xi32>
    %round_f = vc4kernel.fragment_bitcast %as_i : vector<16xi32> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out_i32, %lane_bytes, %round_i, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out_f32, %lane_bytes, %round_f, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
