module {
  vc4kernel.kernel @vdw_empty_store_preserve_vc4kernel(%out : i32) attributes {
    public_name = "vdw_empty_store_preserve_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %tag = arith.constant 1694498816 : i32
    %empty = vc4kernel.pred.empty : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %tag_v = vc4kernel.splat %tag : i32 -> vector<16xi32>
    %values = vc4kernel.fragment_add %tag_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %values, %empty : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
