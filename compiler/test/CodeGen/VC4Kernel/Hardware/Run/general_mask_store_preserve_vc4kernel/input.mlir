module {
  vc4kernel.kernel @general_mask_store_preserve_vc4kernel(%mask_in : i32, %out : i32) attributes {
    public_name = "general_mask_store_preserve_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "mask_in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %tag = arith.constant 1409286144 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %one_v = vc4kernel.splat %c1 : i32 -> vector<16xi32>
    %tag_v = vc4kernel.splat %tag : i32 -> vector<16xi32>
    %mask_values = vc4kernel.tmu_load_fragment %mask_in, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %mask = vc4kernel.fragment_cmp %mask_values, %one_v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %value = vc4kernel.fragment_add %tag_v, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %value, %mask : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
