module {
  vc4kernel.kernel @vdw_tail_store_vc4kernel(%out : i32, %n : i32) attributes {
    public_name = "vdw_tail_store_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    %tag = arith.constant 1677721600 : i32
    %request = vc4kernel.program_id {axis = 0 : i32} : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %base_index = arith.shli %request, %c4 : i32
    %base_bytes = arith.shli %request, %c6 : i32
    %tail = vc4kernel.pred.tail %base_index, %n : i32, i32 -> !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %base_bytes_v = vc4kernel.splat %base_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_add %base_bytes_v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %tag_v = vc4kernel.splat %tag : i32 -> vector<16xi32>
    %base_index_v = vc4kernel.splat %base_index : i32 -> vector<16xi32>
    %base_value = vc4kernel.fragment_add %tag_v, %base_index_v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %values = vc4kernel.fragment_add %base_value, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %values, %tail : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
