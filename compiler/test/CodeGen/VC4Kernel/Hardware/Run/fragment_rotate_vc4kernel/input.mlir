module {
  vc4kernel.kernel @fragment_rotate_vc4kernel(%out : i32, %offset_elems : i32, %base_value : i32) attributes {
    public_name = "fragment_rotate_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "base_value", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_alu.add %lanes, %lane_bytes_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offset_bytes = arith.shli %offset_elems, %c2 : i32
    %offset_vec = vc4kernel.splat %offset_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %offset_vec, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %base_vec = vc4kernel.splat %base_value : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %base_vec, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %rot = vc4kernel.fragment_rotate %value {amount = 3 : i32} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %rot, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
