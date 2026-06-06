module {
  vc4kernel.kernel @tail_and_tail_canonicalized_vc4kernel(%out : i32, %n : i32, %cap : i32) attributes {
    public_name = "tail_and_tail_canonicalized_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "cap", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %tag = arith.constant 1711276032 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %lanes, %byte_offsets_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tag_v = vc4kernel.splat %tag : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %tag_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tail_n = vc4kernel.pred.tail %c0, %n : i32, i32 -> !vc4kernel.pred<16>
    %tail_cap = vc4kernel.pred.tail %c0, %cap : i32, i32 -> !vc4kernel.pred<16>
    %mask = vc4kernel.pred.and %tail_n, %tail_cap : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %mask : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
