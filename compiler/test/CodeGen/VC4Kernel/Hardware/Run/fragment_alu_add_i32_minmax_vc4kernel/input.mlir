module {
  vc4kernel.kernel @fragment_alu_add_i32_minmax_vc4kernel(%a : i32, %b : i32, %out : i32) attributes {
    public_name = "fragment_alu_add_i32_minmax_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %b1 = arith.constant 64 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %c2v = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_alu.add %lanes, %c2v {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %av = vc4kernel.tmu_load_fragment %a, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %bv = vc4kernel.tmu_load_fragment %b, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %minv = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<min>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %maxv = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<max>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %minv, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b1v = vc4kernel.splat %b1 : i32 -> vector<16xi32>
    %o1 = vc4kernel.fragment_alu.add %b1v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o1, %maxv, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
