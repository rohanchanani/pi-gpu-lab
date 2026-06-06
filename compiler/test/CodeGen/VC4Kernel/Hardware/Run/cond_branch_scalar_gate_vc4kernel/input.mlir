module {
  vc4kernel.kernel @cond_branch_scalar_gate_vc4kernel(%out : i32, %control : i32, %offset_elems : i32) attributes {
    public_name = "cond_branch_scalar_gate_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "control", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %c8 = arith.constant 8 : i32
    %tag = arith.constant 1761607680 : i32
    %skip = arith.cmpi eq, %control, %c0 : i32
    cf.cond_br %skip, ^done, ^write
  ^write:
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_alu.add %lanes, %lane_bytes_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offset_bytes = arith.shli %offset_elems, %c2 : i32
    %offset_vec = vc4kernel.splat %offset_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %offset_vec, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %control_scaled = arith.shli %control, %c8 : i32
    %tag_control = arith.addi %tag, %control_scaled : i32
    %base = vc4kernel.splat %tag_control : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    cf.br ^done
  ^done:
    vc4kernel.return
  }
}
