module {
  vc4kernel.kernel @i32_mul32_fallback_vc4kernel(%out : i32, %lhs_base : i32, %rhs_base : i32, %offset_elems : i32) attributes {
    public_name = "i32_mul32_fallback_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "lhs_base", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "rhs_base", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %offset_bytes = arith.shli %offset_elems, %c2 : i32
    %offset_vec = vc4kernel.splat %offset_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %offset_vec, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %lhs_base_v = vc4kernel.splat %lhs_base : i32 -> vector<16xi32>
    %rhs_base_v = vc4kernel.splat %rhs_base : i32 -> vector<16xi32>
    %lhs = vc4kernel.fragment_alu.add %lhs_base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %rhs = vc4kernel.fragment_alu.add %rhs_base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_mask = vc4kernel.fragment_const {value = dense<65535> : vector<16xi32>} : vector<16xi32>
    %value_shift = vc4kernel.fragment_const {value = dense<16> : vector<16xi32>} : vector<16xi32>
    %value_lhs_lo = vc4kernel.fragment_alu.add %lhs, %value_mask {opcode = #vc4kernel.add_alu_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_rhs_lo = vc4kernel.fragment_alu.add %rhs, %value_mask {opcode = #vc4kernel.add_alu_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_lhs_hi = vc4kernel.fragment_alu.add %lhs, %value_shift {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_rhs_hi = vc4kernel.fragment_alu.add %rhs, %value_shift {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_lo_lo = vc4kernel.fragment_alu.mul %value_lhs_lo, %value_rhs_lo {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_lo_hi = vc4kernel.fragment_alu.mul %value_lhs_lo, %value_rhs_hi {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_hi_lo = vc4kernel.fragment_alu.mul %value_lhs_hi, %value_rhs_lo {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_cross = vc4kernel.fragment_alu.add %value_lo_hi, %value_hi_lo {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value_cross_shifted = vc4kernel.fragment_alu.add %value_cross, %value_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %value_lo_lo, %value_cross_shifted {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
