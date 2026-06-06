module {
  vc4kernel.kernel @fragment_const_per_lane_i2_vc4kernel(%out : i32) attributes {
    public_name = "fragment_const_per_lane_i2_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %u2 = vc4kernel.fragment_const {value = dense<[0, 1, 2, 3, 0, 1, 2, 3, 3, 2, 1, 0, 0, 2, 1, 3]> : vector<16xi32>} : vector<16xi32>
    %s2 = vc4kernel.fragment_const {value = dense<[-2, -1, 0, 1, -2, -1, 0, 1, 1, 0, -1, -2, -2, 0, -1, 1]> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %u2, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %o1 = vc4kernel.fragment_alu.add %b1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o1, %s2, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
