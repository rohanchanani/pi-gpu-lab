module {
  vc4kernel.kernel @program_id_writeback_vc4kernel(%out : i32) attributes {
    public_name = "program_id_writeback_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c6 = arith.constant 6 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %request = vc4kernel.program_id {axis = 0 : i32} : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %request_bytes = arith.shli %request, %c6 : i32
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %request_bytes_v = vc4kernel.splat %request_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %request_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tag_v = vc4kernel.fragment_const {value = dense<1644167168> : vector<16xi32>} : vector<16xi32>
    %request_v = vc4kernel.splat %request : i32 -> vector<16xi32>
    %request_scaled_shift = vc4kernel.fragment_const {value = dense<4> : vector<16xi32>} : vector<16xi32>
    %request_scaled = vc4kernel.fragment_alu.add %request_v, %request_scaled_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %base = vc4kernel.fragment_alu.add %tag_v, %request_scaled {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %values = vc4kernel.fragment_alu.add %base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %values, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
