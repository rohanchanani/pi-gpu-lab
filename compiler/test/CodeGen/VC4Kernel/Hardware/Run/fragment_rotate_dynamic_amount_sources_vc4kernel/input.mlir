module {
  vc4kernel.kernel @fragment_rotate_dynamic_amount_sources_vc4kernel(%out : i32, %stride : i32, %bias : i32) attributes {
    public_name = "fragment_rotate_dynamic_amount_sources_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "stride", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "bias", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %request = vc4kernel.program_id {axis = 0 : i32} : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %amount_mul = arith.muli %request, %stride : i32
    %amount = arith.addi %amount_mul, %bias : i32
    %row_base_index = arith.shli %request, %c4 : i32
    %row_base_bytes = arith.shli %request, %c6 : i32
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %row_base_bytes_v = vc4kernel.splat %row_base_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %row_base_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tag_v = vc4kernel.fragment_const {value = dense<1694498816> : vector<16xi32>} : vector<16xi32>
    %row_base_index_v = vc4kernel.splat %row_base_index : i32 -> vector<16xi32>
    %base = vc4kernel.fragment_alu.add %tag_v, %row_base_index_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %rot = vc4kernel.fragment_rotate %value, %amount : vector<16xi32>, i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %rot, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
