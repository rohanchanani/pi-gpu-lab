module {
  vc4kernel.kernel @fragment_rotate_dynamic_i32_vc4kernel(%out : i32, %amount : i32, %n : i32, %base_value : i32) attributes {
    public_name = "fragment_rotate_dynamic_i32_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "amount", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "base_value", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %zero = arith.constant 0 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %tail = vc4kernel.pred.tail %zero, %n : i32, i32 -> !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %base_vec = vc4kernel.splat %base_value : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %base_vec, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %rot = vc4kernel.fragment_rotate %value, %amount : vector<16xi32>, i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %rot, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
