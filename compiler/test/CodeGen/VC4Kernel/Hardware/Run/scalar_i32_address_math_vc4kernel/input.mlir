module {
  vc4kernel.kernel @scalar_i32_address_math_vc4kernel(%out : i32, %selector : i32, %stride : i32) attributes {
    public_name = "scalar_i32_address_math_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "selector", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "stride", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c7 = arith.constant 7 : i32
    %c85 = arith.constant 85 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>

    %masked = arith.andi %selector, %c7 : i32
    %group = arith.shrui %selector, %c3 : i32
    %group_scaled = arith.shli %group, %c1 : i32
    %scaled = arith.muli %masked, %stride : i32
    %addr_elems = arith.addi %scaled, %group_scaled : i32
    %addr_bytes = arith.shli %addr_elems, %c2 : i32
    %addr_vec = vc4kernel.splat %addr_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %addr_vec, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    %tag = arith.xori %selector, %c85 : i32
    %value_scalar = arith.addi %addr_elems, %tag : i32
    %value = vc4kernel.splat %value_scalar : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
