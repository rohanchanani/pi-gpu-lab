module {
  vc4kernel.kernel @control_flow_tail_block_args_vc4kernel(%out : i32, %n : i32, %offset_elems : i32) attributes {
    public_name = "control_flow_tail_block_args_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %empty_base = arith.constant 1795174400 : i32
    %partial_base = arith.constant 1795170304 : i32
    %all_base = arith.constant 1795166208 : i32
    %tail = vc4kernel.pred.tail %c0, %n : i32, i32 -> !vc4kernel.pred<16>
    %any = vc4kernel.pred.any %tail : !vc4kernel.pred<16> -> i1
    cf.cond_br %any, ^has_any, ^merge(%empty_base : i32)
  ^has_any:
    %all = vc4kernel.pred.all %tail : !vc4kernel.pred<16> -> i1
    cf.cond_br %all, ^merge(%all_base : i32), ^merge(%partial_base : i32)
  ^merge(%base : i32):
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %offset_bytes = arith.shli %offset_elems, %c2 : i32
    %offset_vec = vc4kernel.splat %offset_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %offset_vec, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %base_v = vc4kernel.splat %base : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
