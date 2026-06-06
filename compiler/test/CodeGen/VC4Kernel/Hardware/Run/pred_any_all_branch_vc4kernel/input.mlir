module {
  vc4kernel.kernel @pred_any_all_branch_vc4kernel(%out : i32, %threshold : i32, %offset_elems : i32) attributes {
    public_name = "pred_any_all_branch_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "threshold", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %none_base = arith.constant 1778397184 : i32
    %partial_base = arith.constant 1778393088 : i32
    %all_base = arith.constant 1778388992 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %threshold_v = vc4kernel.splat %threshold : i32 -> vector<16xi32>
    %mask = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %any = vc4kernel.pred.any %mask : !vc4kernel.pred<16> -> i1
    cf.cond_br %any, ^has_any, ^merge(%none_base : i32)
  ^has_any:
    %all = vc4kernel.pred.all %mask : !vc4kernel.pred<16> -> i1
    cf.cond_br %all, ^merge(%all_base : i32), ^merge(%partial_base : i32)
  ^merge(%base : i32):
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %offset_bytes = arith.shli %offset_elems, %c2 : i32
    %offset_vec = vc4kernel.splat %offset_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %offset_vec, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %base_v = vc4kernel.splat %base : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
