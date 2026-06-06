module {
  vc4kernel.kernel @general_mask_store_preserve_vc4kernel(%mask_in : i32, %out : i32) attributes {
    public_name = "general_mask_store_preserve_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "mask_in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %one_v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %tag_v = vc4kernel.fragment_const {value = dense<1409286144> : vector<16xi32>} : vector<16xi32>
    %mask_values = vc4kernel.tmu_load_fragment %mask_in, %lane_bytes, %full {memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %mask = vc4kernel.fragment_cmp %mask_values, %one_v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %value = vc4kernel.fragment_alu.add %tag_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %value, %mask {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
