module {
  vc4kernel.kernel @fragment_reduce_i32_broadcast_consumers_vc4kernel(%input : i32, %out : i32) attributes {
    public_name = "fragment_reduce_i32_broadcast_consumers_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %values = vc4kernel.tmu_load_fragment %input, %lane_bytes, %full {memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %k0 = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %k2 = vc4kernel.fragment_const {value = dense<2> : vector<16xi32>} : vector<16xi32>
    %k6 = vc4kernel.fragment_const {value = dense<6> : vector<16xi32>} : vector<16xi32>
    %k8 = vc4kernel.fragment_const {value = dense<8> : vector<16xi32>} : vector<16xi32>
    %k11 = vc4kernel.fragment_const {value = dense<11> : vector<16xi32>} : vector<16xi32>
    %eq0 = vc4kernel.fragment_cmp %lanes, %k0 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %eq2 = vc4kernel.fragment_cmp %lanes, %k2 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %eq6 = vc4kernel.fragment_cmp %lanes, %k6 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %eq11 = vc4kernel.fragment_cmp %lanes, %k11 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sparse0 = vc4kernel.pred.or %eq0, %eq2 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %sparse1 = vc4kernel.pred.or %eq6, %eq11 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %sparse = vc4kernel.pred.or %sparse0, %sparse1 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %low8 = vc4kernel.fragment_cmp %lanes, %k8 {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>

    %add = vc4kernel.fragment_reduce %values, %sparse {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %add, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %off1 = vc4kernel.fragment_alu.add %base1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %min_s = vc4kernel.fragment_reduce %values, %full {kind = #vc4kernel.reduce<min_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off1, %min_s, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base2 = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %off2 = vc4kernel.fragment_alu.add %base2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %max_u = vc4kernel.fragment_reduce %values, %sparse {kind = #vc4kernel.reduce<max_u>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off2, %max_u, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base3 = vc4kernel.fragment_const {value = dense<192> : vector<16xi32>} : vector<16xi32>
    %off3 = vc4kernel.fragment_alu.add %base3, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %xor = vc4kernel.fragment_reduce %values, %full {kind = #vc4kernel.reduce<bit_xor>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off3, %xor, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %base4 = vc4kernel.fragment_const {value = dense<256> : vector<16xi32>} : vector<16xi32>
    %off4 = vc4kernel.fragment_alu.add %base4, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %add_consumer = vc4kernel.fragment_alu.add %add, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off4, %add_consumer, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base5 = vc4kernel.fragment_const {value = dense<320> : vector<16xi32>} : vector<16xi32>
    %off5 = vc4kernel.fragment_alu.add %base5, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %select_consumer = vc4kernel.fragment_select %low8, %min_s, %max_u : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off5, %select_consumer, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
