module {
  vc4kernel.kernel @fragment_reduce_i32_signed_minmax_vc4kernel(%input : i32, %out : i32) attributes {
    public_name = "fragment_reduce_i32_signed_minmax_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %empty = vc4kernel.pred.empty : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %values = vc4kernel.tmu_load_fragment %input, %lane_bytes, %full {memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %k1 = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %k4 = vc4kernel.fragment_const {value = dense<4> : vector<16xi32>} : vector<16xi32>
    %k7 = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %k10 = vc4kernel.fragment_const {value = dense<10> : vector<16xi32>} : vector<16xi32>
    %k15 = vc4kernel.fragment_const {value = dense<15> : vector<16xi32>} : vector<16xi32>
    %eq1 = vc4kernel.fragment_cmp %lanes, %k1 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %eq4 = vc4kernel.fragment_cmp %lanes, %k4 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %one_hot = vc4kernel.fragment_cmp %lanes, %k7 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %eq10 = vc4kernel.fragment_cmp %lanes, %k10 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %eq15 = vc4kernel.fragment_cmp %lanes, %k15 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sparse0 = vc4kernel.pred.or %eq1, %eq4 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %sparse1 = vc4kernel.pred.or %eq10, %eq15 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %sparse = vc4kernel.pred.or %sparse0, %sparse1 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>

    %r0 = vc4kernel.fragment_reduce %values, %full {kind = #vc4kernel.reduce<min_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %r0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %off1 = vc4kernel.fragment_alu.add %base1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r1 = vc4kernel.fragment_reduce %values, %empty {kind = #vc4kernel.reduce<min_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off1, %r1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base2 = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %off2 = vc4kernel.fragment_alu.add %base2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r2 = vc4kernel.fragment_reduce %values, %one_hot {kind = #vc4kernel.reduce<min_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off2, %r2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base3 = vc4kernel.fragment_const {value = dense<192> : vector<16xi32>} : vector<16xi32>
    %off3 = vc4kernel.fragment_alu.add %base3, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r3 = vc4kernel.fragment_reduce %values, %sparse {kind = #vc4kernel.reduce<min_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off3, %r3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %base4 = vc4kernel.fragment_const {value = dense<256> : vector<16xi32>} : vector<16xi32>
    %off4 = vc4kernel.fragment_alu.add %base4, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r4 = vc4kernel.fragment_reduce %values, %full {kind = #vc4kernel.reduce<max_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off4, %r4, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base5 = vc4kernel.fragment_const {value = dense<320> : vector<16xi32>} : vector<16xi32>
    %off5 = vc4kernel.fragment_alu.add %base5, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r5 = vc4kernel.fragment_reduce %values, %empty {kind = #vc4kernel.reduce<max_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off5, %r5, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base6 = vc4kernel.fragment_const {value = dense<384> : vector<16xi32>} : vector<16xi32>
    %off6 = vc4kernel.fragment_alu.add %base6, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r6 = vc4kernel.fragment_reduce %values, %one_hot {kind = #vc4kernel.reduce<max_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off6, %r6, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %base7 = vc4kernel.fragment_const {value = dense<448> : vector<16xi32>} : vector<16xi32>
    %off7 = vc4kernel.fragment_alu.add %base7, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r7 = vc4kernel.fragment_reduce %values, %sparse {kind = #vc4kernel.reduce<max_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off7, %r7, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
