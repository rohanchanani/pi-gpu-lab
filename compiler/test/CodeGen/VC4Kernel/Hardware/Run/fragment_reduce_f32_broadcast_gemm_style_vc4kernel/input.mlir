module {
  vc4kernel.kernel @fragment_reduce_f32_broadcast_gemm_style_vc4kernel(%input : i32, %out : i32) attributes {
    public_name = "fragment_reduce_f32_broadcast_gemm_style_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %p7_safe0 = arith.constant 0 : i32
    %values = vc4kernel.tmu_load_fragment %input, %lane_bytes, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %k2 = vc4kernel.fragment_const {value = dense<2> : vector<16xi32>} : vector<16xi32>
    %k9 = vc4kernel.fragment_const {value = dense<9> : vector<16xi32>} : vector<16xi32>
    %k14 = vc4kernel.fragment_const {value = dense<14> : vector<16xi32>} : vector<16xi32>
    %p2 = vc4kernel.fragment_cmp %lanes, %k2 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %p9 = vc4kernel.fragment_cmp %lanes, %k9 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %p14 = vc4kernel.fragment_cmp %lanes, %k14 {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %b2 = vc4kernel.fragment_reduce %values, %p2 {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %b2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %base1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %off1 = vc4kernel.fragment_alu.add %base1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b9 = vc4kernel.fragment_reduce %values, %p9 {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %off1, %b9, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %base2 = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %off2 = vc4kernel.fragment_alu.add %base2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b14 = vc4kernel.fragment_reduce %values, %p14 {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %off2, %b14, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
