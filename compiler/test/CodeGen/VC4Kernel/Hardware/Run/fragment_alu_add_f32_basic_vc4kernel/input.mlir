module {
  vc4kernel.kernel @fragment_alu_add_f32_basic_vc4kernel(%a : i32, %b : i32, %out : i32) attributes {
    public_name = "fragment_alu_add_f32_basic_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %av = vc4kernel.tmu_load_fragment %a, %lane_bytes, %full {memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %bv = vc4kernel.tmu_load_fragment %b, %lane_bytes, %full {memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %r0 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r1 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r2 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fmin>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r3 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fmax>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r4 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fminabs>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r5 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fmaxabs>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %r0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b1v = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %o1 = vc4kernel.fragment_alu.add %b1v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o1, %r1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b2v = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %o2 = vc4kernel.fragment_alu.add %b2v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o2, %r2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b3v = vc4kernel.fragment_const {value = dense<192> : vector<16xi32>} : vector<16xi32>
    %o3 = vc4kernel.fragment_alu.add %b3v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o3, %r3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b4v = vc4kernel.fragment_const {value = dense<256> : vector<16xi32>} : vector<16xi32>
    %o4 = vc4kernel.fragment_alu.add %b4v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o4, %r4, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b5v = vc4kernel.fragment_const {value = dense<320> : vector<16xi32>} : vector<16xi32>
    %o5 = vc4kernel.fragment_alu.add %b5v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o5, %r5, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
