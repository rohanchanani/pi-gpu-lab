module {
  vc4kernel.kernel @fragment_f16_pack_from_f32_vc4kernel(%input : i32, %out : i32) attributes {
    public_name = "fragment_f16_pack_from_f32_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %safe0 = arith.constant 0 : i32
    %values = vc4kernel.tmu_load_fragment %input, %lane_bytes, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %computed = vc4kernel.fragment_alu.add %values, %zero {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %packed = vc4kernel.fragment_pack %computed {dest = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<from_f32>} : vector<16xf32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %packed, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
