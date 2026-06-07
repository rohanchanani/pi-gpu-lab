module {
  vc4kernel.kernel @fragment_sfu_recip_rsqrt_vc4kernel(%input : i32, %out : i32) attributes {
    public_name = "fragment_sfu_recip_rsqrt_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %safe0 = arith.constant 0 : i32
    %safe64 = arith.constant 64 : i32
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %row1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %row1_offsets = vc4kernel.fragment_alu.add %row1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %recip_raw = vc4kernel.tmu_load_fragment %input, %lane_bytes, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %rsqrt_raw = vc4kernel.tmu_load_fragment %input, %row1_offsets, %full, %safe64 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %zero_v = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %one_v = vc4kernel.fragment_const {value = dense<1.000000e+00> : vector<16xf32>} : vector<16xf32>
    %quarter_v = vc4kernel.fragment_const {value = dense<2.500000e-01> : vector<16xf32>} : vector<16xf32>
    %recip_is_zero = vc4kernel.fragment_cmp %recip_raw, %zero_v {predicate = #vc4kernel.cmp<oeq>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %recip_input = vc4kernel.fragment_select %recip_is_zero, %one_v, %recip_raw : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %rsqrt_positive = vc4kernel.fragment_cmp %rsqrt_raw, %zero_v {predicate = #vc4kernel.cmp<ogt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %rsqrt_input = vc4kernel.fragment_select %rsqrt_positive, %rsqrt_raw, %quarter_v : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %recip = vc4kernel.fragment_sfu %recip_input {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    %rsqrt = vc4kernel.fragment_sfu %rsqrt_input {kind = #vc4kernel.sfu_kind<rsqrt>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %recip, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %row1_offsets, %rsqrt, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
