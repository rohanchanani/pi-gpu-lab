module {
  vc4kernel.kernel @fragment_rotate_dynamic_f32_vc4kernel(%input : i32, %out : i32, %amount : i32, %n : i32) attributes {
    public_name = "fragment_rotate_dynamic_f32_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "amount", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %zero = arith.constant 0 : i32
    %safe0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tail = vc4kernel.pred.tail %zero, %n : i32, i32 -> !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %values = vc4kernel.tmu_load_fragment %input, %lane_bytes, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %one = vc4kernel.fragment_const {value = dense<1.000000e+00> : vector<16xf32>} : vector<16xf32>
    %threshold = vc4kernel.fragment_const {value = dense<1.000000e+01> : vector<16xf32>} : vector<16xf32>
    %rot = vc4kernel.fragment_rotate %values, %amount : vector<16xf32>, i32 -> vector<16xf32>
    %plus = vc4kernel.fragment_alu.add %rot, %one {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %gt = vc4kernel.fragment_cmp %plus, %threshold {predicate = #vc4kernel.cmp<ogt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %selected = vc4kernel.fragment_select %gt, %plus, %rot : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %selected, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
