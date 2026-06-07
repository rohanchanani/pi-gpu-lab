module {
  vc4kernel.kernel @fragment_broadcast_lane_composite_vc4kernel(%bits_in : i32, %f_bits_in : i32, %finite_f32_in : i32, %out_i32 : i32, %out_f32_bits : i32, %out_finite_f32 : i32, %lane_seed : i32) attributes {
    public_name = "fragment_broadcast_lane_composite_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "bits_in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "f_bits_in", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "finite_f32_in", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out_i32", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "out_f32_bits", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "out_finite_f32", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "lane_seed", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %selected_lane = arith.addi %lane_seed, %c1 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %selected_v = vc4kernel.splat %selected_lane : i32 -> vector<16xi32>
    %one_hot = vc4kernel.fragment_cmp %lanes, %selected_v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>

    %safe_i32 = arith.constant 0 : i32
    %payload = vc4kernel.tmu_load_fragment %bits_in, %lane_bytes, %full, %safe_i32 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %masked_payload = vc4kernel.fragment_select %one_hot, %payload, %zero : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %broadcast = vc4kernel.fragment_reduce %masked_payload, %full {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out_i32, %lane_bytes, %broadcast, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %safe_f32 = arith.constant 0 : i32
    %f_payload = vc4kernel.tmu_load_fragment %f_bits_in, %lane_bytes, %full, %safe_f32 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %f_as_i = vc4kernel.fragment_bitcast %f_payload : vector<16xf32> -> vector<16xi32>
    %masked_f = vc4kernel.fragment_select %one_hot, %f_as_i, %zero : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %broadcast_f_bits = vc4kernel.fragment_reduce %masked_f, %full {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %broadcast_f = vc4kernel.fragment_bitcast %broadcast_f_bits : vector<16xi32> -> vector<16xf32>
    %roundtrip_f_bits = vc4kernel.fragment_bitcast %broadcast_f : vector<16xf32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out_f32_bits, %lane_bytes, %roundtrip_f_bits, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %safe_finite = arith.constant 0 : i32
    %finite_payload = vc4kernel.tmu_load_fragment %finite_f32_in, %lane_bytes, %full, %safe_finite {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %zero_f32 = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %masked_finite = vc4kernel.fragment_select %one_hot, %finite_payload, %zero_f32 : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %finite_broadcast = vc4kernel.fragment_reduce %masked_finite, %full {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out_finite_f32, %lane_bytes, %finite_broadcast, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
