module {
  vc4kernel.kernel @fragment_rotate_dynamic_tmu_sfu_pack_interaction_vc4kernel(%input_f32 : i32, %input_i32 : i32, %out_f32 : i32, %audit_i32 : i32, %amount : i32) attributes {
    public_name = "fragment_rotate_dynamic_tmu_sfu_pack_interaction_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input_f32", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "input_i32", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out_f32", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "audit_i32", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "amount", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %safe0 = arith.constant 0 : i32
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %raw_f32 = vc4kernel.tmu_load_fragment %input_f32, %lane_bytes, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %recip = vc4kernel.fragment_sfu %raw_f32 {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    %rot_f32 = vc4kernel.fragment_rotate %recip, %amount : vector<16xf32>, i32 -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out_f32, %lane_bytes, %rot_f32, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>

    %raw_i32 = vc4kernel.tmu_load_fragment %input_i32, %lane_bytes, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %u8 = vc4kernel.fragment_unpack %raw_i32 {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>
    %packed = vc4kernel.fragment_pack %u8 {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    %round = vc4kernel.fragment_unpack %packed {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>
    %rot_i32 = vc4kernel.fragment_rotate %round, %amount : vector<16xi32>, i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %audit_i32, %lane_bytes, %rot_i32, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
