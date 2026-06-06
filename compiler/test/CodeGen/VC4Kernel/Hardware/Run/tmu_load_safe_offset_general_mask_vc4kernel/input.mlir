module {
  vc4kernel.kernel @tmu_load_safe_offset_general_mask_vc4kernel(%input : i32, %out : i32) attributes {
    public_name = "tmu_load_safe_offset_general_mask_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %three = vc4kernel.fragment_const {value = dense<3> : vector<16xi32>} : vector<16xi32>
    %five = vc4kernel.fragment_const {value = dense<5> : vector<16xi32>} : vector<16xi32>
    %twelve = vc4kernel.fragment_const {value = dense<12> : vector<16xi32>} : vector<16xi32>
    %below3 = vc4kernel.fragment_cmp %lanes, %three {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %eq5 = vc4kernel.fragment_cmp %lanes, %five {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %below12 = vc4kernel.fragment_cmp %lanes, %twelve {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %at_or_above12 = vc4kernel.pred.not %below12 : !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %left = vc4kernel.pred.or %below3, %eq5 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %mask = vc4kernel.pred.or %left, %at_or_above12 : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %store_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %poison_offsets = vc4kernel.fragment_const {value = dense<[268435456, 268435460, 268435464, 268435468, 268435472, 268435476, 268435480, 268435484, 268435488, 268435492, 268435496, 268435500, 268435504, 268435508, 268435512, 268435516]> : vector<16xi32>} : vector<16xi32>
    %request_offsets = vc4kernel.fragment_select %mask, %store_offsets, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %safe = arith.constant 0 : i32
    %loaded = vc4kernel.tmu_load_fragment %input, %request_offsets, %mask, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %store_offsets, %loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
