module {
  vc4kernel.kernel @tmu_load_safe_offset_rect_vc4kernel(%input : i32, %out : i32) attributes {
    public_name = "tmu_load_safe_offset_rect_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c7 = arith.constant 7 : i32
    %c8 = arith.constant 8 : i32
    %c15 = arith.constant 15 : i32
    %c16 = arith.constant 16 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %rect_full = vc4kernel.pred.rect %c0, %c1, %c0, %c16 : i32, i32, i32, i32 -> !vc4kernel.pred<16>
    %rect_center = vc4kernel.pred.rect %c0, %c1, %c3, %c8 : i32, i32, i32, i32 -> !vc4kernel.pred<16>
    %rect_empty_row = vc4kernel.pred.rect %c1, %c1, %c3, %c8 : i32, i32, i32, i32 -> !vc4kernel.pred<16>
    %rect_pair = vc4kernel.pred.rect %c0, %c1, %c7, %c2 : i32, i32, i32, i32 -> !vc4kernel.pred<16>
    %rect_last = vc4kernel.pred.rect %c0, %c1, %c15, %c1 : i32, i32, i32, i32 -> !vc4kernel.pred<16>
    %real_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %poison_offsets = vc4kernel.fragment_const {value = dense<[268435456, 268435460, 268435464, 268435468, 268435472, 268435476, 268435480, 268435484, 268435488, 268435492, 268435496, 268435500, 268435504, 268435508, 268435512, 268435516]> : vector<16xi32>} : vector<16xi32>
    %center_offsets = vc4kernel.fragment_select %rect_center, %real_offsets, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %empty_offsets = vc4kernel.fragment_select %rect_empty_row, %real_offsets, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %pair_offsets = vc4kernel.fragment_select %rect_pair, %real_offsets, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %last_offsets = vc4kernel.fragment_select %rect_last, %real_offsets, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %safe = arith.constant 0 : i32
    %full_loaded = vc4kernel.tmu_load_fragment %input, %real_offsets, %rect_full, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %center_loaded = vc4kernel.tmu_load_fragment %input, %center_offsets, %rect_center, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %empty_loaded = vc4kernel.tmu_load_fragment %input, %empty_offsets, %rect_empty_row, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %pair_loaded = vc4kernel.tmu_load_fragment %input, %pair_offsets, %rect_pair, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %last_loaded = vc4kernel.tmu_load_fragment %input, %last_offsets, %rect_last, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %store1 = vc4kernel.fragment_const {value = dense<[64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124]> : vector<16xi32>} : vector<16xi32>
    %store2 = vc4kernel.fragment_const {value = dense<[128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188]> : vector<16xi32>} : vector<16xi32>
    %store3 = vc4kernel.fragment_const {value = dense<[192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252]> : vector<16xi32>} : vector<16xi32>
    %store4 = vc4kernel.fragment_const {value = dense<[256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316]> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %real_offsets, %full_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store1, %center_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store2, %empty_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store3, %pair_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store4, %last_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
