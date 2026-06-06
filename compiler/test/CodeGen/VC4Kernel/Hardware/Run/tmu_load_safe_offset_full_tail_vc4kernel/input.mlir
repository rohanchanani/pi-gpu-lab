module {
  vc4kernel.kernel @tmu_load_safe_offset_full_tail_vc4kernel(%input : i32, %out : i32) attributes {
    public_name = "tmu_load_safe_offset_full_tail_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c15 = arith.constant 15 : i32
    %c16 = arith.constant 16 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tail0 = vc4kernel.pred.tail %c0, %c0 : i32, i32 -> !vc4kernel.pred<16>
    %tail1 = vc4kernel.pred.tail %c0, %c1 : i32, i32 -> !vc4kernel.pred<16>
    %tail15 = vc4kernel.pred.tail %c0, %c15 : i32, i32 -> !vc4kernel.pred<16>
    %tail16 = vc4kernel.pred.tail %c0, %c16 : i32, i32 -> !vc4kernel.pred<16>
    %real_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %poison_offsets = vc4kernel.fragment_const {value = dense<[268435456, 268435460, 268435464, 268435468, 268435472, 268435476, 268435480, 268435484, 268435488, 268435492, 268435496, 268435500, 268435504, 268435508, 268435512, 268435516]> : vector<16xi32>} : vector<16xi32>
    %tail0_offsets = vc4kernel.fragment_select %tail0, %real_offsets, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %tail1_offsets = vc4kernel.fragment_select %tail1, %real_offsets, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %tail15_offsets = vc4kernel.fragment_select %tail15, %real_offsets, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %safe = arith.constant 0 : i32
    %full_loaded = vc4kernel.tmu_load_fragment %input, %real_offsets, %full, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %tail0_loaded = vc4kernel.tmu_load_fragment %input, %tail0_offsets, %tail0, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %tail1_loaded = vc4kernel.tmu_load_fragment %input, %tail1_offsets, %tail1, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %tail15_loaded = vc4kernel.tmu_load_fragment %input, %tail15_offsets, %tail15, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %tail16_loaded = vc4kernel.tmu_load_fragment %input, %real_offsets, %tail16, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %store1 = vc4kernel.fragment_const {value = dense<[64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124]> : vector<16xi32>} : vector<16xi32>
    %store2 = vc4kernel.fragment_const {value = dense<[128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188]> : vector<16xi32>} : vector<16xi32>
    %store3 = vc4kernel.fragment_const {value = dense<[192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252]> : vector<16xi32>} : vector<16xi32>
    %store4 = vc4kernel.fragment_const {value = dense<[256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 304, 308, 312, 316]> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %real_offsets, %full_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store1, %tail0_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store2, %tail1_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store3, %tail15_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store4, %tail16_loaded, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
