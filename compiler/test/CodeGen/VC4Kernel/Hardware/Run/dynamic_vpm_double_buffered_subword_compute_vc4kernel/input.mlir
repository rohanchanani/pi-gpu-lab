module {
  vc4kernel.kernel @dynamic_vpm_double_buffered_subword_compute_vc4kernel(%in : i32, %out : i32, %n : i32, %row : i32, %word_x : i32, %sel8 : i32, %amount : i32) attributes {
    public_name = "dynamic_vpm_double_buffered_subword_compute_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "u8"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "word_x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "sel8", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "amount", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %off256 = arith.constant 256 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %reduce_bytes = vc4kernel.fragment_const {value = dense<[128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188]> : vector<16xi32>} : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %known = vc4kernel.fragment_const {value = dense<[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]> : vector<16xi32>} : vector<16xi32>
    %n_v = vc4kernel.splat %n : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 32 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %row dynamic_dst_x %word_x dynamic_subword_selector %sel8 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 16 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32
    vc4kernel.vdw_store_vpm_fragment %tile, %row dynamic_src_x %word_x dynamic_subword_selector %sel8, %out, %off256, %full {elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, !vc4kernel.pred<16>
    %compute_row = arith.addi %row, %c1 : i32
    %packed = vc4kernel.fragment_pack %known {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %compute_row dynamic_subword_selector %sel8, %packed, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %raw = vc4kernel.vpm_read_fragment %tile, %compute_row dynamic_subword_selector %sel8, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %u8 = vc4kernel.fragment_unpack %raw {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>
    %biased = vc4kernel.fragment_alu.add %u8, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %active = vc4kernel.fragment_cmp %lanes, %n_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %selected = vc4kernel.fragment_select %active, %biased, %zero : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sum = vc4kernel.fragment_reduce %selected, %full {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %rot = vc4kernel.fragment_rotate %selected, %amount : vector<16xi32>, i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %rot, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %reduce_bytes, %sum, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
