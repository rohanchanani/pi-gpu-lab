module {
  vc4kernel.kernel @vpm_qpu_dynamic_selector_tail_preserve_vc4kernel(%out : i32, %active_cols : i32, %row_h : i32, %row_v : i32, %word_x : i32, %sel8 : i32, %sel16 : i32) attributes {
    public_name = "vpm_qpu_dynamic_selector_tail_preserve_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "row_h", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "row_v", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "word_x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "sel8", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "sel16", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tail = vc4kernel.pred.tail %c0, %active_cols : i32, i32 -> !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %seg1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %off1 = vc4kernel.fragment_alu.add %seg1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 32 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    %base_h8 = vc4kernel.fragment_const {value = dense<90> : vector<16xi32>} : vector<16xi32>
    %value_h8 = vc4kernel.fragment_alu.add %base_h8, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %pack_h8 = vc4kernel.fragment_pack %value_h8 {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_h dynamic_subword_selector %sel8, %pack_h8, %tail {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %raw_h8 = vc4kernel.vpm_read_fragment %tile, %row_h dynamic_subword_selector %sel8, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %out_h8 = vc4kernel.fragment_unpack %raw_h8 {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>

    %base_v16 = vc4kernel.fragment_const {value = dense<1200> : vector<16xi32>} : vector<16xi32>
    %value_v16 = vc4kernel.fragment_alu.add %base_v16, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %pack_v16 = vc4kernel.fragment_pack %value_v16 {dest = #vc4kernel.subword_type<u16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_v dynamic_x %word_x dynamic_subword_selector %sel16, %pack_v16, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<laned>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %raw_v16 = vc4kernel.vpm_read_fragment %tile, %row_v dynamic_x %word_x dynamic_subword_selector %sel16, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<laned>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %out_v16 = vc4kernel.fragment_unpack %raw_v16 {source = #vc4kernel.subword_type<s16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<sign_extend>} : vector<16xi32> -> vector<16xi32>

    vc4kernel.vdw_store_fragment %out, %lane_bytes, %out_h8, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off1, %out_v16, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
