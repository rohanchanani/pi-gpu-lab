module {
  vc4kernel.kernel @vdw_subword_preserve_tail_rect_vc4kernel(%out : i32, %n : i32, %active_rows : i32, %active_cols : i32, %stride : i32) attributes {
    public_name = "vdw_subword_preserve_tail_rect_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "u8"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "stride", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %off128 = arith.constant 128 : i32
    %tail = vc4kernel.pred.tail %c0, %n : i32, i32 -> !vc4kernel.pred<16>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    %b8 = vc4kernel.fragment_const {value = dense<40> : vector<16xi32>} : vector<16xi32>
    %v8 = vc4kernel.fragment_alu.add %b8, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p8 = vc4kernel.fragment_pack %v8 {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %c0, %p8, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, subword_selector = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_vpm_fragment %tile, %c0, %out, %c0, %tail {elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, src_x = 0 : i32, subword_selector = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>

    %b16r0 = vc4kernel.fragment_const {value = dense<2000> : vector<16xi32>} : vector<16xi32>
    %v16r0 = vc4kernel.fragment_alu.add %b16r0, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p16r0 = vc4kernel.fragment_pack %v16r0 {dest = #vc4kernel.subword_type<u16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %c1, %p16r0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, subword_selector = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %b16r1 = vc4kernel.fragment_const {value = dense<2200> : vector<16xi32>} : vector<16xi32>
    %v16r1 = vc4kernel.fragment_alu.add %b16r1, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p16r1 = vc4kernel.fragment_pack %v16r1 {dest = #vc4kernel.subword_type<u16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %c2, %p16r1, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, subword_selector = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_rect_from_vpm %tile, %c1, %out, %off128, %active_rows, %active_cols, %stride {max_rows = 2 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, src_x = 0 : i32, subword_selector = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
