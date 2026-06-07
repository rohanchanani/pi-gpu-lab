module {
  vc4kernel.kernel @vdr_dynamic_selector_loop_branch_vc4kernel(%in : i32, %out : i32, %row_h : i32, %row_v : i32, %control : i32) attributes {
    public_name = "vdr_dynamic_selector_loop_branch_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "u8"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "row_h", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "row_v", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "control", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c8 = arith.constant 8 : i32
    %off64 = arith.constant 64 : i32
    %off256 = arith.constant 256 : i32
    %off320 = arith.constant 320 : i32
    %row_h8 = arith.addi %row_h, %c1 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %seg1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %seg2 = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %out1 = vc4kernel.fragment_alu.add %seg1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %out2 = vc4kernel.fragment_alu.add %seg2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 63 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    cf.br ^loop(%c0, %c0 : i32, i32)

  ^loop(%j : i32, %acc : i32):
    %more = arith.cmpi ult, %j, %c2 : i32
    cf.cond_br %more, ^step(%j : i32), ^after_loop(%acc : i32)

  ^step(%j_step : i32):
    %next = arith.addi %j_step, %c1 : i32
    cf.br ^loop(%next, %next : i32, i32)

  ^after_loop(%loop_x : i32):
    %sent_word = vc4kernel.fragment_const {value = dense<1515870810> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_h, %sent_word, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %row_h dynamic_dst_x %control {rows = 1 : i32, cols = 1 : i32, global_stride_bytes = 8 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32
    %r_h32 = vc4kernel.vpm_read_fragment %tile, %row_h, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdr_load_to_vpm %in, %off64, %tile, %row_v dynamic_dst_x %control {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32
    %r_v32 = vc4kernel.vpm_read_fragment %tile, %row_v dynamic_x %control, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, !vc4kernel.pred<16> -> vector<16xi32>
    %use_alt = arith.cmpi ne, %control, %c0 : i32
    cf.cond_br %use_alt, ^alt(%loop_x : i32), ^main(%loop_x : i32)

  ^main(%x_main : i32):
    %sent_byte = vc4kernel.fragment_const {value = dense<90> : vector<16xi32>} : vector<16xi32>
    %pack_byte = vc4kernel.fragment_pack %sent_byte {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_h8 dynamic_subword_selector %control, %pack_byte, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdr_load_to_vpm %in, %off256, %tile, %row_h8 dynamic_dst_x %x_main dynamic_subword_selector %control {rows = 1 : i32, cols = 1 : i32, global_stride_bytes = 8 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32
    %raw_h8 = vc4kernel.vpm_read_fragment %tile, %row_h8 dynamic_subword_selector %control, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %out_h8 = vc4kernel.fragment_unpack %raw_h8 {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>
    cf.br ^store(%out_h8 : vector<16xi32>)

  ^alt(%x_seed : i32):
    %x_alt = arith.addi %x_seed, %control : i32
    %sent_half = vc4kernel.fragment_const {value = dense<21930> : vector<16xi32>} : vector<16xi32>
    %pack_half = vc4kernel.fragment_pack %sent_half {dest = #vc4kernel.subword_type<u16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_v dynamic_x %x_alt dynamic_subword_selector %control, %pack_half, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdr_load_to_vpm %in, %off320, %tile, %row_v dynamic_dst_x %x_alt dynamic_subword_selector %control {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 32 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32
    %raw_v16 = vc4kernel.vpm_read_fragment %tile, %row_v dynamic_x %x_alt dynamic_subword_selector %control, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %out_v16 = vc4kernel.fragment_unpack %raw_v16 {source = #vc4kernel.subword_type<s16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<sign_extend>} : vector<16xi32> -> vector<16xi32>
    cf.br ^store(%out_v16 : vector<16xi32>)

  ^store(%branch_out : vector<16xi32>):
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %r_h32, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %out1, %r_v32, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %out2, %branch_out, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
