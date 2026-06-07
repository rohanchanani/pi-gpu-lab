module {
  vc4kernel.kernel @mixed_surface_lock_vpm_pipeline_full_vc4kernel(%in : i32, %out : i32, %iters : i32, %active_cols : i32) attributes {
    public_name = "mixed_surface_lock_vpm_pipeline_full_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "u8"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "u8"},
      {name = "iters", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %c160 = arith.constant 160 : i32
    %c640 = arith.constant 640 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %seg512 = vc4kernel.fragment_const {value = dense<512> : vector<16xi32>} : vector<16xi32>
    %seg576 = vc4kernel.fragment_const {value = dense<576> : vector<16xi32>} : vector<16xi32>
    %off512 = vc4kernel.fragment_alu.add %seg512, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %off576 = vc4kernel.fragment_alu.add %seg576, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %known = vc4kernel.fragment_const {value = dense<[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]> : vector<16xi32>} : vector<16xi32>
    %n_v = vc4kernel.splat %active_cols : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 63 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    cf.br ^loop(%c0 : i32)

  ^loop(%j : i32):
    %more = arith.cmpi ult, %j, %iters : i32
    cf.cond_br %more, ^body(%j : i32), ^done

  ^body(%j_body : i32):
    %ping = arith.andi %j_body, %c1 : i32
    %row = arith.shli %ping, %c2 : i32
    %byte_off = arith.shli %j_body, %c5 : i32
    vc4kernel.vdr_load_rect_to_vpm %in, %byte_off, %tile, %row dynamic_dst_x %ping dynamic_subword_selector %ping, %c1, %active_cols, %c32 {max_rows = 1 : i32, max_cols = 7 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32, i32, i32, i32
    vc4kernel.vdw_store_rect_from_vpm %tile, %row dynamic_src_x %ping dynamic_subword_selector %ping, %out, %byte_off, %c1, %active_cols, %c32 {max_rows = 1 : i32, max_cols = 7 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32
    %next = arith.addi %j_body, %c1 : i32
    cf.br ^loop(%next : i32)

  ^done:
    %last_ping = arith.andi %iters, %c1 : i32
    %qpu_row = arith.shli %last_ping, %c4 : i32
    %word_x = arith.andi %active_cols, %c1 : i32
    %sel8_bounded = arith.andi %iters, %c3 : i32
    %packed = vc4kernel.fragment_pack %known {dest = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %qpu_row dynamic_x %word_x dynamic_subword_selector %sel8_bounded, %packed, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %raw = vc4kernel.vpm_read_fragment %tile, %qpu_row dynamic_x %word_x dynamic_subword_selector %sel8_bounded, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %u8 = vc4kernel.fragment_unpack %raw {source = #vc4kernel.subword_type<u8>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<zero_extend>} : vector<16xi32> -> vector<16xi32>
    %biased = vc4kernel.fragment_alu.add %u8, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %active = vc4kernel.fragment_cmp %lanes, %n_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %selected = vc4kernel.fragment_select %active, %biased, %zero : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sum = vc4kernel.fragment_reduce %selected, %full {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %rot = vc4kernel.fragment_rotate %selected, %iters : vector<16xi32>, i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off512, %rot, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off576, %sum, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdr_load_rect_to_vpm %in, %c160, %tile, %c32 dynamic_dst_x %word_x, %c1, %c1, %c64 {max_rows = 1 : i32, max_cols = 1 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32, i32, i32, i32
    vc4kernel.vdw_store_rect_from_vpm %tile, %c32 dynamic_src_x %word_x, %out, %c640, %c1, %c1, %c64 {max_rows = 1 : i32, max_cols = 1 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
