module {
  vc4kernel.kernel @mixed_f16_storage_conversion_vc4kernel(%in : i32, %out : i32, %n : i32, %row : i32, %word_x : i32, %sel16 : i32) attributes {
    public_name = "mixed_f16_storage_conversion_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "u16"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "u16"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "word_x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "sel16", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %active_cols = arith.minui %n, %c16 : i32
    %compute_sel = arith.andi %sel16, %c0 : i32
    %out_row = arith.addi %row, %c1 : i32
    %guard_row = arith.addi %row, %c2 : i32
    %scale = vc4kernel.fragment_const {value = dense<1.000000e+00> : vector<16xf32>} : vector<16xf32>
    %bias = vc4kernel.fragment_const {value = dense<1.000000e+00> : vector<16xf32>} : vector<16xf32>
    %tile = vc4kernel.vpm_alloc {rows = 48 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %row dynamic_dst_x %c0 dynamic_subword_selector %compute_sel {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 32 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32
    %raw = vc4kernel.vpm_read_fragment %tile, %row dynamic_subword_selector %compute_sel, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %selector_shift = arith.shli %compute_sel, %c4 : i32
    %selector_shift_v = vc4kernel.splat %selector_shift : i32 -> vector<16xi32>
    %raw_16a = vc4kernel.fragment_alu.add %raw, %selector_shift_v {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %x = vc4kernel.fragment_unpack %raw_16a {source = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<to_f32>} : vector<16xi32> -> vector<16xf32>
    %scaled = vc4kernel.fragment_alu.mul %x, %scale {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %y = vc4kernel.fragment_alu.add %scaled, %bias {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %packed = vc4kernel.fragment_pack %y {dest = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<from_f32>} : vector<16xf32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %out_row dynamic_subword_selector %compute_sel, %packed, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_rect_from_vpm %tile, %out_row dynamic_src_x %c0 dynamic_subword_selector %compute_sel, %out, %c0, %c1, %active_cols, %c32 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32

    vc4kernel.vdr_load_to_vpm %in, %c32, %tile, %guard_row dynamic_dst_x %word_x dynamic_subword_selector %sel16 {rows = 1 : i32, cols = 1 : i32, global_stride_bytes = 32 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32
    vc4kernel.vdw_store_rect_from_vpm %tile, %guard_row dynamic_src_x %word_x dynamic_subword_selector %sel16, %out, %c64, %c1, %c1, %c32 {max_rows = 1 : i32, max_cols = 1 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
