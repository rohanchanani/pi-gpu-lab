module {
  vc4kernel.kernel @dynamic_vpm_coord_selector_forced_spill_vc4kernel(%in : i32, %out : i32, %row : i32, %word_x : i32, %sel16 : i32, %beta : i32) attributes {
    public_name = "dynamic_vpm_coord_selector_forced_spill_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "u8"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "u8"},
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "word_x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "sel16", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "beta", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %off256 = arith.constant 256 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %known = vc4kernel.fragment_const {value = dense<[6144, 6145, 6146, 6147, 6148, 6149, 6150, 6151, 6152, 6153, 6154, 6155, 6156, 6157, 6158, 6159]> : vector<16xi32>} : vector<16xi32>
    %beta_v = vc4kernel.splat %beta : i32 -> vector<16xi32>
    %p01 = vc4kernel.fragment_alu.add %beta_v, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p02 = vc4kernel.fragment_alu.add %p01, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p03 = vc4kernel.fragment_alu.add %p02, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p04 = vc4kernel.fragment_alu.add %p03, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p05 = vc4kernel.fragment_alu.add %p04, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p06 = vc4kernel.fragment_alu.add %p05, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p07 = vc4kernel.fragment_alu.add %p06, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p08 = vc4kernel.fragment_alu.add %p07, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p09 = vc4kernel.fragment_alu.add %p08, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p10 = vc4kernel.fragment_alu.add %p09, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p11 = vc4kernel.fragment_alu.add %p10, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p12 = vc4kernel.fragment_alu.add %p11, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p13 = vc4kernel.fragment_alu.add %p12, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p14 = vc4kernel.fragment_alu.add %p13, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p15 = vc4kernel.fragment_alu.add %p14, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p16 = vc4kernel.fragment_alu.add %p15, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p17 = vc4kernel.fragment_alu.add %p16, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p18 = vc4kernel.fragment_alu.add %p17, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p19 = vc4kernel.fragment_alu.add %p18, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p20 = vc4kernel.fragment_alu.add %p19, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p21 = vc4kernel.fragment_alu.add %p20, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p22 = vc4kernel.fragment_alu.add %p21, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p23 = vc4kernel.fragment_alu.add %p22, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p24 = vc4kernel.fragment_alu.add %p23, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 32 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    vc4kernel.vdr_load_to_vpm %in, %c0, %tile, %row dynamic_dst_x %word_x dynamic_subword_selector %sel16 {rows = 1 : i32, cols = 16 : i32, global_stride_bytes = 32 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32 dynamic_dst_x i32 dynamic_subword_selector i32
    vc4kernel.vdw_store_vpm_fragment %tile, %row dynamic_src_x %word_x dynamic_subword_selector %sel16, %out, %off256, %full {elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, !vc4kernel.pred<16>
    %compute_row = arith.addi %row, %c1 : i32
    %packed_known = vc4kernel.fragment_pack %known {dest = #vc4kernel.subword_type<u16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<truncate>} : vector<16xi32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %compute_row dynamic_subword_selector %sel16, %packed_known, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, vector<16xi32>, !vc4kernel.pred<16>
    %raw = vc4kernel.vpm_read_fragment %tile, %compute_row dynamic_subword_selector %sel16, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_subword_selector i32, !vc4kernel.pred<16> -> vector<16xi32>
    %values = vc4kernel.fragment_unpack %raw {source = #vc4kernel.subword_type<s16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<sign_extend>} : vector<16xi32> -> vector<16xi32>
    %s01 = vc4kernel.fragment_alu.add %values, %p01 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s02 = vc4kernel.fragment_alu.add %s01, %p02 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s03 = vc4kernel.fragment_alu.add %s02, %p03 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s04 = vc4kernel.fragment_alu.add %s03, %p04 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s05 = vc4kernel.fragment_alu.add %s04, %p05 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s06 = vc4kernel.fragment_alu.add %s05, %p06 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s07 = vc4kernel.fragment_alu.add %s06, %p07 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s08 = vc4kernel.fragment_alu.add %s07, %p08 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s09 = vc4kernel.fragment_alu.add %s08, %p09 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s10 = vc4kernel.fragment_alu.add %s09, %p10 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s11 = vc4kernel.fragment_alu.add %s10, %p11 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s12 = vc4kernel.fragment_alu.add %s11, %p12 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s13 = vc4kernel.fragment_alu.add %s12, %p13 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s14 = vc4kernel.fragment_alu.add %s13, %p14 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s15 = vc4kernel.fragment_alu.add %s14, %p15 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s16 = vc4kernel.fragment_alu.add %s15, %p16 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s17 = vc4kernel.fragment_alu.add %s16, %p17 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s18 = vc4kernel.fragment_alu.add %s17, %p18 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s19 = vc4kernel.fragment_alu.add %s18, %p19 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s20 = vc4kernel.fragment_alu.add %s19, %p20 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s21 = vc4kernel.fragment_alu.add %s20, %p21 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s22 = vc4kernel.fragment_alu.add %s21, %p22 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s23 = vc4kernel.fragment_alu.add %s22, %p23 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s24 = vc4kernel.fragment_alu.add %s23, %p24 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %s24, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
