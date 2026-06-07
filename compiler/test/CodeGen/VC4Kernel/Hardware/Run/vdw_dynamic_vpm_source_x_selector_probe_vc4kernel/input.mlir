module {
  vc4kernel.kernel @vdw_dynamic_vpm_source_x_selector_probe_vc4kernel(%out : i32, %row_h : i32, %row_v32 : i32, %row_v8 : i32, %row_v16 : i32, %word_x : i32, %sel8 : i32, %sel16 : i32) attributes {
    public_name = "vdw_dynamic_vpm_source_x_selector_probe_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "u8"},
      {name = "row_h", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "row_v32", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "row_v8", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "row_v16", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "word_x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "sel8", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "sel16", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %c5 = arith.constant 5 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %off64 = arith.constant 64 : i32
    %off128 = arith.constant 128 : i32
    %off192 = arith.constant 192 : i32
    %off256 = arith.constant 256 : i32
    %off320 = arith.constant 320 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %shift8 = vc4kernel.fragment_const {value = dense<8> : vector<16xi32>} : vector<16xi32>
    %shift16 = vc4kernel.fragment_const {value = dense<16> : vector<16xi32>} : vector<16xi32>
    %shift24 = vc4kernel.fragment_const {value = dense<24> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 63 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    %row_h8 = arith.addi %row_h, %c3 : i32
    %row_h16 = arith.addi %row_h, %c5 : i32

    %base_h32 = vc4kernel.fragment_const {value = dense<16640> : vector<16xi32>} : vector<16xi32>
    %val_h32 = vc4kernel.fragment_alu.add %base_h32, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_h, %val_h32, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_rect_from_vpm %tile, %row_h dynamic_src_x %word_x, %out, %c0, %c1, %c1, %c64 {max_rows = 1 : i32, max_cols = 1 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32, i32, i32, i32, i32, i32

    %base_v32 = vc4kernel.fragment_const {value = dense<16896> : vector<16xi32>} : vector<16xi32>
    %val_v32 = vc4kernel.fragment_alu.add %base_v32, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_v32 dynamic_x %word_x, %val_v32, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_rect_from_vpm %tile, %row_v32 dynamic_src_x %word_x, %out, %off64, %c1, %c16, %c64 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32, i32, i32, i32, i32, i32

    %base_h8 = vc4kernel.fragment_const {value = dense<49> : vector<16xi32>} : vector<16xi32>
    %base_h8_b1 = vc4kernel.fragment_const {value = dense<81> : vector<16xi32>} : vector<16xi32>
    %base_h8_b2 = vc4kernel.fragment_const {value = dense<113> : vector<16xi32>} : vector<16xi32>
    %base_h8_b3 = vc4kernel.fragment_const {value = dense<145> : vector<16xi32>} : vector<16xi32>
    %h8_b0 = vc4kernel.fragment_alu.add %base_h8, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h8_b1v = vc4kernel.fragment_alu.add %base_h8_b1, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h8_b2v = vc4kernel.fragment_alu.add %base_h8_b2, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h8_b3v = vc4kernel.fragment_alu.add %base_h8_b3, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h8_b1s = vc4kernel.fragment_alu.add %h8_b1v, %shift8 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h8_b2s = vc4kernel.fragment_alu.add %h8_b2v, %shift16 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h8_b3s = vc4kernel.fragment_alu.add %h8_b3v, %shift24 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h8_or01 = vc4kernel.fragment_alu.add %h8_b0, %h8_b1s {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h8_or23 = vc4kernel.fragment_alu.add %h8_b2s, %h8_b3s {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h8_words = vc4kernel.fragment_alu.add %h8_or01, %h8_or23 {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_h8, %h8_words, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_rect_from_vpm %tile, %row_h8 dynamic_src_x %word_x dynamic_subword_selector %sel8, %out, %off128, %c1, %c1, %c16 {max_rows = 1 : i32, max_cols = 1 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32

    %base_h16 = vc4kernel.fragment_const {value = dense<4608> : vector<16xi32>} : vector<16xi32>
    %base_h16_h1 = vc4kernel.fragment_const {value = dense<8704> : vector<16xi32>} : vector<16xi32>
    %h16_h0 = vc4kernel.fragment_alu.add %base_h16, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h16_h1v = vc4kernel.fragment_alu.add %base_h16_h1, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h16_h1s = vc4kernel.fragment_alu.add %h16_h1v, %shift16 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %h16_words = vc4kernel.fragment_alu.add %h16_h0, %h16_h1s {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_h16, %h16_words, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_rect_from_vpm %tile, %row_h16 dynamic_src_x %word_x dynamic_subword_selector %sel16, %out, %off192, %c1, %c1, %c16 {max_rows = 1 : i32, max_cols = 1 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32

    %base_v8 = vc4kernel.fragment_const {value = dense<81> : vector<16xi32>} : vector<16xi32>
    %base_v8_b1 = vc4kernel.fragment_const {value = dense<113> : vector<16xi32>} : vector<16xi32>
    %base_v8_b2 = vc4kernel.fragment_const {value = dense<145> : vector<16xi32>} : vector<16xi32>
    %base_v8_b3 = vc4kernel.fragment_const {value = dense<177> : vector<16xi32>} : vector<16xi32>
    %v8_b0 = vc4kernel.fragment_alu.add %base_v8, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_b1v = vc4kernel.fragment_alu.add %base_v8_b1, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_b2v = vc4kernel.fragment_alu.add %base_v8_b2, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_b3v = vc4kernel.fragment_alu.add %base_v8_b3, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_b1s = vc4kernel.fragment_alu.add %v8_b1v, %shift8 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_b2s = vc4kernel.fragment_alu.add %v8_b2v, %shift16 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_b3s = vc4kernel.fragment_alu.add %v8_b3v, %shift24 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_or01 = vc4kernel.fragment_alu.add %v8_b0, %v8_b1s {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_or23 = vc4kernel.fragment_alu.add %v8_b2s, %v8_b3s {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_words = vc4kernel.fragment_alu.add %v8_or01, %v8_or23 {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_v8 dynamic_x %word_x, %v8_words, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_rect_from_vpm %tile, %row_v8 dynamic_src_x %word_x dynamic_subword_selector %sel8, %out, %off256, %c1, %c16, %c16 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32

    %base_v16 = vc4kernel.fragment_const {value = dense<5120> : vector<16xi32>} : vector<16xi32>
    %base_v16_h1 = vc4kernel.fragment_const {value = dense<9216> : vector<16xi32>} : vector<16xi32>
    %v16_h0 = vc4kernel.fragment_alu.add %base_v16, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v16_h1v = vc4kernel.fragment_alu.add %base_v16_h1, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v16_h1s = vc4kernel.fragment_alu.add %v16_h1v, %shift16 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v16_words = vc4kernel.fragment_alu.add %v16_h0, %v16_h1s {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_v16 dynamic_x %word_x, %v16_words, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32 dynamic_x i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_rect_from_vpm %tile, %row_v16 dynamic_src_x %word_x dynamic_subword_selector %sel16, %out, %off320, %c1, %c16, %c64 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
