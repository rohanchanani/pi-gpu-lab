module {
  vc4kernel.kernel @vdw_dynamic_selector_preserve_tail_rect_vc4kernel(%out : i32, %n : i32, %active_rows : i32, %active_cols : i32, %stride : i32, %row : i32, %word_x : i32, %sel8 : i32, %sel16 : i32) attributes {
    public_name = "vdw_dynamic_selector_preserve_tail_rect_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "u8"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "stride", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "row", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "word_x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "sel8", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "sel16", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c64 = arith.constant 64 : i32
    %off256 = arith.constant 256 : i32
    %tail = vc4kernel.pred.tail %c0, %n : i32, i32 -> !vc4kernel.pred<16>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %shift16 = vc4kernel.fragment_const {value = dense<16> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 12 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    %row_rect1 = arith.addi %row, %c1 : i32

    %b8_0 = vc4kernel.fragment_const {value = dense<40> : vector<16xi32>} : vector<16xi32>
    %b8_1 = vc4kernel.fragment_const {value = dense<80> : vector<16xi32>} : vector<16xi32>
    %v8_0 = vc4kernel.fragment_alu.add %b8_0, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_1 = vc4kernel.fragment_alu.add %b8_1, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8_1s = vc4kernel.fragment_alu.add %v8_1, %shift16 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %packed8 = vc4kernel.fragment_alu.add %v8_0, %v8_1s {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row, %packed8, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_vpm_fragment %tile, %row, %out, %c0, %tail {elem_bytes = 1 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w8>, subword = #vc4kernel.vpm_subword<packed>, src_x = 0 : i32, subword_selector = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>

    %b16r0_0 = vc4kernel.fragment_const {value = dense<2000> : vector<16xi32>} : vector<16xi32>
    %b16r0_1 = vc4kernel.fragment_const {value = dense<3000> : vector<16xi32>} : vector<16xi32>
    %v16r0_0 = vc4kernel.fragment_alu.add %b16r0_0, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v16r0_1 = vc4kernel.fragment_alu.add %b16r0_1, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v16r0_1s = vc4kernel.fragment_alu.add %v16r0_1, %shift16 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %packed16r0 = vc4kernel.fragment_alu.add %v16r0_0, %v16r0_1s {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row, %packed16r0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>

    %b16r1_0 = vc4kernel.fragment_const {value = dense<4000> : vector<16xi32>} : vector<16xi32>
    %b16r1_1 = vc4kernel.fragment_const {value = dense<5000> : vector<16xi32>} : vector<16xi32>
    %v16r1_0 = vc4kernel.fragment_alu.add %b16r1_0, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v16r1_1 = vc4kernel.fragment_alu.add %b16r1_1, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v16r1_1s = vc4kernel.fragment_alu.add %v16r1_1, %shift16 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %packed16r1 = vc4kernel.fragment_alu.add %v16r1_0, %v16r1_1s {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row_rect1, %packed16r1, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>

    vc4kernel.vdw_store_rect_from_vpm %tile, %row dynamic_src_x %word_x dynamic_subword_selector %sel16, %out, %off256, %active_rows, %active_cols, %stride {max_rows = 2 : i32, max_cols = 7 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32 dynamic_src_x i32 dynamic_subword_selector i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
