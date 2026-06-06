module {
  vc4kernel.kernel @vdr_vpmread_loop_pingpong_rows_vc4kernel(%in : i32, %out : i32, %active_cols : i32, %pitch : i32) attributes {
    public_name = "vdr_vpmread_loop_pingpong_rows_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "in", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "pitch", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    // SAME_ROW_REUSE_POLICY: SAME_ROW_REUSE_SAFE_WITH_REQUIRED_WAIT.
    // Ping-pong rows remain a proven blocked VDR->VPM + QPU VPM-read primitive:
    // iteration 0 loads rows 0-1, iteration 1 loads rows 2-3.
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c7 = arith.constant 7 : i32
    %c16 = arith.constant 16 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    cf.br ^loop(%c0, %c0, %zero : i32, i32, vector<16xi32>)

  ^loop(%k : i32, %row : i32, %acc : vector<16xi32>):
    %more = arith.cmpi ult, %k, %c2 : i32
    cf.cond_br %more, ^step(%k, %row, %acc : i32, i32, vector<16xi32>), ^store(%acc : vector<16xi32>)

  ^step(%k_step : i32, %row_step : i32, %acc_step : vector<16xi32>):
    %byte_offset = arith.shli %k_step, %c7 : i32
    vc4kernel.vdr_load_rect_to_vpm %in, %byte_offset, %tile, %row_step, %c2, %c16, %pitch {max_rows = 2 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    %row_tail = arith.addi %row_step, %c1 : i32
    %tail = vc4kernel.pred.tail %c0, %active_cols : i32, i32 -> !vc4kernel.pred<16>
    %r0 = vc4kernel.vpm_read_fragment %tile, %row_step, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r1 = vc4kernel.vpm_read_fragment %tile, %row_tail, %tail {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %sum0 = vc4kernel.fragment_alu.add %acc_step, %r0 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sum1 = vc4kernel.fragment_alu.add %sum0, %r1 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %k_next = arith.addi %k_step, %c1 : i32
    %row_next = arith.addi %row_step, %c2 : i32
    cf.br ^loop(%k_next, %row_next, %sum1 : i32, i32, vector<16xi32>)

  ^store(%sum : vector<16xi32>):
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %sum, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
