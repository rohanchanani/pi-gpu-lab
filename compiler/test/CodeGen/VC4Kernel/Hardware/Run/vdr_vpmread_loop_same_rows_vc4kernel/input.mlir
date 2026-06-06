module {
  vc4kernel.kernel @vdr_vpmread_loop_same_rows_vc4kernel(%in : i32, %out : i32, %active_cols : i32, %pitch : i32) attributes {
    public_name = "vdr_vpmread_loop_same_rows_vc4kernel",
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
    // Same-row VDR->VPM reuse is accepted only when each QPU VPM read is
    // followed by a read-side VPM/VCD wait and scheduled backedges target the
    // loop header after dynamic-DMA slot accounting.
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c7 = arith.constant 7 : i32
    %c16 = arith.constant 16 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 2 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    cf.br ^loop(%c0, %zero : i32, vector<16xi32>)

  ^loop(%k : i32, %acc : vector<16xi32>):
    %more = arith.cmpi ult, %k, %c2 : i32
    cf.cond_br %more, ^step(%k, %acc : i32, vector<16xi32>), ^store(%acc : vector<16xi32>)

  ^step(%k_step : i32, %acc_step : vector<16xi32>):
    %byte_offset = arith.shli %k_step, %c7 : i32
    vc4kernel.vdr_load_rect_to_vpm %in, %byte_offset, %tile, %c0, %c2, %c16, %pitch {max_rows = 2 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    %tail = vc4kernel.pred.tail %c0, %active_cols : i32, i32 -> !vc4kernel.pred<16>
    %r0 = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r1 = vc4kernel.vpm_read_fragment %tile, %c1, %tail {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %sum0 = vc4kernel.fragment_alu.add %acc_step, %r0 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sum1 = vc4kernel.fragment_alu.add %sum0, %r1 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %k_next = arith.addi %k_step, %c1 : i32
    cf.br ^loop(%k_next, %sum1 : i32, vector<16xi32>)

  ^store(%sum : vector<16xi32>):
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %sum, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
