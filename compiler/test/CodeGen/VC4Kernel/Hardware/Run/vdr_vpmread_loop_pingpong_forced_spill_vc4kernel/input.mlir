module {
  vc4kernel.kernel @vdr_vpmread_loop_pingpong_forced_spill_vc4kernel(%in : i32, %out : i32, %active_cols : i32, %pitch : i32) attributes {
    public_name = "vdr_vpmread_loop_pingpong_forced_spill_vc4kernel",
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
    // This is the same ping-pong primitive as vdr_vpmread_loop_pingpong_rows_vc4kernel,
    // with additional live vector pressure across the loop to force a spill frame.
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c7 = arith.constant 7 : i32
    %c16 = arith.constant 16 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %v1 = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %v2 = vc4kernel.fragment_const {value = dense<2> : vector<16xi32>} : vector<16xi32>
    %v3 = vc4kernel.fragment_const {value = dense<3> : vector<16xi32>} : vector<16xi32>
    %v4 = vc4kernel.fragment_const {value = dense<4> : vector<16xi32>} : vector<16xi32>
    %v5 = vc4kernel.fragment_const {value = dense<5> : vector<16xi32>} : vector<16xi32>
    %v6 = vc4kernel.fragment_const {value = dense<6> : vector<16xi32>} : vector<16xi32>
    %v7 = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %v8 = vc4kernel.fragment_const {value = dense<8> : vector<16xi32>} : vector<16xi32>
    %v9 = vc4kernel.fragment_const {value = dense<9> : vector<16xi32>} : vector<16xi32>
    %v10 = vc4kernel.fragment_const {value = dense<10> : vector<16xi32>} : vector<16xi32>
    %v11 = vc4kernel.fragment_const {value = dense<11> : vector<16xi32>} : vector<16xi32>
    %v12 = vc4kernel.fragment_const {value = dense<12> : vector<16xi32>} : vector<16xi32>
    %v13 = vc4kernel.fragment_const {value = dense<13> : vector<16xi32>} : vector<16xi32>
    %v14 = vc4kernel.fragment_const {value = dense<14> : vector<16xi32>} : vector<16xi32>
    %v15 = vc4kernel.fragment_const {value = dense<15> : vector<16xi32>} : vector<16xi32>
    %v16 = vc4kernel.fragment_const {value = dense<16> : vector<16xi32>} : vector<16xi32>
    %v17 = vc4kernel.fragment_const {value = dense<17> : vector<16xi32>} : vector<16xi32>
    %v18 = vc4kernel.fragment_const {value = dense<18> : vector<16xi32>} : vector<16xi32>
    %v19 = vc4kernel.fragment_const {value = dense<19> : vector<16xi32>} : vector<16xi32>
    %v20 = vc4kernel.fragment_const {value = dense<20> : vector<16xi32>} : vector<16xi32>
    %v21 = vc4kernel.fragment_const {value = dense<21> : vector<16xi32>} : vector<16xi32>
    %v22 = vc4kernel.fragment_const {value = dense<22> : vector<16xi32>} : vector<16xi32>
    %v23 = vc4kernel.fragment_const {value = dense<23> : vector<16xi32>} : vector<16xi32>
    %v24 = vc4kernel.fragment_const {value = dense<24> : vector<16xi32>} : vector<16xi32>
    %v25 = vc4kernel.fragment_const {value = dense<25> : vector<16xi32>} : vector<16xi32>
    %v26 = vc4kernel.fragment_const {value = dense<26> : vector<16xi32>} : vector<16xi32>
    %v27 = vc4kernel.fragment_const {value = dense<27> : vector<16xi32>} : vector<16xi32>
    %v28 = vc4kernel.fragment_const {value = dense<28> : vector<16xi32>} : vector<16xi32>
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
    %a1 = vc4kernel.fragment_alu.add %sum, %v1 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a2 = vc4kernel.fragment_alu.add %a1, %v2 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a3 = vc4kernel.fragment_alu.add %a2, %v3 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a4 = vc4kernel.fragment_alu.add %a3, %v4 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a5 = vc4kernel.fragment_alu.add %a4, %v5 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a6 = vc4kernel.fragment_alu.add %a5, %v6 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a7 = vc4kernel.fragment_alu.add %a6, %v7 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a8 = vc4kernel.fragment_alu.add %a7, %v8 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a9 = vc4kernel.fragment_alu.add %a8, %v9 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a10 = vc4kernel.fragment_alu.add %a9, %v10 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a11 = vc4kernel.fragment_alu.add %a10, %v11 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a12 = vc4kernel.fragment_alu.add %a11, %v12 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a13 = vc4kernel.fragment_alu.add %a12, %v13 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a14 = vc4kernel.fragment_alu.add %a13, %v14 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a15 = vc4kernel.fragment_alu.add %a14, %v15 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a16 = vc4kernel.fragment_alu.add %a15, %v16 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a17 = vc4kernel.fragment_alu.add %a16, %v17 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a18 = vc4kernel.fragment_alu.add %a17, %v18 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a19 = vc4kernel.fragment_alu.add %a18, %v19 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a20 = vc4kernel.fragment_alu.add %a19, %v20 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a21 = vc4kernel.fragment_alu.add %a20, %v21 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a22 = vc4kernel.fragment_alu.add %a21, %v22 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a23 = vc4kernel.fragment_alu.add %a22, %v23 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a24 = vc4kernel.fragment_alu.add %a23, %v24 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a25 = vc4kernel.fragment_alu.add %a24, %v25 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a26 = vc4kernel.fragment_alu.add %a25, %v26 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a27 = vc4kernel.fragment_alu.add %a26, %v27 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %final = vc4kernel.fragment_alu.add %a27, %v28 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %final, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
