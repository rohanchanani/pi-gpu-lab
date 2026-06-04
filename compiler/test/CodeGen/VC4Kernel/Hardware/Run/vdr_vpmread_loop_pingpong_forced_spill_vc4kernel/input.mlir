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
    %k1 = arith.constant 1 : i32
    %k2 = arith.constant 2 : i32
    %k3 = arith.constant 3 : i32
    %k4 = arith.constant 4 : i32
    %k5 = arith.constant 5 : i32
    %k6 = arith.constant 6 : i32
    %k7 = arith.constant 7 : i32
    %k8 = arith.constant 8 : i32
    %k9 = arith.constant 9 : i32
    %k10 = arith.constant 10 : i32
    %k11 = arith.constant 11 : i32
    %k12 = arith.constant 12 : i32
    %k13 = arith.constant 13 : i32
    %k14 = arith.constant 14 : i32
    %k15 = arith.constant 15 : i32
    %k16 = arith.constant 16 : i32
    %k17 = arith.constant 17 : i32
    %k18 = arith.constant 18 : i32
    %k19 = arith.constant 19 : i32
    %k20 = arith.constant 20 : i32
    %k21 = arith.constant 21 : i32
    %k22 = arith.constant 22 : i32
    %k23 = arith.constant 23 : i32
    %k24 = arith.constant 24 : i32
    %k25 = arith.constant 25 : i32
    %k26 = arith.constant 26 : i32
    %k27 = arith.constant 27 : i32
    %k28 = arith.constant 28 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %zero_scalar = arith.constant 0 : i32
    %zero = vc4kernel.splat %zero_scalar : i32 -> vector<16xi32>
    %v1 = vc4kernel.splat %k1 : i32 -> vector<16xi32>
    %v2 = vc4kernel.splat %k2 : i32 -> vector<16xi32>
    %v3 = vc4kernel.splat %k3 : i32 -> vector<16xi32>
    %v4 = vc4kernel.splat %k4 : i32 -> vector<16xi32>
    %v5 = vc4kernel.splat %k5 : i32 -> vector<16xi32>
    %v6 = vc4kernel.splat %k6 : i32 -> vector<16xi32>
    %v7 = vc4kernel.splat %k7 : i32 -> vector<16xi32>
    %v8 = vc4kernel.splat %k8 : i32 -> vector<16xi32>
    %v9 = vc4kernel.splat %k9 : i32 -> vector<16xi32>
    %v10 = vc4kernel.splat %k10 : i32 -> vector<16xi32>
    %v11 = vc4kernel.splat %k11 : i32 -> vector<16xi32>
    %v12 = vc4kernel.splat %k12 : i32 -> vector<16xi32>
    %v13 = vc4kernel.splat %k13 : i32 -> vector<16xi32>
    %v14 = vc4kernel.splat %k14 : i32 -> vector<16xi32>
    %v15 = vc4kernel.splat %k15 : i32 -> vector<16xi32>
    %v16 = vc4kernel.splat %k16 : i32 -> vector<16xi32>
    %v17 = vc4kernel.splat %k17 : i32 -> vector<16xi32>
    %v18 = vc4kernel.splat %k18 : i32 -> vector<16xi32>
    %v19 = vc4kernel.splat %k19 : i32 -> vector<16xi32>
    %v20 = vc4kernel.splat %k20 : i32 -> vector<16xi32>
    %v21 = vc4kernel.splat %k21 : i32 -> vector<16xi32>
    %v22 = vc4kernel.splat %k22 : i32 -> vector<16xi32>
    %v23 = vc4kernel.splat %k23 : i32 -> vector<16xi32>
    %v24 = vc4kernel.splat %k24 : i32 -> vector<16xi32>
    %v25 = vc4kernel.splat %k25 : i32 -> vector<16xi32>
    %v26 = vc4kernel.splat %k26 : i32 -> vector<16xi32>
    %v27 = vc4kernel.splat %k27 : i32 -> vector<16xi32>
    %v28 = vc4kernel.splat %k28 : i32 -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    cf.br ^loop(%c0, %c0, %zero : i32, i32, vector<16xi32>)

  ^loop(%k : i32, %row : i32, %acc : vector<16xi32>):
    %more = arith.cmpi ult, %k, %c2 : i32
    cf.cond_br %more, ^step(%k, %row, %acc : i32, i32, vector<16xi32>), ^store(%acc : vector<16xi32>)

  ^step(%k_step : i32, %row_step : i32, %acc_step : vector<16xi32>):
    %byte_offset = arith.shli %k_step, %c7 : i32
    vc4kernel.vdr_load_rect_to_vpm %in, %byte_offset, %tile, %row_step, %c2, %c16, %pitch {max_rows = 2 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    %row_tail = arith.addi %row_step, %c1 : i32
    %tail = vc4kernel.pred.tail %c0, %active_cols : i32, i32 -> !vc4kernel.pred<16>
    %r0 = vc4kernel.vpm_read_fragment %tile, %row_step, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %r1 = vc4kernel.vpm_read_fragment %tile, %row_tail, %tail {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %sum0 = vc4kernel.fragment_add %acc_step, %r0 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sum1 = vc4kernel.fragment_add %sum0, %r1 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %k_next = arith.addi %k_step, %c1 : i32
    %row_next = arith.addi %row_step, %c2 : i32
    cf.br ^loop(%k_next, %row_next, %sum1 : i32, i32, vector<16xi32>)

  ^store(%sum : vector<16xi32>):
    %a1 = vc4kernel.fragment_add %sum, %v1 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a2 = vc4kernel.fragment_add %a1, %v2 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a3 = vc4kernel.fragment_add %a2, %v3 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a4 = vc4kernel.fragment_add %a3, %v4 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a5 = vc4kernel.fragment_add %a4, %v5 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a6 = vc4kernel.fragment_add %a5, %v6 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a7 = vc4kernel.fragment_add %a6, %v7 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a8 = vc4kernel.fragment_add %a7, %v8 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a9 = vc4kernel.fragment_add %a8, %v9 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a10 = vc4kernel.fragment_add %a9, %v10 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a11 = vc4kernel.fragment_add %a10, %v11 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a12 = vc4kernel.fragment_add %a11, %v12 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a13 = vc4kernel.fragment_add %a12, %v13 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a14 = vc4kernel.fragment_add %a13, %v14 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a15 = vc4kernel.fragment_add %a14, %v15 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a16 = vc4kernel.fragment_add %a15, %v16 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a17 = vc4kernel.fragment_add %a16, %v17 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a18 = vc4kernel.fragment_add %a17, %v18 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a19 = vc4kernel.fragment_add %a18, %v19 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a20 = vc4kernel.fragment_add %a19, %v20 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a21 = vc4kernel.fragment_add %a20, %v21 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a22 = vc4kernel.fragment_add %a21, %v22 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a23 = vc4kernel.fragment_add %a22, %v23 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a24 = vc4kernel.fragment_add %a23, %v24 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a25 = vc4kernel.fragment_add %a24, %v25 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a26 = vc4kernel.fragment_add %a25, %v26 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a27 = vc4kernel.fragment_add %a26, %v27 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %final = vc4kernel.fragment_add %a27, %v28 : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %final, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
