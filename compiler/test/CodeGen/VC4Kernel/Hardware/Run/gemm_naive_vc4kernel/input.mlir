module {
  vc4kernel.kernel @gemm_naive_vc4kernel(%a : i32, %b : i32, %c : i32, %m : i32, %n : i32, %k : i32, %lda : i32, %ldb : i32, %ldc : i32) attributes {
    public_name = "gemm_naive_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "c", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "m", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "k", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "lda", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "ldb", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "ldc", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %zero_scalar = arith.constant 0.000000e+00 : f32
    %pid_x = vc4kernel.program_id {axis = 0 : i32} : i32
    %row = vc4kernel.program_id {axis = 1 : i32} : i32
    %col_base = arith.shli %pid_x, %c4 : i32
    %row_active = arith.cmpi ult, %row, %m : i32
    cf.cond_br %row_active, ^check_cols, ^done

  ^check_cols:
    %col_active = arith.cmpi ult, %col_base, %n : i32
    cf.cond_br %col_active, ^compute, ^done

  ^compute:
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %tail = vc4kernel.pred.tail %col_base, %n : i32, i32 -> !vc4kernel.pred<16>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %zero = vc4kernel.splat %zero_scalar : f32 -> vector<16xf32>
    cf.br ^loop(%c0, %zero : i32, vector<16xf32>)

  ^loop(%kk : i32, %acc : vector<16xf32>):
    %more = arith.cmpi ult, %kk, %k : i32
    cf.cond_br %more, ^step(%kk, %acc : i32, vector<16xf32>), ^store(%acc : vector<16xf32>)

  ^step(%kk_step : i32, %acc_step : vector<16xf32>):
    %a_row_elems = arith.muli %row, %lda : i32
    %a_elem = arith.addi %a_row_elems, %kk_step : i32
    %a_bytes = arith.shli %a_elem, %c2 : i32
    %a_offsets = vc4kernel.splat %a_bytes : i32 -> vector<16xi32>

    %b_row_elems = arith.muli %kk_step, %ldb : i32
    %b_col_base = arith.addi %b_row_elems, %col_base : i32
    %b_base_bytes = arith.shli %b_col_base, %c2 : i32
    %b_base_vec = vc4kernel.splat %b_base_bytes : i32 -> vector<16xi32>
    %b_offsets = vc4kernel.fragment_add %b_base_vec, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>

    %a_values = vc4kernel.tmu_load_fragment %a, %a_offsets, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b_values = vc4kernel.tmu_load_fragment %b, %b_offsets, %tail : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %products = vc4kernel.fragment_mul %a_values, %b_values : vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %acc_next = vc4kernel.fragment_add %acc_step, %products : vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %kk_next = arith.addi %kk_step, %c1 : i32
    cf.br ^loop(%kk_next, %acc_next : i32, vector<16xf32>)

  ^store(%sum : vector<16xf32>):
    %c_row_elems = arith.muli %row, %ldc : i32
    %c_col_base = arith.addi %c_row_elems, %col_base : i32
    %c_base_bytes = arith.shli %c_col_base, %c2 : i32
    %c_base_vec = vc4kernel.splat %c_base_bytes : i32 -> vector<16xi32>
    %c_offsets = vc4kernel.fragment_add %c_base_vec, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %c, %c_offsets, %sum, %tail : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return

  ^done:
    vc4kernel.return
  }
}
