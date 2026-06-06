module {
  vc4kernel.kernel @tmu_two_load_loop_no_spill_vc4kernel(%a : i32, %b : i32, %out : i32, %k : i32, %active_cols : i32) attributes {
    public_name = "tmu_two_load_loop_no_spill_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "k", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c6 = arith.constant 6 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %tail = vc4kernel.pred.tail %c0, %active_cols : i32, i32 -> !vc4kernel.pred<16>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    cf.br ^loop(%c0, %zero : i32, vector<16xf32>)

  ^loop(%j : i32, %acc : vector<16xf32>):
    %more = arith.cmpi ult, %j, %k : i32
    cf.cond_br %more, ^step(%j, %acc : i32, vector<16xf32>), ^store(%acc : vector<16xf32>)

  ^step(%j_step : i32, %acc_step : vector<16xf32>):
    %a_bytes = arith.shli %j_step, %c2 : i32
    %a_offsets = vc4kernel.splat %a_bytes : i32 -> vector<16xi32>
    %b_row_bytes = arith.shli %j_step, %c6 : i32
    %b_row_bytes_v = vc4kernel.splat %b_row_bytes : i32 -> vector<16xi32>
    %b_offsets = vc4kernel.fragment_alu.add %b_row_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %a_values = vc4kernel.tmu_load_fragment %a, %a_offsets, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b_values = vc4kernel.tmu_load_fragment %b, %b_offsets, %tail : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %products = vc4kernel.fragment_alu.mul %a_values, %b_values {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc_next = vc4kernel.fragment_alu.add %acc_step, %products {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %j_next = arith.addi %j_step, %c1 : i32
    cf.br ^loop(%j_next, %acc_next : i32, vector<16xf32>)

  ^store(%sum : vector<16xf32>):
    %out_offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %out_offsets, %sum, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
