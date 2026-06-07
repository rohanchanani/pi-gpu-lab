module {
  vc4kernel.kernel @fragment_sfu_predicated_loop_vc4kernel(%a : i32, %b : i32, %out : i32, %k : i32, %active_cols : i32) attributes {
    public_name = "fragment_sfu_predicated_loop_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "k", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c6 = arith.constant 6 : i32
    %tail = vc4kernel.pred.tail %c0, %active_cols : i32, i32 -> !vc4kernel.pred<16>
    %zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %one = vc4kernel.fragment_const {value = dense<1.000000e+00> : vector<16xf32>} : vector<16xf32>
    %quarter = vc4kernel.fragment_const {value = dense<2.500000e-01> : vector<16xf32>} : vector<16xf32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %poison = vc4kernel.fragment_const {value = dense<[268435456, 268435460, 268435464, 268435468, 268435472, 268435476, 268435480, 268435484, 268435488, 268435492, 268435496, 268435500, 268435504, 268435508, 268435512, 268435516]> : vector<16xi32>} : vector<16xi32>
    cf.br ^loop(%c0, %zero : i32, vector<16xf32>)

  ^loop(%j : i32, %acc : vector<16xf32>):
    %more = arith.cmpi ult, %j, %k : i32
    cf.cond_br %more, ^step(%j, %acc : i32, vector<16xf32>), ^store(%acc : vector<16xf32>)

  ^step(%j_step : i32, %acc_step : vector<16xf32>):
    %a_byte = arith.shli %j_step, %c2 : i32
    %a_raw_offsets = vc4kernel.splat %a_byte : i32 -> vector<16xi32>
    %a_offsets = vc4kernel.fragment_select %tail, %a_raw_offsets, %poison : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %b_row = arith.shli %j_step, %c6 : i32
    %b_row_v = vc4kernel.splat %b_row : i32 -> vector<16xi32>
    %b_raw_offsets = vc4kernel.fragment_alu.add %b_row_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b_offsets = vc4kernel.fragment_select %tail, %b_raw_offsets, %poison : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %safe = arith.constant 0 : i32
    %a_load = vc4kernel.tmu_load_fragment %a, %a_offsets, %tail, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %b_load = vc4kernel.tmu_load_fragment %b, %b_offsets, %tail, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %recip_input = vc4kernel.fragment_select %tail, %a_load, %one : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %rsqrt_input = vc4kernel.fragment_select %tail, %b_load, %quarter : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %recip = vc4kernel.fragment_sfu %recip_input {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    %rsqrt = vc4kernel.fragment_sfu %rsqrt_input {kind = #vc4kernel.sfu_kind<rsqrt>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    %sum = vc4kernel.fragment_alu.add %recip, %rsqrt {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %active_sum = vc4kernel.fragment_select %tail, %sum, %zero : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    %next_acc = vc4kernel.fragment_alu.add %acc_step, %active_sum {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %j_next = arith.addi %j_step, %c1 : i32
    cf.br ^loop(%j_next, %next_acc : i32, vector<16xf32>)

  ^store(%result : vector<16xf32>):
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %result, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
