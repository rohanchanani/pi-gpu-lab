module {
  vc4kernel.kernel @mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel(%a : i32, %b : i32, %out : i32, %k : i32, %active_cols : i32, %beta : f32) attributes {
    public_name = "mixed_f32_reduce_tmu_loop_spill_vdw_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "k", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "active_cols", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "beta", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c6 = arith.constant 6 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tail = vc4kernel.pred.tail %c0, %active_cols : i32, i32 -> !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %poison = vc4kernel.fragment_const {value = dense<[268435456, 268435460, 268435464, 268435468, 268435472, 268435476, 268435480, 268435484, 268435488, 268435492, 268435496, 268435500, 268435504, 268435508, 268435512, 268435516]> : vector<16xi32>} : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %beta_v = vc4kernel.splat %beta : f32 -> vector<16xf32>
    %p01 = vc4kernel.fragment_alu.add %beta_v, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p02 = vc4kernel.fragment_alu.add %p01, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p03 = vc4kernel.fragment_alu.add %p02, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p04 = vc4kernel.fragment_alu.add %p03, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p05 = vc4kernel.fragment_alu.add %p04, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p06 = vc4kernel.fragment_alu.add %p05, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p07 = vc4kernel.fragment_alu.add %p06, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p08 = vc4kernel.fragment_alu.add %p07, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p09 = vc4kernel.fragment_alu.add %p08, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p10 = vc4kernel.fragment_alu.add %p09, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p11 = vc4kernel.fragment_alu.add %p10, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p12 = vc4kernel.fragment_alu.add %p11, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p13 = vc4kernel.fragment_alu.add %p12, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p14 = vc4kernel.fragment_alu.add %p13, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p15 = vc4kernel.fragment_alu.add %p14, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p16 = vc4kernel.fragment_alu.add %p15, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p17 = vc4kernel.fragment_alu.add %p16, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p18 = vc4kernel.fragment_alu.add %p17, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p19 = vc4kernel.fragment_alu.add %p18, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p20 = vc4kernel.fragment_alu.add %p19, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p21 = vc4kernel.fragment_alu.add %p20, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p22 = vc4kernel.fragment_alu.add %p21, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p23 = vc4kernel.fragment_alu.add %p22, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p24 = vc4kernel.fragment_alu.add %p23, %beta_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    cf.br ^loop(%c0, %zero, %zero, %zero : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^loop(%j : i32, %acc : vector<16xf32>, %mn : vector<16xf32>, %mx : vector<16xf32>):
    %more = arith.cmpi ult, %j, %k : i32
    cf.cond_br %more, ^step(%j, %acc, %mn, %mx : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>), ^after(%acc, %mn, %mx : vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^step(%j_step : i32, %acc_step : vector<16xf32>, %mn_step : vector<16xf32>, %mx_step : vector<16xf32>):
    %a_byte = arith.shli %j_step, %c2 : i32
    %a_raw = vc4kernel.splat %a_byte : i32 -> vector<16xi32>
    %a_offsets = vc4kernel.fragment_select %tail, %a_raw, %poison : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %b_row = arith.shli %j_step, %c6 : i32
    %b_row_v = vc4kernel.splat %b_row : i32 -> vector<16xi32>
    %b_raw = vc4kernel.fragment_alu.add %b_row_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b_offsets = vc4kernel.fragment_select %tail, %b_raw, %poison : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %safe = arith.constant 0 : i32
    %av = vc4kernel.tmu_load_fragment %a, %a_offsets, %tail, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %bv = vc4kernel.tmu_load_fragment %b, %b_offsets, %tail, %safe {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %prod = vc4kernel.fragment_alu.mul %av, %bv {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %next_acc = vc4kernel.fragment_alu.add %acc_step, %prod {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %next_min = vc4kernel.fragment_alu.add %mn_step, %prod {opcode = #vc4kernel.add_alu_opcode<fmin>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %next_max = vc4kernel.fragment_alu.add %mx_step, %prod {opcode = #vc4kernel.add_alu_opcode<fmax>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %j_next = arith.addi %j_step, %c1 : i32
    cf.br ^loop(%j_next, %next_acc, %next_min, %next_max : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^after(%acc_final : vector<16xf32>, %mn_final : vector<16xf32>, %mx_final : vector<16xf32>):
    %radd = vc4kernel.fragment_reduce %acc_final, %tail {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %rmin = vc4kernel.fragment_reduce %mn_final, %tail {kind = #vc4kernel.reduce<fmin>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %rmax = vc4kernel.fragment_reduce %mx_final, %tail {kind = #vc4kernel.reduce<fmax>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %mix0 = vc4kernel.fragment_alu.add %radd, %rmin {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %mix1 = vc4kernel.fragment_alu.add %mix0, %rmax {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s01 = vc4kernel.fragment_alu.add %mix1, %p01 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s02 = vc4kernel.fragment_alu.add %s01, %p02 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s03 = vc4kernel.fragment_alu.add %s02, %p03 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s04 = vc4kernel.fragment_alu.add %s03, %p04 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s05 = vc4kernel.fragment_alu.add %s04, %p05 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s06 = vc4kernel.fragment_alu.add %s05, %p06 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s07 = vc4kernel.fragment_alu.add %s06, %p07 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s08 = vc4kernel.fragment_alu.add %s07, %p08 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s09 = vc4kernel.fragment_alu.add %s08, %p09 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s10 = vc4kernel.fragment_alu.add %s09, %p10 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s11 = vc4kernel.fragment_alu.add %s10, %p11 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s12 = vc4kernel.fragment_alu.add %s11, %p12 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s13 = vc4kernel.fragment_alu.add %s12, %p13 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s14 = vc4kernel.fragment_alu.add %s13, %p14 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s15 = vc4kernel.fragment_alu.add %s14, %p15 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s16 = vc4kernel.fragment_alu.add %s15, %p16 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s17 = vc4kernel.fragment_alu.add %s16, %p17 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s18 = vc4kernel.fragment_alu.add %s17, %p18 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s19 = vc4kernel.fragment_alu.add %s18, %p19 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s20 = vc4kernel.fragment_alu.add %s19, %p20 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s21 = vc4kernel.fragment_alu.add %s20, %p21 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s22 = vc4kernel.fragment_alu.add %s21, %p22 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s23 = vc4kernel.fragment_alu.add %s22, %p23 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %s24 = vc4kernel.fragment_alu.add %s23, %p24 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %positive = vc4kernel.fragment_cmp %s24, %zero {predicate = #vc4kernel.cmp<ogt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %selected = vc4kernel.fragment_select %positive, %s24, %zero : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %selected, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
