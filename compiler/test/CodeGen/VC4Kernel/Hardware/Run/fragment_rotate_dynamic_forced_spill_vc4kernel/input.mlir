module {
  vc4kernel.kernel @fragment_rotate_dynamic_forced_spill_vc4kernel(%out : i32, %amount : i32, %base_value : i32, %beta : i32) attributes {
    public_name = "fragment_rotate_dynamic_forced_spill_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "inout", elem_type = "i32"},
      {name = "amount", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "base_value", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "beta", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %beta_v = vc4kernel.splat %beta : i32 -> vector<16xi32>
    %p01 = vc4kernel.fragment_alu.add %beta_v, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p02 = vc4kernel.fragment_alu.add %p01, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p03 = vc4kernel.fragment_alu.add %p02, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p04 = vc4kernel.fragment_alu.add %p03, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p05 = vc4kernel.fragment_alu.add %p04, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p06 = vc4kernel.fragment_alu.add %p05, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p07 = vc4kernel.fragment_alu.add %p06, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p08 = vc4kernel.fragment_alu.add %p07, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p09 = vc4kernel.fragment_alu.add %p08, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p10 = vc4kernel.fragment_alu.add %p09, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p11 = vc4kernel.fragment_alu.add %p10, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p12 = vc4kernel.fragment_alu.add %p11, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p13 = vc4kernel.fragment_alu.add %p12, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p14 = vc4kernel.fragment_alu.add %p13, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p15 = vc4kernel.fragment_alu.add %p14, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p16 = vc4kernel.fragment_alu.add %p15, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p17 = vc4kernel.fragment_alu.add %p16, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p18 = vc4kernel.fragment_alu.add %p17, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p19 = vc4kernel.fragment_alu.add %p18, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p20 = vc4kernel.fragment_alu.add %p19, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p21 = vc4kernel.fragment_alu.add %p20, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p22 = vc4kernel.fragment_alu.add %p21, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p23 = vc4kernel.fragment_alu.add %p22, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p24 = vc4kernel.fragment_alu.add %p23, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p25 = vc4kernel.fragment_alu.add %p24, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p26 = vc4kernel.fragment_alu.add %p25, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p27 = vc4kernel.fragment_alu.add %p26, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p28 = vc4kernel.fragment_alu.add %p27, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p29 = vc4kernel.fragment_alu.add %p28, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p30 = vc4kernel.fragment_alu.add %p29, %beta_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %base_v = vc4kernel.splat %base_value : i32 -> vector<16xi32>
    %value = vc4kernel.fragment_alu.add %base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %rot = vc4kernel.fragment_rotate %value, %amount : vector<16xi32>, i32 -> vector<16xi32>
    %s01 = vc4kernel.fragment_alu.add %rot, %p01 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s02 = vc4kernel.fragment_alu.add %s01, %p02 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s03 = vc4kernel.fragment_alu.add %s02, %p03 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s04 = vc4kernel.fragment_alu.add %s03, %p04 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s05 = vc4kernel.fragment_alu.add %s04, %p05 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s06 = vc4kernel.fragment_alu.add %s05, %p06 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s07 = vc4kernel.fragment_alu.add %s06, %p07 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s08 = vc4kernel.fragment_alu.add %s07, %p08 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s09 = vc4kernel.fragment_alu.add %s08, %p09 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s10 = vc4kernel.fragment_alu.add %s09, %p10 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s11 = vc4kernel.fragment_alu.add %s10, %p11 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s12 = vc4kernel.fragment_alu.add %s11, %p12 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s13 = vc4kernel.fragment_alu.add %s12, %p13 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s14 = vc4kernel.fragment_alu.add %s13, %p14 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s15 = vc4kernel.fragment_alu.add %s14, %p15 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s16 = vc4kernel.fragment_alu.add %s15, %p16 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s17 = vc4kernel.fragment_alu.add %s16, %p17 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s18 = vc4kernel.fragment_alu.add %s17, %p18 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s19 = vc4kernel.fragment_alu.add %s18, %p19 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s20 = vc4kernel.fragment_alu.add %s19, %p20 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s21 = vc4kernel.fragment_alu.add %s20, %p21 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s22 = vc4kernel.fragment_alu.add %s21, %p22 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s23 = vc4kernel.fragment_alu.add %s22, %p23 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s24 = vc4kernel.fragment_alu.add %s23, %p24 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s25 = vc4kernel.fragment_alu.add %s24, %p25 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s26 = vc4kernel.fragment_alu.add %s25, %p26 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s27 = vc4kernel.fragment_alu.add %s26, %p27 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s28 = vc4kernel.fragment_alu.add %s27, %p28 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s29 = vc4kernel.fragment_alu.add %s28, %p29 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s30 = vc4kernel.fragment_alu.add %s29, %p30 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %s30, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
