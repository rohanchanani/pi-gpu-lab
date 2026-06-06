module {
  vc4kernel.kernel @tmu_load_safe_offset_loop_two_loads_vc4kernel(%a : i32, %b : i32, %out : i32, %k : i32) attributes {
    public_name = "tmu_load_safe_offset_loop_two_loads_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "k", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c6 = arith.constant 6 : i32
    %c15 = arith.constant 15 : i32
    %c16 = arith.constant 16 : i32
    %tail1 = vc4kernel.pred.tail %c0, %c1 : i32, i32 -> !vc4kernel.pred<16>
    %tail15 = vc4kernel.pred.tail %c0, %c15 : i32, i32 -> !vc4kernel.pred<16>
    %tail16 = vc4kernel.pred.tail %c0, %c16 : i32, i32 -> !vc4kernel.pred<16>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %poison_offsets = vc4kernel.fragment_const {value = dense<[268435456, 268435460, 268435464, 268435468, 268435472, 268435476, 268435480, 268435484, 268435488, 268435492, 268435496, 268435500, 268435504, 268435508, 268435512, 268435516]> : vector<16xi32>} : vector<16xi32>
    cf.br ^loop(%c0, %zero, %zero, %zero : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^loop(%j : i32, %acc1 : vector<16xf32>, %acc15 : vector<16xf32>, %acc16 : vector<16xf32>):
    %more = arith.cmpi ult, %j, %k : i32
    cf.cond_br %more, ^step(%j, %acc1, %acc15, %acc16 : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>), ^store(%acc1, %acc15, %acc16 : vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^step(%j_step : i32, %acc1_step : vector<16xf32>, %acc15_step : vector<16xf32>, %acc16_step : vector<16xf32>):
    %a_bytes = arith.shli %j_step, %c2 : i32
    %a_offsets_raw = vc4kernel.splat %a_bytes : i32 -> vector<16xi32>
    %a_offsets_1 = vc4kernel.fragment_select %tail1, %a_offsets_raw, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %a_offsets_15 = vc4kernel.fragment_select %tail15, %a_offsets_raw, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %b_row_bytes = arith.shli %j_step, %c6 : i32
    %b_row_bytes_v = vc4kernel.splat %b_row_bytes : i32 -> vector<16xi32>
    %b_offsets_16 = vc4kernel.fragment_alu.add %b_row_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b_offsets_1_raw = vc4kernel.fragment_alu.add %b_row_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b_offsets_15_raw = vc4kernel.fragment_alu.add %b_row_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b_offsets_1 = vc4kernel.fragment_select %tail1, %b_offsets_1_raw, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %b_offsets_15 = vc4kernel.fragment_select %tail15, %b_offsets_15_raw, %poison_offsets : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %safe_a = arith.constant 0 : i32
    %a1 = vc4kernel.tmu_load_fragment %a, %a_offsets_1, %tail1, %safe_a {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %a15 = vc4kernel.tmu_load_fragment %a, %a_offsets_15, %tail15, %safe_a {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %a16 = vc4kernel.tmu_load_fragment %a, %a_offsets_raw, %tail16, %safe_a {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %safe_b = arith.constant 0 : i32
    %b1 = vc4kernel.tmu_load_fragment %b, %b_offsets_1, %tail1, %safe_b {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %b15 = vc4kernel.tmu_load_fragment %b, %b_offsets_15, %tail15, %safe_b {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %b16 = vc4kernel.tmu_load_fragment %b, %b_offsets_16, %tail16, %safe_b {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %p1 = vc4kernel.fragment_alu.mul %a1, %b1 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p15 = vc4kernel.fragment_alu.mul %a15, %b15 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p16 = vc4kernel.fragment_alu.mul %a16, %b16 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %next1 = vc4kernel.fragment_alu.add %acc1_step, %p1 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %next15 = vc4kernel.fragment_alu.add %acc15_step, %p15 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %next16 = vc4kernel.fragment_alu.add %acc16_step, %p16 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %j_next = arith.addi %j_step, %c1 : i32
    cf.br ^loop(%j_next, %next1, %next15, %next16 : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^store(%sum1 : vector<16xf32>, %sum15 : vector<16xf32>, %sum16 : vector<16xf32>):
    %store1 = vc4kernel.fragment_const {value = dense<[64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124]> : vector<16xi32>} : vector<16xi32>
    %store2 = vc4kernel.fragment_const {value = dense<[128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188]> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %sum1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store1, %sum15, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %store2, %sum16, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
