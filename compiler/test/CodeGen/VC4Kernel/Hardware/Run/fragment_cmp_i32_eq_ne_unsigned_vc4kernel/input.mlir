module {
  vc4kernel.kernel @fragment_cmp_i32_eq_ne_unsigned_vc4kernel(%lhs : i32, %rhs : i32, %out : i32) attributes {
    public_name = "fragment_cmp_i32_eq_ne_unsigned_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "lhs", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "rhs", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %p7_safe0 = arith.constant 0 : i32
    %lhs_v = vc4kernel.tmu_load_fragment %lhs, %lane_bytes, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %p7_safe1 = arith.constant 0 : i32
    %rhs_v = vc4kernel.tmu_load_fragment %rhs, %lane_bytes, %full, %p7_safe1 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>

    %true0_base = vc4kernel.fragment_const {value = dense<822083584> : vector<16xi32>} : vector<16xi32>
    %true1_base = vc4kernel.fragment_const {value = dense<822087680> : vector<16xi32>} : vector<16xi32>
    %true2_base = vc4kernel.fragment_const {value = dense<822091776> : vector<16xi32>} : vector<16xi32>
    %true3_base = vc4kernel.fragment_const {value = dense<822095872> : vector<16xi32>} : vector<16xi32>
    %true4_base = vc4kernel.fragment_const {value = dense<822099968> : vector<16xi32>} : vector<16xi32>
    %true5_base = vc4kernel.fragment_const {value = dense<822104064> : vector<16xi32>} : vector<16xi32>
    %false0_base = vc4kernel.fragment_const {value = dense<1090519040> : vector<16xi32>} : vector<16xi32>
    %false1_base = vc4kernel.fragment_const {value = dense<1090523136> : vector<16xi32>} : vector<16xi32>
    %false2_base = vc4kernel.fragment_const {value = dense<1090527232> : vector<16xi32>} : vector<16xi32>
    %false3_base = vc4kernel.fragment_const {value = dense<1090531328> : vector<16xi32>} : vector<16xi32>
    %false4_base = vc4kernel.fragment_const {value = dense<1090535424> : vector<16xi32>} : vector<16xi32>
    %false5_base = vc4kernel.fragment_const {value = dense<1090539520> : vector<16xi32>} : vector<16xi32>
    %true0 = vc4kernel.fragment_alu.add %true0_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true1 = vc4kernel.fragment_alu.add %true1_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true2 = vc4kernel.fragment_alu.add %true2_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true3 = vc4kernel.fragment_alu.add %true3_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true4 = vc4kernel.fragment_alu.add %true4_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true5 = vc4kernel.fragment_alu.add %true5_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false0 = vc4kernel.fragment_alu.add %false0_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false1 = vc4kernel.fragment_alu.add %false1_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false2 = vc4kernel.fragment_alu.add %false2_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false3 = vc4kernel.fragment_alu.add %false3_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false4 = vc4kernel.fragment_alu.add %false4_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false5 = vc4kernel.fragment_alu.add %false5_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    %eq = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ne = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ne>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ult = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ule = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ule>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ugt = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ugt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %uge = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<uge>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel0 = vc4kernel.fragment_select %eq, %true0, %false0 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel1 = vc4kernel.fragment_select %ne, %true1, %false1 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel2 = vc4kernel.fragment_select %ult, %true2, %false2 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel3 = vc4kernel.fragment_select %ule, %true3, %false3 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel4 = vc4kernel.fragment_select %ugt, %true4, %false4 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel5 = vc4kernel.fragment_select %uge, %true5, %false5 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>

    %off0 = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %off1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %off2 = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %off3 = vc4kernel.fragment_const {value = dense<192> : vector<16xi32>} : vector<16xi32>
    %off4 = vc4kernel.fragment_const {value = dense<256> : vector<16xi32>} : vector<16xi32>
    %off5 = vc4kernel.fragment_const {value = dense<320> : vector<16xi32>} : vector<16xi32>
    %offs0 = vc4kernel.fragment_alu.add %off0, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs1 = vc4kernel.fragment_alu.add %off1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs2 = vc4kernel.fragment_alu.add %off2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs3 = vc4kernel.fragment_alu.add %off3, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs4 = vc4kernel.fragment_alu.add %off4, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs5 = vc4kernel.fragment_alu.add %off5, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs0, %sel0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs1, %sel1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs2, %sel2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs3, %sel3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs4, %sel4, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs5, %sel5, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
