module {
  vc4kernel.kernel @fragment_cmp_f32_predicate_consumers_vc4kernel(%lhs : i32, %rhs : i32, %out : i32) attributes {
    public_name = "fragment_cmp_f32_predicate_consumers_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "lhs", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "rhs", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lhs_v = vc4kernel.tmu_load_fragment %lhs, %lane_bytes, %full {memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %rhs_v = vc4kernel.tmu_load_fragment %rhs, %lane_bytes, %full {memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>

    %lt = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<olt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %gt = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ogt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %eq_all = vc4kernel.fragment_cmp %lhs_v, %lhs_v {predicate = #vc4kernel.cmp<oeq>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %ne = vc4kernel.pred.or %lt, %gt : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %not_lt = vc4kernel.pred.not %lt : !vc4kernel.pred<16> -> !vc4kernel.pred<16>

    %true0_base = vc4kernel.fragment_const {value = dense<1694498816> : vector<16xi32>} : vector<16xi32>
    %true1_base = vc4kernel.fragment_const {value = dense<1694502912> : vector<16xi32>} : vector<16xi32>
    %true2_base = vc4kernel.fragment_const {value = dense<1694507008> : vector<16xi32>} : vector<16xi32>
    %true3_base = vc4kernel.fragment_const {value = dense<1694511104> : vector<16xi32>} : vector<16xi32>
    %false0_base = vc4kernel.fragment_const {value = dense<1711276032> : vector<16xi32>} : vector<16xi32>
    %false1_base = vc4kernel.fragment_const {value = dense<1711280128> : vector<16xi32>} : vector<16xi32>
    %false2_base = vc4kernel.fragment_const {value = dense<1711284224> : vector<16xi32>} : vector<16xi32>
    %false3_base = vc4kernel.fragment_const {value = dense<1711288320> : vector<16xi32>} : vector<16xi32>
    %true0 = vc4kernel.fragment_alu.add %true0_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true1 = vc4kernel.fragment_alu.add %true1_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true2 = vc4kernel.fragment_alu.add %true2_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true3 = vc4kernel.fragment_alu.add %true3_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false0 = vc4kernel.fragment_alu.add %false0_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false1 = vc4kernel.fragment_alu.add %false1_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false2 = vc4kernel.fragment_alu.add %false2_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false3 = vc4kernel.fragment_alu.add %false3_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true_any_base = vc4kernel.fragment_const {value = dense<1996492800> : vector<16xi32>} : vector<16xi32>
    %false_any_base = vc4kernel.fragment_const {value = dense<1996496896> : vector<16xi32>} : vector<16xi32>
    %true_all_base = vc4kernel.fragment_const {value = dense<2013270016> : vector<16xi32>} : vector<16xi32>
    %false_all_base = vc4kernel.fragment_const {value = dense<2013274112> : vector<16xi32>} : vector<16xi32>
    %true_any = vc4kernel.fragment_alu.add %true_any_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false_any = vc4kernel.fragment_alu.add %false_any_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true_all = vc4kernel.fragment_alu.add %true_all_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false_all = vc4kernel.fragment_alu.add %false_all_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    %sel0 = vc4kernel.fragment_select %lt, %true0, %false0 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel1 = vc4kernel.fragment_select %gt, %true1, %false1 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel2 = vc4kernel.fragment_select %ne, %true2, %false2 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel3 = vc4kernel.fragment_select %not_lt, %true3, %false3 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>

    %off0 = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %off1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %off2 = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %off3 = vc4kernel.fragment_const {value = dense<192> : vector<16xi32>} : vector<16xi32>
    %offs0 = vc4kernel.fragment_alu.add %off0, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs1 = vc4kernel.fragment_alu.add %off1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs2 = vc4kernel.fragment_alu.add %off2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs3 = vc4kernel.fragment_alu.add %off3, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs0, %sel0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs1, %sel1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs2, %sel2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs3, %sel3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %any_lt = vc4kernel.pred.any %lt : !vc4kernel.pred<16> -> i1
    cf.cond_br %any_lt, ^any_merge(%true_any : vector<16xi32>), ^any_merge(%false_any : vector<16xi32>)
  ^any_merge(%any_value : vector<16xi32>):
    %off4 = vc4kernel.fragment_const {value = dense<256> : vector<16xi32>} : vector<16xi32>
    %offs4 = vc4kernel.fragment_alu.add %off4, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs4, %any_value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %all_eq = vc4kernel.pred.all %eq_all : !vc4kernel.pred<16> -> i1
    cf.cond_br %all_eq, ^all_merge(%true_all : vector<16xi32>), ^all_merge(%false_all : vector<16xi32>)
  ^all_merge(%all_value : vector<16xi32>):
    %off5 = vc4kernel.fragment_const {value = dense<320> : vector<16xi32>} : vector<16xi32>
    %offs5 = vc4kernel.fragment_alu.add %off5, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs5, %all_value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
