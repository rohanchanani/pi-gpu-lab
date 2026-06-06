module {
  vc4kernel.kernel @fragment_cmp_i32_predicate_consumers_vc4kernel(%lhs : i32, %rhs : i32, %out : i32) attributes {
    public_name = "fragment_cmp_i32_predicate_consumers_vc4kernel",
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

    %signed = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<slt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %unsigned = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %eq_all = vc4kernel.fragment_cmp %lhs_v, %lhs_v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %and = vc4kernel.pred.and %signed, %unsigned : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %or = vc4kernel.pred.or %signed, %unsigned : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %not_signed = vc4kernel.pred.not %signed : !vc4kernel.pred<16> -> !vc4kernel.pred<16>

    %true0_base = vc4kernel.fragment_const {value = dense<1392508928> : vector<16xi32>} : vector<16xi32>
    %true1_base = vc4kernel.fragment_const {value = dense<1392513024> : vector<16xi32>} : vector<16xi32>
    %true2_base = vc4kernel.fragment_const {value = dense<1392517120> : vector<16xi32>} : vector<16xi32>
    %true3_base = vc4kernel.fragment_const {value = dense<1392521216> : vector<16xi32>} : vector<16xi32>
    %true4_base = vc4kernel.fragment_const {value = dense<1392525312> : vector<16xi32>} : vector<16xi32>
    %false0_base = vc4kernel.fragment_const {value = dense<1409286144> : vector<16xi32>} : vector<16xi32>
    %false1_base = vc4kernel.fragment_const {value = dense<1409290240> : vector<16xi32>} : vector<16xi32>
    %false2_base = vc4kernel.fragment_const {value = dense<1409294336> : vector<16xi32>} : vector<16xi32>
    %false3_base = vc4kernel.fragment_const {value = dense<1409298432> : vector<16xi32>} : vector<16xi32>
    %false4_base = vc4kernel.fragment_const {value = dense<1409302528> : vector<16xi32>} : vector<16xi32>
    %true0 = vc4kernel.fragment_alu.add %true0_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true1 = vc4kernel.fragment_alu.add %true1_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true2 = vc4kernel.fragment_alu.add %true2_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true3 = vc4kernel.fragment_alu.add %true3_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true4 = vc4kernel.fragment_alu.add %true4_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false0 = vc4kernel.fragment_alu.add %false0_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false1 = vc4kernel.fragment_alu.add %false1_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false2 = vc4kernel.fragment_alu.add %false2_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false3 = vc4kernel.fragment_alu.add %false3_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false4 = vc4kernel.fragment_alu.add %false4_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true_any_base = vc4kernel.fragment_const {value = dense<1895829504> : vector<16xi32>} : vector<16xi32>
    %false_any_base = vc4kernel.fragment_const {value = dense<1895833600> : vector<16xi32>} : vector<16xi32>
    %true_all_base = vc4kernel.fragment_const {value = dense<1912606720> : vector<16xi32>} : vector<16xi32>
    %false_all_base = vc4kernel.fragment_const {value = dense<1912610816> : vector<16xi32>} : vector<16xi32>
    %true_any = vc4kernel.fragment_alu.add %true_any_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false_any = vc4kernel.fragment_alu.add %false_any_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true_all = vc4kernel.fragment_alu.add %true_all_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false_all = vc4kernel.fragment_alu.add %false_all_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sel0 = vc4kernel.fragment_select %signed, %true0, %false0 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel1 = vc4kernel.fragment_select %unsigned, %true1, %false1 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel2 = vc4kernel.fragment_select %and, %true2, %false2 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel3 = vc4kernel.fragment_select %or, %true3, %false3 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel4 = vc4kernel.fragment_select %not_signed, %true4, %false4 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>

    %off0 = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %off1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %off2 = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %off3 = vc4kernel.fragment_const {value = dense<192> : vector<16xi32>} : vector<16xi32>
    %off4 = vc4kernel.fragment_const {value = dense<256> : vector<16xi32>} : vector<16xi32>
    %offs0 = vc4kernel.fragment_alu.add %off0, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs1 = vc4kernel.fragment_alu.add %off1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs2 = vc4kernel.fragment_alu.add %off2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs3 = vc4kernel.fragment_alu.add %off3, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs4 = vc4kernel.fragment_alu.add %off4, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs0, %sel0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs1, %sel1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs2, %sel2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs3, %sel3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs4, %sel4, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %any_signed = vc4kernel.pred.any %signed : !vc4kernel.pred<16> -> i1
    cf.cond_br %any_signed, ^any_merge(%true_any : vector<16xi32>), ^any_merge(%false_any : vector<16xi32>)
  ^any_merge(%any_value : vector<16xi32>):
    %off5 = vc4kernel.fragment_const {value = dense<320> : vector<16xi32>} : vector<16xi32>
    %offs5 = vc4kernel.fragment_alu.add %off5, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs5, %any_value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %all_unsigned = vc4kernel.pred.all %unsigned : !vc4kernel.pred<16> -> i1
    cf.cond_br %all_unsigned, ^all_merge(%true_all : vector<16xi32>), ^all_merge(%false_all : vector<16xi32>)
  ^all_merge(%all_value : vector<16xi32>):
    %off6 = vc4kernel.fragment_const {value = dense<384> : vector<16xi32>} : vector<16xi32>
    %offs6 = vc4kernel.fragment_alu.add %off6, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs6, %all_value, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
