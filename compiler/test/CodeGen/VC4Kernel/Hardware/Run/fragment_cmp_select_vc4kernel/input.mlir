module {
  vc4kernel.kernel @fragment_cmp_select_vc4kernel(%out : i32, %threshold : i32) attributes {
    public_name = "fragment_cmp_select_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "threshold", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %threshold_v = vc4kernel.splat %threshold : i32 -> vector<16xi32>
    %true_base_v = vc4kernel.fragment_const {value = dense<805306368> : vector<16xi32>} : vector<16xi32>
    %false_base_v = vc4kernel.fragment_const {value = dense<1073741824> : vector<16xi32>} : vector<16xi32>
    %step_v = vc4kernel.fragment_const {value = dense<4096> : vector<16xi32>} : vector<16xi32>
    %one_v = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %two_v = vc4kernel.fragment_const {value = dense<2> : vector<16xi32>} : vector<16xi32>
    %three_v = vc4kernel.fragment_const {value = dense<3> : vector<16xi32>} : vector<16xi32>
    %four_v = vc4kernel.fragment_const {value = dense<4> : vector<16xi32>} : vector<16xi32>
    %five_v = vc4kernel.fragment_const {value = dense<5> : vector<16xi32>} : vector<16xi32>

    %true0 = vc4kernel.fragment_alu.add %true_base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false0 = vc4kernel.fragment_alu.add %false_base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true1_base = vc4kernel.fragment_alu.add %true_base_v, %step_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false1_base = vc4kernel.fragment_alu.add %false_base_v, %step_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true1 = vc4kernel.fragment_alu.add %true1_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false1 = vc4kernel.fragment_alu.add %false1_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true2_base_mul = vc4kernel.fragment_alu.mul %step_v, %two_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false2_base_mul = vc4kernel.fragment_alu.mul %step_v, %two_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true2_base = vc4kernel.fragment_alu.add %true_base_v, %true2_base_mul {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false2_base = vc4kernel.fragment_alu.add %false_base_v, %false2_base_mul {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true2 = vc4kernel.fragment_alu.add %true2_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false2 = vc4kernel.fragment_alu.add %false2_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true3_base_mul = vc4kernel.fragment_alu.mul %step_v, %three_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false3_base_mul = vc4kernel.fragment_alu.mul %step_v, %three_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true3_base = vc4kernel.fragment_alu.add %true_base_v, %true3_base_mul {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false3_base = vc4kernel.fragment_alu.add %false_base_v, %false3_base_mul {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true3 = vc4kernel.fragment_alu.add %true3_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false3 = vc4kernel.fragment_alu.add %false3_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true4_base_mul = vc4kernel.fragment_alu.mul %step_v, %four_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false4_base_mul = vc4kernel.fragment_alu.mul %step_v, %four_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true4_base = vc4kernel.fragment_alu.add %true_base_v, %true4_base_mul {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false4_base = vc4kernel.fragment_alu.add %false_base_v, %false4_base_mul {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true4 = vc4kernel.fragment_alu.add %true4_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false4 = vc4kernel.fragment_alu.add %false4_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true5_base_mul = vc4kernel.fragment_alu.mul %step_v, %five_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false5_base_mul = vc4kernel.fragment_alu.mul %step_v, %five_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true5_base = vc4kernel.fragment_alu.add %true_base_v, %true5_base_mul {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false5_base = vc4kernel.fragment_alu.add %false_base_v, %false5_base_mul {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true5 = vc4kernel.fragment_alu.add %true5_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false5 = vc4kernel.fragment_alu.add %false5_base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    %eq = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ne = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ne>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ult = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ule = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ule>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ugt = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<ugt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %uge = vc4kernel.fragment_cmp %lanes, %threshold_v {predicate = #vc4kernel.cmp<uge>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %sel0 = vc4kernel.fragment_select %eq, %true0, %false0 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel1 = vc4kernel.fragment_select %ne, %true1, %false1 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel2 = vc4kernel.fragment_select %ult, %true2, %false2 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel3 = vc4kernel.fragment_select %ule, %true3, %false3 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel4 = vc4kernel.fragment_select %ugt, %true4, %false4 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel5 = vc4kernel.fragment_select %uge, %true5, %false5 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>

    %off0v = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %off64v = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %off128v = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %off192v = vc4kernel.fragment_const {value = dense<192> : vector<16xi32>} : vector<16xi32>
    %off256v = vc4kernel.fragment_const {value = dense<256> : vector<16xi32>} : vector<16xi32>
    %off320v = vc4kernel.fragment_const {value = dense<320> : vector<16xi32>} : vector<16xi32>
    %offs0 = vc4kernel.fragment_alu.add %off0v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs1 = vc4kernel.fragment_alu.add %off64v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs2 = vc4kernel.fragment_alu.add %off128v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs3 = vc4kernel.fragment_alu.add %off192v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs4 = vc4kernel.fragment_alu.add %off256v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs5 = vc4kernel.fragment_alu.add %off320v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs0, %sel0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs1, %sel1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs2, %sel2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs3, %sel3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs4, %sel4, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs5, %sel5, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
