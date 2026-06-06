module {
  vc4kernel.kernel @fragment_cmp_f32_finite_ordered_vc4kernel(%lhs : i32, %rhs : i32, %out : i32) attributes {
    public_name = "fragment_cmp_f32_finite_ordered_vc4kernel",
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
    %lhs_v = vc4kernel.tmu_load_fragment %lhs, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %rhs_v = vc4kernel.tmu_load_fragment %rhs, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>

    %true0_base = vc4kernel.fragment_const {value = dense<1627389952> : vector<16xi32>} : vector<16xi32>
    %true1_base = vc4kernel.fragment_const {value = dense<1627394048> : vector<16xi32>} : vector<16xi32>
    %true2_base = vc4kernel.fragment_const {value = dense<1627398144> : vector<16xi32>} : vector<16xi32>
    %true3_base = vc4kernel.fragment_const {value = dense<1627402240> : vector<16xi32>} : vector<16xi32>
    %true4_base = vc4kernel.fragment_const {value = dense<1627406336> : vector<16xi32>} : vector<16xi32>
    %true5_base = vc4kernel.fragment_const {value = dense<1627410432> : vector<16xi32>} : vector<16xi32>
    %false0_base = vc4kernel.fragment_const {value = dense<1644167168> : vector<16xi32>} : vector<16xi32>
    %false1_base = vc4kernel.fragment_const {value = dense<1644171264> : vector<16xi32>} : vector<16xi32>
    %false2_base = vc4kernel.fragment_const {value = dense<1644175360> : vector<16xi32>} : vector<16xi32>
    %false3_base = vc4kernel.fragment_const {value = dense<1644179456> : vector<16xi32>} : vector<16xi32>
    %false4_base = vc4kernel.fragment_const {value = dense<1644183552> : vector<16xi32>} : vector<16xi32>
    %false5_base = vc4kernel.fragment_const {value = dense<1644187648> : vector<16xi32>} : vector<16xi32>
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

    %oeq = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<oeq>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %one = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<one>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %olt = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<olt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %ole = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ole>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %ogt = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<ogt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %oge = vc4kernel.fragment_cmp %lhs_v, %rhs_v {predicate = #vc4kernel.cmp<oge>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %sel0 = vc4kernel.fragment_select %oeq, %true0, %false0 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel1 = vc4kernel.fragment_select %one, %true1, %false1 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel2 = vc4kernel.fragment_select %olt, %true2, %false2 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel3 = vc4kernel.fragment_select %ole, %true3, %false3 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel4 = vc4kernel.fragment_select %ogt, %true4, %false4 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %sel5 = vc4kernel.fragment_select %oge, %true5, %false5 : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>

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
    vc4kernel.vdw_store_fragment %out, %offs0, %sel0, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs1, %sel1, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs2, %sel2, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs3, %sel3, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs4, %sel4, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %offs5, %sel5, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
