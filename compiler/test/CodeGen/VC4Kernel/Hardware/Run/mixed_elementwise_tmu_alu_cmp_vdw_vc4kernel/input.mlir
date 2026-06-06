module {
  vc4kernel.kernel @mixed_elementwise_tmu_alu_cmp_vdw_vc4kernel(%x : i32, %y : i32, %out : i32, %audit : i32, %n : i32, %scale : f32, %bias : f32, %threshold : f32) attributes {
    public_name = "mixed_elementwise_tmu_alu_cmp_vdw_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "y", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "audit", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "scale", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "bias", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "threshold", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    %safe0 = arith.constant 0 : i32
    %request = vc4kernel.program_id {axis = 0 : i32} : i32
    %base_index = arith.shli %request, %c4 : i32
    %base_bytes = arith.shli %request, %c6 : i32
    %tail = vc4kernel.pred.tail %base_index, %n : i32, i32 -> !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %poison = vc4kernel.fragment_const {value = dense<[268435456, 268435460, 268435464, 268435468, 268435472, 268435476, 268435480, 268435484, 268435488, 268435492, 268435496, 268435500, 268435504, 268435508, 268435512, 268435516]> : vector<16xi32>} : vector<16xi32>
    %base_bytes_v = vc4kernel.splat %base_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %base_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %request_offsets = vc4kernel.fragment_select %tail, %byte_offsets, %poison : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %xv = vc4kernel.tmu_load_fragment %x, %request_offsets, %tail, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %yv = vc4kernel.tmu_load_fragment %y, %request_offsets, %tail, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %scale_v = vc4kernel.splat %scale : f32 -> vector<16xf32>
    %bias_v = vc4kernel.splat %bias : f32 -> vector<16xf32>
    %threshold_v = vc4kernel.splat %threshold : f32 -> vector<16xf32>
    %scaled = vc4kernel.fragment_alu.mul %xv, %scale_v {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %sum0 = vc4kernel.fragment_alu.add %scaled, %yv {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %z = vc4kernel.fragment_alu.add %sum0, %bias_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %above = vc4kernel.fragment_cmp %z, %threshold_v {predicate = #vc4kernel.cmp<ogt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
    %store_pred = vc4kernel.pred.and %tail, %above : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %selected = vc4kernel.fragment_select %above, %z, %yv : !vc4kernel.pred<16>, vector<16xf32>, vector<16xf32> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %selected, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>

    %idx_base_v = vc4kernel.splat %base_index : i32 -> vector<16xi32>
    %idx = vc4kernel.fragment_alu.add %idx_base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %z_bits = vc4kernel.fragment_bitcast %z : vector<16xf32> -> vector<16xi32>
    %mask = vc4kernel.fragment_alu.add %z_bits, %idx {opcode = #vc4kernel.add_alu_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %three = vc4kernel.fragment_const {value = dense<3> : vector<16xi32>} : vector<16xi32>
    %mul = vc4kernel.fragment_alu.mul %idx, %three {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %mix = vc4kernel.fragment_alu.add %mask, %mul {opcode = #vc4kernel.add_alu_opcode<xor>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %clz = vc4kernel.fragment_alu.add %mix {opcode = #vc4kernel.add_alu_opcode<clz>} : (vector<16xi32>) -> vector<16xi32>
    %audit_v = vc4kernel.fragment_alu.add %mix, %clz {opcode = #vc4kernel.add_alu_opcode<max>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %limit = vc4kernel.fragment_const {value = dense<1000000> : vector<16xi32>} : vector<16xi32>
    %idx_in_range = vc4kernel.fragment_cmp %idx, %limit {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %audit_pred = vc4kernel.pred.and %store_pred, %idx_in_range : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %audit, %byte_offsets, %audit_v, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %any = vc4kernel.pred.any %audit_pred : !vc4kernel.pred<16> -> i1
    cf.cond_br %any, ^done, ^done
  ^done:
    vc4kernel.return
  }
}
