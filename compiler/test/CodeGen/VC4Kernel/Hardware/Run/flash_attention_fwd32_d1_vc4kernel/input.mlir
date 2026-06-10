module {
  vc4kernel.kernel @flash_attention_fwd32_d1_vc4kernel(%q : i32, %k : i32, %v : i32, %out : i32, %rows : i32, %scale : f32) attributes {
    public_name = "flash_attention_fwd32_d1_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "q", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "k", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "v", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "rows", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "scale", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c6 = arith.constant 6 : i32
    %request = vc4kernel.program_id {axis = 0 : i32} : i32
    %in_bounds = arith.cmpi ult, %request, %rows : i32
    cf.cond_br %in_bounds, ^compute, ^done

  ^compute:
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %one_lane = vc4kernel.pred.tail %c0, %c1 : i32, i32 -> !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %tile1_byte_base = arith.constant 64 : i32
    %tile1_base_v = vc4kernel.splat %tile1_byte_base : i32 -> vector<16xi32>
    %tile1_offsets = vc4kernel.fragment_alu.add %tile1_base_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %safe0 = arith.constant 0 : i32

    %q_byte = arith.shli %request, %c2 : i32
    %q_offsets = vc4kernel.splat %q_byte : i32 -> vector<16xi32>
    %qv = vc4kernel.tmu_load_fragment %q, %q_offsets, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %scale_v = vc4kernel.splat %scale : f32 -> vector<16xf32>
    %log2e = vc4kernel.fragment_const {value = dense<1.4426950408889634> : vector<16xf32>} : vector<16xf32>

    %k0 = vc4kernel.tmu_load_fragment %k, %lane_bytes, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %v0 = vc4kernel.tmu_load_fragment %v, %lane_bytes, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %qk0 = vc4kernel.fragment_alu.mul %qv, %k0 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %scores0 = vc4kernel.fragment_alu.mul %qk0, %scale_v {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %m0 = vc4kernel.fragment_reduce %scores0, %full {kind = #vc4kernel.reduce<fmax>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %scores0_centered = vc4kernel.fragment_alu.add %scores0, %m0 {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %scores0_log2 = vc4kernel.fragment_alu.mul %scores0_centered, %log2e {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p0 = vc4kernel.fragment_sfu %scores0_log2 {kind = #vc4kernel.sfu_kind<exp>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    %l0 = vc4kernel.fragment_reduce %p0, %full {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %pv0 = vc4kernel.fragment_alu.mul %p0, %v0 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc0 = vc4kernel.fragment_reduce %pv0, %full {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>

    %k1 = vc4kernel.tmu_load_fragment %k, %tile1_offsets, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %v1 = vc4kernel.tmu_load_fragment %v, %tile1_offsets, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %qk1 = vc4kernel.fragment_alu.mul %qv, %k1 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %scores1 = vc4kernel.fragment_alu.mul %qk1, %scale_v {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %m1_block = vc4kernel.fragment_reduce %scores1, %full {kind = #vc4kernel.reduce<fmax>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %m_new = vc4kernel.fragment_alu.add %m0, %m1_block {opcode = #vc4kernel.add_alu_opcode<fmax>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>

    %old_delta = vc4kernel.fragment_alu.add %m0, %m_new {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %old_delta_log2 = vc4kernel.fragment_alu.mul %old_delta, %log2e {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %alpha = vc4kernel.fragment_sfu %old_delta_log2 {kind = #vc4kernel.sfu_kind<exp>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    %l0_scaled = vc4kernel.fragment_alu.mul %l0, %alpha {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc0_scaled = vc4kernel.fragment_alu.mul %acc0, %alpha {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>

    %scores1_centered = vc4kernel.fragment_alu.add %scores1, %m_new {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %scores1_log2 = vc4kernel.fragment_alu.mul %scores1_centered, %log2e {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p1 = vc4kernel.fragment_sfu %scores1_log2 {kind = #vc4kernel.sfu_kind<exp>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    %l1 = vc4kernel.fragment_reduce %p1, %full {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %pv1 = vc4kernel.fragment_alu.mul %p1, %v1 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc1 = vc4kernel.fragment_reduce %pv1, %full {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>

    %l = vc4kernel.fragment_alu.add %l0_scaled, %l1 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc = vc4kernel.fragment_alu.add %acc0_scaled, %acc1 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %inv_l = vc4kernel.fragment_sfu %l {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    %result = vc4kernel.fragment_alu.mul %acc, %inv_l {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>

    %row_bytes = arith.shli %request, %c6 : i32
    %row_bytes_v = vc4kernel.splat %row_bytes : i32 -> vector<16xi32>
    %out_offsets = vc4kernel.fragment_alu.add %row_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %out_offsets, %result, %one_lane {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return

  ^done:
    vc4kernel.return
  }
}
