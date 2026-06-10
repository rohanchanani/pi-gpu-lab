module {
  vc4kernel.kernel @flash_attention_coop12_fwd24_d1_vc4kernel(%q : i32, %k : i32, %v : i32, %out : i32, %scale : f32) attributes {
    public_name = "flash_attention_coop12_fwd24_d1_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [
      {name = "q", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "k", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "v", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "scale", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 12 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c6 = arith.constant 6 : i32
    %c12 = arith.constant 12 : i32
    %c24 = arith.constant 24 : i32
    %safe0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %one_lane = vc4kernel.pred.tail %c0, %c1 : i32, i32 -> !vc4kernel.pred<16>
    %warp = vc4kernel.warp_id : i32
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %neg_inf = vc4kernel.fragment_const {value = dense<-3.4028234663852886E+38> : vector<16xf32>} : vector<16xf32>
    %log2e = vc4kernel.fragment_const {value = dense<1.4426950408889634> : vector<16xf32>} : vector<16xf32>
    %scale_v = vc4kernel.splat %scale : f32 -> vector<16xf32>
    %tile = vc4kernel.vpm_alloc {rows = 24 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile

    %q_byte = arith.shli %warp, %c2 : i32
    %q_offsets = vc4kernel.splat %q_byte : i32 -> vector<16xi32>
    %qv = vc4kernel.tmu_load_fragment %q, %q_offsets, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>

    %k0_byte = arith.shli %warp, %c2 : i32
    %k0_offsets = vc4kernel.splat %k0_byte : i32 -> vector<16xi32>
    %k0_load = vc4kernel.tmu_load_fragment %k, %k0_offsets, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %v0_load = vc4kernel.tmu_load_fragment %v, %k0_offsets, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %vrow0 = arith.addi %warp, %c12 : i32
    vc4kernel.vpm_write_fragment %tile, %warp, %k0_load, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %vrow0, %v0_load, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.barrier
    cf.br ^loop0(%c0, %neg_inf, %zero, %zero : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^loop0(%j0 : i32, %m0 : vector<16xf32>, %l0 : vector<16xf32>, %acc0 : vector<16xf32>):
    %more0 = arith.cmpi ult, %j0, %c12 : i32
    cf.cond_br %more0, ^step0(%j0, %m0, %l0, %acc0 : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>), ^after0(%m0, %l0, %acc0 : vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^step0(%j0_step : i32, %m0_step : vector<16xf32>, %l0_step : vector<16xf32>, %acc0_step : vector<16xf32>):
    %vrow0_step = arith.addi %j0_step, %c12 : i32
    %kj0 = vc4kernel.vpm_read_fragment %tile, %j0_step, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %vj0 = vc4kernel.vpm_read_fragment %tile, %vrow0_step, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %qk0 = vc4kernel.fragment_alu.mul %qv, %kj0 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %score0 = vc4kernel.fragment_alu.mul %qk0, %scale_v {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %m0_new = vc4kernel.fragment_alu.add %m0_step, %score0 {opcode = #vc4kernel.add_alu_opcode<fmax>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %old0_delta = vc4kernel.fragment_alu.add %m0_step, %m0_new {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %old0_log2 = vc4kernel.fragment_alu.mul %old0_delta, %log2e {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %alpha0 = vc4kernel.fragment_sfu %old0_log2 {kind = #vc4kernel.sfu_kind<exp>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    %score0_delta = vc4kernel.fragment_alu.add %score0, %m0_new {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %score0_log2 = vc4kernel.fragment_alu.mul %score0_delta, %log2e {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p0 = vc4kernel.fragment_sfu %score0_log2 {kind = #vc4kernel.sfu_kind<exp>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    %l0_scaled = vc4kernel.fragment_alu.mul %l0_step, %alpha0 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %l0_new = vc4kernel.fragment_alu.add %l0_scaled, %p0 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc0_scaled = vc4kernel.fragment_alu.mul %acc0_step, %alpha0 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %pv0 = vc4kernel.fragment_alu.mul %p0, %vj0 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc0_new = vc4kernel.fragment_alu.add %acc0_scaled, %pv0 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %j0_next = arith.addi %j0_step, %c1 : i32
    cf.br ^loop0(%j0_next, %m0_new, %l0_new, %acc0_new : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^after0(%m_after0 : vector<16xf32>, %l_after0 : vector<16xf32>, %acc_after0 : vector<16xf32>):
    vc4kernel.barrier
    %global1 = arith.addi %warp, %c12 : i32
    %k1_byte = arith.shli %global1, %c2 : i32
    %k1_offsets = vc4kernel.splat %k1_byte : i32 -> vector<16xi32>
    %k1_load = vc4kernel.tmu_load_fragment %k, %k1_offsets, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %v1_load = vc4kernel.tmu_load_fragment %v, %k1_offsets, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %vrow1 = arith.addi %warp, %c12 : i32
    vc4kernel.vpm_write_fragment %tile, %warp, %k1_load, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.vpm_write_fragment %tile, %vrow1, %v1_load, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.barrier
    cf.br ^loop1(%c0, %m_after0, %l_after0, %acc_after0 : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^loop1(%j1 : i32, %m1 : vector<16xf32>, %l1 : vector<16xf32>, %acc1 : vector<16xf32>):
    %more1 = arith.cmpi ult, %j1, %c12 : i32
    cf.cond_br %more1, ^step1(%j1, %m1, %l1, %acc1 : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>), ^store(%l1, %acc1 : vector<16xf32>, vector<16xf32>)

  ^step1(%j1_step : i32, %m1_step : vector<16xf32>, %l1_step : vector<16xf32>, %acc1_step : vector<16xf32>):
    %vrow1_step = arith.addi %j1_step, %c12 : i32
    %kj1 = vc4kernel.vpm_read_fragment %tile, %j1_step, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %vj1 = vc4kernel.vpm_read_fragment %tile, %vrow1_step, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %qk1 = vc4kernel.fragment_alu.mul %qv, %kj1 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %score1 = vc4kernel.fragment_alu.mul %qk1, %scale_v {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %m1_new = vc4kernel.fragment_alu.add %m1_step, %score1 {opcode = #vc4kernel.add_alu_opcode<fmax>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %old1_delta = vc4kernel.fragment_alu.add %m1_step, %m1_new {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %old1_log2 = vc4kernel.fragment_alu.mul %old1_delta, %log2e {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %alpha1 = vc4kernel.fragment_sfu %old1_log2 {kind = #vc4kernel.sfu_kind<exp>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    %score1_delta = vc4kernel.fragment_alu.add %score1, %m1_new {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %score1_log2 = vc4kernel.fragment_alu.mul %score1_delta, %log2e {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %p1 = vc4kernel.fragment_sfu %score1_log2 {kind = #vc4kernel.sfu_kind<exp>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    %l1_scaled = vc4kernel.fragment_alu.mul %l1_step, %alpha1 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %l1_new = vc4kernel.fragment_alu.add %l1_scaled, %p1 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc1_scaled = vc4kernel.fragment_alu.mul %acc1_step, %alpha1 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %pv1 = vc4kernel.fragment_alu.mul %p1, %vj1 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc1_new = vc4kernel.fragment_alu.add %acc1_scaled, %pv1 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %j1_next = arith.addi %j1_step, %c1 : i32
    cf.br ^loop1(%j1_next, %m1_new, %l1_new, %acc1_new : i32, vector<16xf32>, vector<16xf32>, vector<16xf32>)

  ^store(%l_final : vector<16xf32>, %acc_final : vector<16xf32>):
    %inv_l = vc4kernel.fragment_sfu %l_final {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    %result = vc4kernel.fragment_alu.mul %acc_final, %inv_l {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %row_bytes = arith.shli %warp, %c6 : i32
    %row_bytes_v = vc4kernel.splat %row_bytes : i32 -> vector<16xi32>
    %out_offsets = vc4kernel.fragment_alu.add %row_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %out_offsets, %result, %one_lane {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
