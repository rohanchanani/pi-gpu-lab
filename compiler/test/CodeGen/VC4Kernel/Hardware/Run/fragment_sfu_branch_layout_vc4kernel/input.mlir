module {
  vc4kernel.kernel @fragment_sfu_branch_layout_vc4kernel(%input : i32, %out : i32, %control : i32, %offset_elems : i32) attributes {
    public_name = "fragment_sfu_branch_layout_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "control", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %safe0 = arith.constant 0 : i32
    %safe64 = arith.constant 64 : i32
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %row1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %row1_offsets = vc4kernel.fragment_alu.add %row1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offset_bytes = arith.shli %offset_elems, %c2 : i32
    %offset_v = vc4kernel.splat %offset_bytes : i32 -> vector<16xi32>
    %out_offsets = vc4kernel.fragment_alu.add %offset_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %do_rsqrt = arith.cmpi ne, %control, %c0 : i32
    cf.cond_br %do_rsqrt, ^rsqrt_path, ^recip_path

  ^recip_path:
    %recip_input = vc4kernel.tmu_load_fragment %input, %lane_bytes, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %recip = vc4kernel.fragment_sfu %recip_input {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %out_offsets, %recip, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    cf.br ^done

  ^rsqrt_path:
    %rsqrt_input = vc4kernel.tmu_load_fragment %input, %row1_offsets, %full, %safe64 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %rsqrt = vc4kernel.fragment_sfu %rsqrt_input {kind = #vc4kernel.sfu_kind<rsqrt>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %out_offsets, %rsqrt, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    cf.br ^done

  ^done:
    vc4kernel.return
  }
}
