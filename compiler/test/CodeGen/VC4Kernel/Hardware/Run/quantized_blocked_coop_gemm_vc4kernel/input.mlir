module {
  vc4kernel.kernel @quantized_blocked_coop_gemm_vc4kernel(%a : i32, %b : i32, %bias : i32, %c : i32) attributes {
    public_name = "quantized_blocked_coop_gemm_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "u16"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "u16"},
      {name = "bias", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "c", kind = "buffer", direction = "inout", elem_type = "u16"}
    ],
    warps_per_block = 12 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c5 = arith.constant 5 : i32
    %c12 = arith.constant 12 : i32
    %c16 = arith.constant 16 : i32
    %c24 = arith.constant 24 : i32
    %c32 = arith.constant 32 : i32
    %safe0 = arith.constant 0 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %warp = vc4kernel.warp_id : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %bias_v = vc4kernel.tmu_load_fragment %bias, %lane_bytes, %full, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %tile = vc4kernel.vpm_alloc {rows = 24 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    cf.br ^k_loop(%c0, %zero : i32, vector<16xf32>)

  ^k_loop(%k0 : i32, %acc : vector<16xf32>):
    %more_k = arith.cmpi ult, %k0, %c24 : i32
    cf.cond_br %more_k, ^load_tile(%acc : vector<16xf32>), ^store(%acc : vector<16xf32>)

  ^load_tile(%body_acc : vector<16xf32>):
    %a_row = arith.muli %warp, %c24 : i32
    %a_elem = arith.addi %a_row, %k0 : i32
    %a_byte = arith.shli %a_elem, %c1 : i32
    %b_row = arith.addi %k0, %warp : i32
    %b_elem = arith.muli %b_row, %c16 : i32
    %b_byte = arith.shli %b_elem, %c1 : i32
    %b_vpm_row = arith.addi %warp, %c12 : i32
    vc4kernel.vdr_load_rect_to_vpm %a, %a_byte, %tile, %warp, %c1, %c12, %c32 {max_rows = 1 : i32, max_cols = 12 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, dst_x = 0 : i32, subword_selector = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.vdr_load_rect_to_vpm %b, %b_byte, %tile, %b_vpm_row, %c1, %c16, %c32 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, dst_x = 0 : i32, subword_selector = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    vc4kernel.barrier
    cf.br ^inner(%c0, %body_acc : i32, vector<16xf32>)

  ^inner(%j : i32, %inner_acc : vector<16xf32>):
    %more_j = arith.cmpi ult, %j, %c12 : i32
    cf.cond_br %more_j, ^dot_step(%inner_acc : vector<16xf32>), ^next_tile(%inner_acc : vector<16xf32>)

  ^dot_step(%step_acc : vector<16xf32>):
    %a_raw = vc4kernel.vpm_read_fragment %tile, %warp, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, subword_selector = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %a_vals = vc4kernel.fragment_unpack %a_raw {source = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<to_f32>} : vector<16xi32> -> vector<16xf32>
    %j_v = vc4kernel.splat %j : i32 -> vector<16xi32>
    %j_pred = vc4kernel.fragment_cmp %lanes, %j_v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %a_scalar = vc4kernel.fragment_reduce %a_vals, %j_pred {kind = #vc4kernel.reduce<add>, fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>

    %b_row_read = arith.addi %j, %c12 : i32
    %b_raw = vc4kernel.vpm_read_fragment %tile, %b_row_read, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, subword_selector = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    %b_vals = vc4kernel.fragment_unpack %b_raw {source = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.unpack_policy<to_f32>} : vector<16xi32> -> vector<16xf32>
    %product = vc4kernel.fragment_alu.mul %a_scalar, %b_vals {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %next_acc = vc4kernel.fragment_alu.add %step_acc, %product {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %j_next = arith.addi %j, %c1 : i32
    cf.br ^inner(%j_next, %next_acc : i32, vector<16xf32>)

  ^next_tile(%tile_acc : vector<16xf32>):
    vc4kernel.barrier
    %k_next = arith.addi %k0, %c12 : i32
    cf.br ^k_loop(%k_next, %tile_acc : i32, vector<16xf32>)

  ^store(%final_acc : vector<16xf32>):
    %with_bias = vc4kernel.fragment_alu.add %final_acc, %bias_v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %packed = vc4kernel.fragment_pack %with_bias {dest = #vc4kernel.subword_type<f16>, layout = #vc4kernel.subword_layout<packed>, policy = #vc4kernel.pack_policy<from_f32>} : vector<16xf32> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %warp, %packed, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, subword_selector = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %row_bytes = arith.shli %warp, %c5 : i32
    vc4kernel.vdw_store_rect_from_vpm %tile, %warp, %c, %row_bytes, %c1, %c16, %c32 {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w16>, subword = #vc4kernel.vpm_subword<packed>, src_x = 0 : i32, subword_selector = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
    vc4kernel.return
  }
}
