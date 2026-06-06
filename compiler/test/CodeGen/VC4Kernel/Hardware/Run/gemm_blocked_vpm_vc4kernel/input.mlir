module {
  vc4kernel.kernel @gemm_blocked_vpm_vc4kernel(%a : i32, %b : i32, %c : i32, %m : i32, %n : i32, %k : i32, %lda : i32, %ldb : i32, %ldc : i32) attributes {
    public_name = "gemm_blocked_vpm_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "c", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "m", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "k", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "lda", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "ldb", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "ldc", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %zero_scalar = arith.constant 0.000000e+00 : f32
    %pid_x = vc4kernel.program_id {axis = 0 : i32} : i32
    %row = vc4kernel.program_id {axis = 1 : i32} : i32
    %col_base = arith.shli %pid_x, %c4 : i32
    %row_active = arith.cmpi ult, %row, %m : i32
    cf.cond_br %row_active, ^check_cols, ^done

  ^check_cols:
    %col_active = arith.cmpi ult, %col_base, %n : i32
    cf.cond_br %col_active, ^compute, ^done

  ^compute:
    %lda_bytes = arith.shli %lda, %c2 : i32
    %ldb_bytes = arith.shli %ldb, %c2 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_alu.add %lanes, %lane_bytes_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %tail = vc4kernel.pred.tail %col_base, %n : i32, i32 -> !vc4kernel.pred<16>
    %zero = vc4kernel.splat %zero_scalar : f32 -> vector<16xf32>
    %a_tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %b_tile = vc4kernel.vpm_alloc {rows = 4 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %cols_remaining = arith.subi %n, %col_base : i32
    %cols_lt_tile = arith.cmpi ult, %cols_remaining, %c16 : i32
    %active_cols = arith.select %cols_lt_tile, %cols_remaining, %c16 : i32
    cf.br ^loop(%c0, %zero : i32, vector<16xf32>)

  ^loop(%k0 : i32, %acc : vector<16xf32>):
    %done_k = arith.cmpi uge, %k0, %k : i32
    cf.cond_br %done_k, ^store(%acc : vector<16xf32>), ^body(%acc : vector<16xf32>)

  ^body(%body_acc : vector<16xf32>):
    %k_remaining = arith.subi %k, %k0 : i32
    %k_lt_tile = arith.cmpi ult, %k_remaining, %c4 : i32
    %active_k = arith.select %k_lt_tile, %k_remaining, %c4 : i32
    %a_row = arith.muli %row, %lda : i32
    %a_elem = arith.addi %a_row, %k0 : i32
    %a_byte_offset = arith.shli %a_elem, %c2 : i32
    vc4kernel.vdr_load_rect_to_vpm %a, %a_byte_offset, %a_tile, %c0, %c1, %active_k, %lda_bytes {max_rows = 1 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    %b_row = arith.muli %k0, %ldb : i32
    %b_col = arith.addi %b_row, %col_base : i32
    %b_byte_offset = arith.shli %b_col, %c2 : i32
    vc4kernel.vdr_load_rect_to_vpm %b, %b_byte_offset, %b_tile, %c0, %active_k, %active_cols, %ldb_bytes {max_rows = 4 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32

    %a_vec0 = vc4kernel.vpm_read_fragment %a_tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %k0v = vc4kernel.splat %c0 : i32 -> vector<16xi32>
    %k0p = vc4kernel.fragment_cmp %lanes, %k0v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %a0 = vc4kernel.fragment_reduce %a_vec0, %k0p {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b0 = vc4kernel.vpm_read_fragment %b_tile, %c0, %tail {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %p0 = vc4kernel.fragment_alu.mul %a0, %b0 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc0 = vc4kernel.fragment_alu.add %body_acc, %p0 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>

    %a_vec1 = vc4kernel.vpm_read_fragment %a_tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %k1v = vc4kernel.splat %c1 : i32 -> vector<16xi32>
    %k1p = vc4kernel.fragment_cmp %lanes, %k1v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %a1 = vc4kernel.fragment_reduce %a_vec1, %k1p {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b1 = vc4kernel.vpm_read_fragment %b_tile, %c1, %tail {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %p1 = vc4kernel.fragment_alu.mul %a1, %b1 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc1 = vc4kernel.fragment_alu.add %acc0, %p1 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>

    %a_vec2 = vc4kernel.vpm_read_fragment %a_tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %k2v = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %k2p = vc4kernel.fragment_cmp %lanes, %k2v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %a2 = vc4kernel.fragment_reduce %a_vec2, %k2p {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b2 = vc4kernel.vpm_read_fragment %b_tile, %c2, %tail {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %p2 = vc4kernel.fragment_alu.mul %a2, %b2 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc2 = vc4kernel.fragment_alu.add %acc1, %p2 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>

    %a_vec3 = vc4kernel.vpm_read_fragment %a_tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %k3v = vc4kernel.splat %c3 : i32 -> vector<16xi32>
    %k3p = vc4kernel.fragment_cmp %lanes, %k3v {predicate = #vc4kernel.cmp<eq>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %a3 = vc4kernel.fragment_reduce %a_vec3, %k3p {kind = #vc4kernel.reduce<add>} : vector<16xf32>, !vc4kernel.pred<16> -> vector<16xf32>
    %b3 = vc4kernel.vpm_read_fragment %b_tile, %c3, %tail {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %p3 = vc4kernel.fragment_alu.mul %a3, %b3 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc3 = vc4kernel.fragment_alu.add %acc2, %p3 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %next_k = arith.addi %k0, %c4 : i32
    cf.br ^loop(%next_k, %acc3 : i32, vector<16xf32>)

  ^store(%sum : vector<16xf32>):
    %c_row = arith.muli %row, %ldc : i32
    %c_col = arith.addi %c_row, %col_base : i32
    %c_base_bytes = arith.shli %c_col, %c2 : i32
    %c_base_vec = vc4kernel.splat %c_base_bytes : i32 -> vector<16xi32>
    %c_offsets = vc4kernel.fragment_alu.add %c_base_vec, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %c, %c_offsets, %sum, %tail : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return

  ^done:
    vc4kernel.return
  }
}
