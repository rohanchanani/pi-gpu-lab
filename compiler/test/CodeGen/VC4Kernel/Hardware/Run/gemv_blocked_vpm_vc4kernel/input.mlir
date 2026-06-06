module {
  vc4kernel.kernel @gemv_blocked_vpm_vc4kernel(%a : i32, %x : i32, %y : i32, %m : i32, %n : i32, %lda : i32) attributes {
    public_name = "gemv_blocked_vpm_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "x", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "y", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "m", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "lda", kind = "scalar", direction = "by_value", type = "i32"}
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
    %one_scalar = arith.constant 1.000000e+00 : f32
    %tile_id = vc4kernel.program_id {axis = 0 : i32} : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    %row_base = arith.shli %tile_id, %c4 : i32
    %has_rows = arith.cmpi ult, %row_base, %m : i32
    cf.cond_br %has_rows, ^compute, ^done

  ^compute:
    %lda_bytes = arith.shli %lda, %c2 : i32
    %row_base_lda = arith.muli %row_base, %lda : i32
    %tile = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %reserve = vc4kernel.vpm_alloc {rows = 47 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    cf.br ^loop(%c0, %zero : i32, vector<16xf32>)

  ^loop(%k0 : i32, %acc : vector<16xf32>):
    %done_k = arith.cmpi uge, %k0, %n : i32
    cf.cond_br %done_k, ^store(%acc : vector<16xf32>), ^body(%acc : vector<16xf32>)

  ^body(%body_acc : vector<16xf32>):
    %a_elem = arith.addi %row_base_lda, %k0 : i32
    %a_byte_offset = arith.shli %a_elem, %c2 : i32
    vc4kernel.vdr_load_rect_to_vpm %a, %a_byte_offset, %tile, %c0, %c16, %c4, %lda_bytes {max_rows = 16 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, memory_path = #vc4kernel.memory_path<vdr_global_to_vpm>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
    %a_col0 = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %x0_bytes = arith.shli %k0, %c2 : i32
    %x0_offsets = vc4kernel.splat %x0_bytes : i32 -> vector<16xi32>
    %p7_safe0 = arith.constant 0 : i32
    %x0 = vc4kernel.tmu_load_fragment %x, %x0_offsets, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %prod0 = vc4kernel.fragment_alu.mul %a_col0, %x0 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc0 = vc4kernel.fragment_alu.add %body_acc, %prod0 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %k1 = arith.addi %k0, %c1 : i32
    %k1_active = arith.cmpi ult, %k1, %n : i32
    %k1_scale_scalar = arith.select %k1_active, %one_scalar, %zero_scalar : f32
    %k1_scale = vc4kernel.splat %k1_scale_scalar : f32 -> vector<16xf32>
    %x1_index = arith.select %k1_active, %k1, %k0 : i32
    %x1_bytes = arith.shli %x1_index, %c2 : i32
    %x1_offsets = vc4kernel.splat %x1_bytes : i32 -> vector<16xi32>
    %p7_safe1 = arith.constant 0 : i32
    %x1 = vc4kernel.tmu_load_fragment %x, %x1_offsets, %full, %p7_safe1 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %a_col1 = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 1 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %prod1_raw = vc4kernel.fragment_alu.mul %a_col1, %x1 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %prod1 = vc4kernel.fragment_alu.mul %prod1_raw, %k1_scale {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc1 = vc4kernel.fragment_alu.add %acc0, %prod1 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %k2 = arith.addi %k0, %c2 : i32
    %k2_active = arith.cmpi ult, %k2, %n : i32
    %k2_scale_scalar = arith.select %k2_active, %one_scalar, %zero_scalar : f32
    %k2_scale = vc4kernel.splat %k2_scale_scalar : f32 -> vector<16xf32>
    %x2_index = arith.select %k2_active, %k2, %k0 : i32
    %x2_bytes = arith.shli %x2_index, %c2 : i32
    %x2_offsets = vc4kernel.splat %x2_bytes : i32 -> vector<16xi32>
    %p7_safe2 = arith.constant 0 : i32
    %x2 = vc4kernel.tmu_load_fragment %x, %x2_offsets, %full, %p7_safe2 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %a_col2 = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 2 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %prod2_raw = vc4kernel.fragment_alu.mul %a_col2, %x2 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %prod2 = vc4kernel.fragment_alu.mul %prod2_raw, %k2_scale {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc2 = vc4kernel.fragment_alu.add %acc1, %prod2 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %k3 = arith.addi %k0, %c3 : i32
    %k3_active = arith.cmpi ult, %k3, %n : i32
    %k3_scale_scalar = arith.select %k3_active, %one_scalar, %zero_scalar : f32
    %k3_scale = vc4kernel.splat %k3_scale_scalar : f32 -> vector<16xf32>
    %x3_index = arith.select %k3_active, %k3, %k0 : i32
    %x3_bytes = arith.shli %x3_index, %c2 : i32
    %x3_offsets = vc4kernel.splat %x3_bytes : i32 -> vector<16xi32>
    %p7_safe3 = arith.constant 0 : i32
    %x3 = vc4kernel.tmu_load_fragment %x, %x3_offsets, %full, %p7_safe3 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %a_col3 = vc4kernel.vpm_read_fragment %tile, %c0, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 3 : i32, stride = 1 : i32, memory_path = #vc4kernel.memory_path<vpm_qpu>, coherency = #vc4kernel.coherency<vpm_local>} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xf32>
    %prod3_raw = vc4kernel.fragment_alu.mul %a_col3, %x3 {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %prod3 = vc4kernel.fragment_alu.mul %prod3_raw, %k3_scale {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc3 = vc4kernel.fragment_alu.add %acc2, %prod3 {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %next_k = arith.addi %k0, %c4 : i32
    cf.br ^loop(%next_k, %acc3 : i32, vector<16xf32>)

  ^store(%sum : vector<16xf32>):
    %row_tail = vc4kernel.pred.tail %row_base, %m : i32, i32 -> !vc4kernel.pred<16>
    %row0_bytes = arith.shli %row_base, %c2 : i32
    %row0_v = vc4kernel.splat %row0_bytes : i32 -> vector<16xi32>
    %offs = vc4kernel.fragment_alu.add %row0_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %y, %offs, %sum, %row_tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return

  ^done:
    vc4kernel.return
  }
}
