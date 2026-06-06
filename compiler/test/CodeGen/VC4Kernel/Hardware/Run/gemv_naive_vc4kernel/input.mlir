module {
  vc4kernel.kernel @gemv_naive_vc4kernel(%a : i32, %x : i32, %y : i32, %m : i32, %n : i32, %lda : i32) attributes {
    public_name = "gemv_naive_vc4kernel",
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
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %request = vc4kernel.program_id {axis = 0 : i32} : i32
    %row_base = arith.shli %request, %c4 : i32
    %has_rows = arith.cmpi ult, %row_base, %m : i32
    cf.cond_br %has_rows, ^compute, ^done

  ^compute:
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %row_base_v = vc4kernel.splat %row_base : i32 -> vector<16xi32>
    %rows = vc4kernel.fragment_alu.add %row_base_v, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %row_tail = vc4kernel.pred.tail %row_base, %m : i32, i32 -> !vc4kernel.pred<16>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %zero = vc4kernel.fragment_const {value = dense<0.000000e+00> : vector<16xf32>} : vector<16xf32>
    cf.br ^loop(%c0, %zero : i32, vector<16xf32>)

  ^loop(%j : i32, %acc : vector<16xf32>):
    %more = arith.cmpi ult, %j, %n : i32
    cf.cond_br %more, ^step(%j, %acc : i32, vector<16xf32>), ^store(%acc : vector<16xf32>)

  ^step(%j_step : i32, %acc_step : vector<16xf32>):
    %lda_bytes = arith.shli %lda, %c2 : i32
    %x_bytes = arith.shli %j_step, %c2 : i32
    %lda_bytes_v = vc4kernel.splat %lda_bytes : i32 -> vector<16xi32>
    %row_byte_offsets = vc4kernel.fragment_alu.mul %rows, %lda_bytes_v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %j_bytes_v = vc4kernel.splat %x_bytes : i32 -> vector<16xi32>
    %a_bytes = vc4kernel.fragment_alu.add %row_byte_offsets, %j_bytes_v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %x_offsets = vc4kernel.splat %x_bytes : i32 -> vector<16xi32>
    %p7_safe0 = arith.constant 0 : i32
    %a_values = vc4kernel.tmu_load_fragment %a, %a_bytes, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %p7_safe1 = arith.constant 0 : i32
    %x_values = vc4kernel.tmu_load_fragment %x, %x_offsets, %full, %p7_safe1 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %products = vc4kernel.fragment_alu.mul %a_values, %x_values {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %acc_next = vc4kernel.fragment_alu.add %acc_step, %products {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %j_next = arith.addi %j_step, %c1 : i32
    cf.br ^loop(%j_next, %acc_next : i32, vector<16xf32>)

  ^store(%sum : vector<16xf32>):
    %row_base_bytes = arith.shli %row_base, %c2 : i32
    %row_base_bytes_v = vc4kernel.splat %row_base_bytes : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %y_bytes = vc4kernel.fragment_alu.add %row_base_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %y, %y_bytes, %sum, %row_tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return

  ^done:
    vc4kernel.return
  }
}
