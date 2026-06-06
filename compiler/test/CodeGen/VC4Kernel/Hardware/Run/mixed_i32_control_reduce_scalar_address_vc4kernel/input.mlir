module {
  vc4kernel.kernel @mixed_i32_control_reduce_scalar_address_vc4kernel(%input : i32, %out : i32, %n : i32, %stride : i32, %seed : i32) attributes {
    public_name = "mixed_i32_control_reduce_scalar_address_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "stride", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "seed", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c5 = arith.constant 5 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %c192 = arith.constant 192 : i32
    %c256 = arith.constant 256 : i32
    %c320 = arith.constant 320 : i32
    %c384 = arith.constant 384 : i32
    %c448 = arith.constant 448 : i32
    %c512 = arith.constant 512 : i32
    %c576 = arith.constant 576 : i32
    %c640 = arith.constant 640 : i32
    %c704 = arith.constant 704 : i32
    %mask = arith.constant 16711935 : i32
    %tag = arith.constant 1426063360 : i32
    %safe0 = arith.constant 0 : i32
    %request = vc4kernel.program_id {axis = 0 : i32} : i32
    %base_index = arith.muli %request, %c16 : i32
    %base_bytes = arith.shli %base_index, %c2 : i32
    %tail = vc4kernel.pred.tail %base_index, %n : i32, i32 -> !vc4kernel.pred<16>
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %empty = vc4kernel.pred.empty : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %poison = vc4kernel.fragment_const {value = dense<[268435456, 268435460, 268435464, 268435468, 268435472, 268435476, 268435480, 268435484, 268435488, 268435492, 268435496, 268435500, 268435504, 268435508, 268435512, 268435516]> : vector<16xi32>} : vector<16xi32>
    %base_bytes_v = vc4kernel.splat %base_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %base_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %request_offsets = vc4kernel.fragment_select %tail, %byte_offsets, %poison : !vc4kernel.pred<16>, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %values = vc4kernel.tmu_load_fragment %input, %request_offsets, %tail, %safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>

    %mul = arith.muli %seed, %stride : i32
    %and = arith.andi %mul, %mask : i32
    %or = arith.ori %and, %tag : i32
    %xor = arith.xori %or, %seed : i32
    %shl = arith.shli %xor, %c3 : i32
    %shru = arith.shrui %xor, %c5 : i32
    %shrs = arith.shrsi %xor, %c5 : i32
    %min_s = arith.minsi %shrs, %shl : i32
    %max_u = arith.maxui %shru, %mul : i32
    %slt = arith.cmpi slt, %seed, %stride : i32
    %ult = arith.cmpi ult, %seed, %stride : i32
    %low = arith.trunci %mul : i32 to i1
    %bool0 = arith.xori %slt, %ult : i1
    %bool1 = arith.ori %bool0, %low : i1
    %bool_i = arith.extui %bool1 : i1 to i32
    %ctrl_a = arith.addi %max_u, %bool_i : i32
    %ctrl_b = arith.subi %min_s, %bool_i : i32
    cf.cond_br %slt, ^signed_less(%ctrl_a : i32), ^not_signed_less(%ctrl_b : i32)
  ^signed_less(%ctrl_less : i32):
    cf.br ^merge(%ctrl_less : i32)
  ^not_signed_less(%ctrl_ge : i32):
    cf.br ^merge(%ctrl_ge : i32)
  ^merge(%ctrl : i32):
    %out_base = arith.muli %request, %c704 : i32
    %out_base_v = vc4kernel.splat %out_base : i32 -> vector<16xi32>
    %off0 = vc4kernel.fragment_alu.add %out_base_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b1 = arith.addi %out_base, %c64 : i32
    %b1v = vc4kernel.splat %b1 : i32 -> vector<16xi32>
    %off1 = vc4kernel.fragment_alu.add %b1v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b2 = arith.addi %out_base, %c128 : i32
    %b2v = vc4kernel.splat %b2 : i32 -> vector<16xi32>
    %off2 = vc4kernel.fragment_alu.add %b2v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b3 = arith.addi %out_base, %c192 : i32
    %b3v = vc4kernel.splat %b3 : i32 -> vector<16xi32>
    %off3 = vc4kernel.fragment_alu.add %b3v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b4 = arith.addi %out_base, %c256 : i32
    %b4v = vc4kernel.splat %b4 : i32 -> vector<16xi32>
    %off4 = vc4kernel.fragment_alu.add %b4v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b5 = arith.addi %out_base, %c320 : i32
    %b5v = vc4kernel.splat %b5 : i32 -> vector<16xi32>
    %off5 = vc4kernel.fragment_alu.add %b5v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b6 = arith.addi %out_base, %c384 : i32
    %b6v = vc4kernel.splat %b6 : i32 -> vector<16xi32>
    %off6 = vc4kernel.fragment_alu.add %b6v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b7 = arith.addi %out_base, %c448 : i32
    %b7v = vc4kernel.splat %b7 : i32 -> vector<16xi32>
    %off7 = vc4kernel.fragment_alu.add %b7v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b8 = arith.addi %out_base, %c512 : i32
    %b8v = vc4kernel.splat %b8 : i32 -> vector<16xi32>
    %off8 = vc4kernel.fragment_alu.add %b8v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b9 = arith.addi %out_base, %c576 : i32
    %b9v = vc4kernel.splat %b9 : i32 -> vector<16xi32>
    %off9 = vc4kernel.fragment_alu.add %b9v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b10 = arith.addi %out_base, %c640 : i32
    %b10v = vc4kernel.splat %b10 : i32 -> vector<16xi32>
    %off10 = vc4kernel.fragment_alu.add %b10v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>

    %zero = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %neg = vc4kernel.fragment_cmp %values, %zero {predicate = #vc4kernel.cmp<slt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %ult_zero = vc4kernel.fragment_cmp %values, %zero {predicate = #vc4kernel.cmp<ult>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
    %cmp_mask = vc4kernel.pred.or %neg, %ult_zero : !vc4kernel.pred<16>, !vc4kernel.pred<16> -> !vc4kernel.pred<16>
    %r_add = vc4kernel.fragment_reduce %values, %tail {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %r_min_s = vc4kernel.fragment_reduce %values, %tail {kind = #vc4kernel.reduce<min_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %r_max_s = vc4kernel.fragment_reduce %values, %tail {kind = #vc4kernel.reduce<max_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %r_min_u = vc4kernel.fragment_reduce %values, %tail {kind = #vc4kernel.reduce<min_u>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %r_max_u = vc4kernel.fragment_reduce %values, %tail {kind = #vc4kernel.reduce<max_u>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %r_and = vc4kernel.fragment_reduce %values, %tail {kind = #vc4kernel.reduce<bit_and>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %r_or = vc4kernel.fragment_reduce %values, %tail {kind = #vc4kernel.reduce<bit_or>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %r_xor = vc4kernel.fragment_reduce %values, %tail {kind = #vc4kernel.reduce<bit_xor>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %ctrl_v = vc4kernel.splat %ctrl : i32 -> vector<16xi32>
    %r_empty = vc4kernel.fragment_reduce %values, %empty {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %r_cmp = vc4kernel.fragment_reduce %values, %cmp_mask {kind = #vc4kernel.reduce<bit_or>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off0, %r_add, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off1, %r_min_s, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off2, %r_max_s, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off3, %r_min_u, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off4, %r_max_u, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off5, %r_and, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off6, %r_or, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off7, %r_xor, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off8, %ctrl_v, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off9, %r_empty, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %off10, %r_cmp, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
