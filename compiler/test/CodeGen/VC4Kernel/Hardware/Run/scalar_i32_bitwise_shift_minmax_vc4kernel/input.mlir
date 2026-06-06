module {
  vc4kernel.kernel @scalar_i32_bitwise_shift_minmax_vc4kernel(%out : i32, %a : i32, %b : i32, %shift : i32, %offset_elems : i32) attributes {
    public_name = "scalar_i32_bitwise_shift_minmax_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "a", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "b", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "shift", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %base = arith.shli %offset_elems, %c2 : i32
    %base_v = vc4kernel.splat %base : i32 -> vector<16xi32>

    %r0 = arith.andi %a, %b : i32
    %r1 = arith.ori %a, %b : i32
    %r2 = arith.xori %a, %b : i32
    %r3 = arith.shli %a, %shift : i32
    %r4 = arith.shrui %a, %shift : i32
    %r5 = arith.shrsi %a, %shift : i32
    %r6 = arith.minsi %a, %b : i32
    %r7 = arith.maxsi %a, %b : i32
    %r8 = arith.minui %a, %b : i32
    %r9 = arith.maxui %a, %b : i32

    %v0 = vc4kernel.splat %r0 : i32 -> vector<16xi32>
    %off0 = vc4kernel.fragment_alu.add %base_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off0, %v0, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s1 = arith.addi %offset_elems, %c16 : i32
    %b1 = arith.shli %s1, %c2 : i32
    %bv1 = vc4kernel.splat %b1 : i32 -> vector<16xi32>
    %v1 = vc4kernel.splat %r1 : i32 -> vector<16xi32>
    %off1 = vc4kernel.fragment_alu.add %bv1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off1, %v1, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s2a = arith.addi %s1, %c16 : i32
    %b2 = arith.shli %s2a, %c2 : i32
    %bv2 = vc4kernel.splat %b2 : i32 -> vector<16xi32>
    %v2 = vc4kernel.splat %r2 : i32 -> vector<16xi32>
    %off2 = vc4kernel.fragment_alu.add %bv2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off2, %v2, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s3a = arith.addi %s2a, %c16 : i32
    %b3 = arith.shli %s3a, %c2 : i32
    %bv3 = vc4kernel.splat %b3 : i32 -> vector<16xi32>
    %v3 = vc4kernel.splat %r3 : i32 -> vector<16xi32>
    %off3 = vc4kernel.fragment_alu.add %bv3, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off3, %v3, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s4a = arith.addi %s3a, %c16 : i32
    %b4 = arith.shli %s4a, %c2 : i32
    %bv4 = vc4kernel.splat %b4 : i32 -> vector<16xi32>
    %v4 = vc4kernel.splat %r4 : i32 -> vector<16xi32>
    %off4 = vc4kernel.fragment_alu.add %bv4, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off4, %v4, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s5a = arith.addi %s4a, %c16 : i32
    %b5 = arith.shli %s5a, %c2 : i32
    %bv5 = vc4kernel.splat %b5 : i32 -> vector<16xi32>
    %v5 = vc4kernel.splat %r5 : i32 -> vector<16xi32>
    %off5 = vc4kernel.fragment_alu.add %bv5, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off5, %v5, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s6a = arith.addi %s5a, %c16 : i32
    %b6 = arith.shli %s6a, %c2 : i32
    %bv6 = vc4kernel.splat %b6 : i32 -> vector<16xi32>
    %v6 = vc4kernel.splat %r6 : i32 -> vector<16xi32>
    %off6 = vc4kernel.fragment_alu.add %bv6, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off6, %v6, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s7a = arith.addi %s6a, %c16 : i32
    %b7 = arith.shli %s7a, %c2 : i32
    %bv7 = vc4kernel.splat %b7 : i32 -> vector<16xi32>
    %v7 = vc4kernel.splat %r7 : i32 -> vector<16xi32>
    %off7 = vc4kernel.fragment_alu.add %bv7, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off7, %v7, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s8a = arith.addi %s7a, %c16 : i32
    %b8 = arith.shli %s8a, %c2 : i32
    %bv8 = vc4kernel.splat %b8 : i32 -> vector<16xi32>
    %v8 = vc4kernel.splat %r8 : i32 -> vector<16xi32>
    %off8 = vc4kernel.fragment_alu.add %bv8, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off8, %v8, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s9a = arith.addi %s8a, %c16 : i32
    %b9 = arith.shli %s9a, %c2 : i32
    %bv9 = vc4kernel.splat %b9 : i32 -> vector<16xi32>
    %v9 = vc4kernel.splat %r9 : i32 -> vector<16xi32>
    %off9 = vc4kernel.fragment_alu.add %bv9, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off9, %v9, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
