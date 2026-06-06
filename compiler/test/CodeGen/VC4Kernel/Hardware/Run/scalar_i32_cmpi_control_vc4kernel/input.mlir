module {
  vc4kernel.kernel @scalar_i32_cmpi_control_vc4kernel(%out : i32, %a : i32, %b : i32, %offset_elems : i32) attributes {
    public_name = "scalar_i32_cmpi_control_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "a", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "b", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %t0 = arith.constant 4096 : i32
    %t1 = arith.constant 4097 : i32
    %t2 = arith.constant 4098 : i32
    %t3 = arith.constant 4099 : i32
    %t4 = arith.constant 4100 : i32
    %t5 = arith.constant 4101 : i32
    %t6 = arith.constant 4102 : i32
    %t7 = arith.constant 4103 : i32
    %t8 = arith.constant 4104 : i32
    %t9 = arith.constant 4105 : i32
    %f0 = arith.constant 8192 : i32
    %f1 = arith.constant 8193 : i32
    %f2 = arith.constant 8194 : i32
    %f3 = arith.constant 8195 : i32
    %f4 = arith.constant 8196 : i32
    %f5 = arith.constant 8197 : i32
    %f6 = arith.constant 8198 : i32
    %f7 = arith.constant 8199 : i32
    %f8 = arith.constant 8200 : i32
    %f9 = arith.constant 8201 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>

    %eq = arith.cmpi eq, %a, %b : i32
    %ne = arith.cmpi ne, %a, %b : i32
    %slt = arith.cmpi slt, %a, %b : i32
    %sle = arith.cmpi sle, %a, %b : i32
    %sgt = arith.cmpi sgt, %a, %b : i32
    %sge = arith.cmpi sge, %a, %b : i32
    %ult = arith.cmpi ult, %a, %b : i32
    %ule = arith.cmpi ule, %a, %b : i32
    %ugt = arith.cmpi ugt, %a, %b : i32
    %uge = arith.cmpi uge, %a, %b : i32

    %r0 = arith.select %eq, %t0, %f0 : i32
    %r1 = arith.select %ne, %t1, %f1 : i32
    %r2 = arith.select %slt, %t2, %f2 : i32
    %r3 = arith.select %sle, %t3, %f3 : i32
    %r4 = arith.select %sgt, %t4, %f4 : i32
    %r5 = arith.select %sge, %t5, %f5 : i32
    %r6 = arith.select %ult, %t6, %f6 : i32
    %r7 = arith.select %ule, %t7, %f7 : i32
    %r8 = arith.select %ugt, %t8, %f8 : i32
    %r9 = arith.select %uge, %t9, %f9 : i32

    %base0b = arith.shli %offset_elems, %c2 : i32
    %base0v = vc4kernel.splat %base0b : i32 -> vector<16xi32>
    %off0 = vc4kernel.fragment_alu.add %base0v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v0 = vc4kernel.splat %r0 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off0, %v0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s1 = arith.addi %offset_elems, %c16 : i32
    %b1 = arith.shli %s1, %c2 : i32
    %bv1 = vc4kernel.splat %b1 : i32 -> vector<16xi32>
    %off1 = vc4kernel.fragment_alu.add %bv1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v1 = vc4kernel.splat %r1 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off1, %v1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s2 = arith.addi %s1, %c16 : i32
    %b2 = arith.shli %s2, %c2 : i32
    %bv2 = vc4kernel.splat %b2 : i32 -> vector<16xi32>
    %off2 = vc4kernel.fragment_alu.add %bv2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v2 = vc4kernel.splat %r2 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off2, %v2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s3 = arith.addi %s2, %c16 : i32
    %b3 = arith.shli %s3, %c2 : i32
    %bv3 = vc4kernel.splat %b3 : i32 -> vector<16xi32>
    %off3 = vc4kernel.fragment_alu.add %bv3, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v3 = vc4kernel.splat %r3 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off3, %v3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s4 = arith.addi %s3, %c16 : i32
    %b4 = arith.shli %s4, %c2 : i32
    %bv4 = vc4kernel.splat %b4 : i32 -> vector<16xi32>
    %off4 = vc4kernel.fragment_alu.add %bv4, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v4 = vc4kernel.splat %r4 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off4, %v4, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s5 = arith.addi %s4, %c16 : i32
    %b5 = arith.shli %s5, %c2 : i32
    %bv5 = vc4kernel.splat %b5 : i32 -> vector<16xi32>
    %off5 = vc4kernel.fragment_alu.add %bv5, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v5 = vc4kernel.splat %r5 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off5, %v5, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s6 = arith.addi %s5, %c16 : i32
    %b6 = arith.shli %s6, %c2 : i32
    %bv6 = vc4kernel.splat %b6 : i32 -> vector<16xi32>
    %off6 = vc4kernel.fragment_alu.add %bv6, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v6 = vc4kernel.splat %r6 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off6, %v6, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s7 = arith.addi %s6, %c16 : i32
    %b7 = arith.shli %s7, %c2 : i32
    %bv7 = vc4kernel.splat %b7 : i32 -> vector<16xi32>
    %off7 = vc4kernel.fragment_alu.add %bv7, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v7 = vc4kernel.splat %r7 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off7, %v7, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s8 = arith.addi %s7, %c16 : i32
    %b8 = arith.shli %s8, %c2 : i32
    %bv8 = vc4kernel.splat %b8 : i32 -> vector<16xi32>
    %off8 = vc4kernel.fragment_alu.add %bv8, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v8 = vc4kernel.splat %r8 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off8, %v8, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s9 = arith.addi %s8, %c16 : i32
    %b9 = arith.shli %s9, %c2 : i32
    %bv9 = vc4kernel.splat %b9 : i32 -> vector<16xi32>
    %off9 = vc4kernel.fragment_alu.add %bv9, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v9 = vc4kernel.splat %r9 : i32 -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off9, %v9, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    cf.cond_br %slt, ^signed_less, ^not_signed_less
  ^signed_less:
    %s10t = arith.addi %s9, %c16 : i32
    %b10t = arith.shli %s10t, %c2 : i32
    %bv10t = vc4kernel.splat %b10t : i32 -> vector<16xi32>
    %off10t = vc4kernel.fragment_alu.add %bv10t, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %vt = vc4kernel.fragment_const {value = dense<12288> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off10t, %vt, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    cf.br ^done
  ^not_signed_less:
    %s10f = arith.addi %s9, %c16 : i32
    %b10f = arith.shli %s10f, %c2 : i32
    %bv10f = vc4kernel.splat %b10f : i32 -> vector<16xi32>
    %off10f = vc4kernel.fragment_alu.add %bv10f, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %vf = vc4kernel.fragment_const {value = dense<16384> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off10f, %vf, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    cf.br ^done
  ^done:
    vc4kernel.return
  }
}
