module {
  vc4kernel.kernel @scalar_bitcast_i1_boundary_vc4kernel(%out : i32, %bits : i32, %x : i32, %y : i32, %offset_elems : i32) attributes {
    public_name = "scalar_bitcast_i1_boundary_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "bits", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "offset_elems", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c16 = arith.constant 16 : i32
    %one_f = arith.constant 1.000000e+00 : f32
    %select_true = arith.constant 286331153 : i32
    %select_false = arith.constant 572662306 : i32
    %ctrl_true = arith.constant 858993459 : i32
    %ctrl_false = arith.constant 1145324612 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>

    %as_f = arith.bitcast %bits : i32 to f32
    %roundtrip_bits = arith.bitcast %as_f : f32 to i32
    %one_bits = arith.bitcast %one_f : f32 to i32
    %gt = arith.cmpi sgt, %x, %y : i32
    %low = arith.trunci %x : i32 to i1
    %and = arith.andi %gt, %low : i1
    %or = arith.ori %gt, %low : i1
    %xor = arith.xori %gt, %low : i1
    %and_i = arith.extui %and : i1 to i32
    %or_i = arith.extui %or : i1 to i32
    %xor_i = arith.extui %xor : i1 to i32
    %low_i = arith.extui %low : i1 to i32
    %selected = arith.select %xor, %select_true, %select_false : i32
    cf.cond_br %and, ^and_true, ^and_false

  ^and_true:
    cf.br ^join(%ctrl_true : i32)

  ^and_false:
    cf.br ^join(%ctrl_false : i32)

  ^join(%control : i32):
    %base0 = arith.shli %offset_elems, %c2 : i32
    %base0_v = vc4kernel.splat %base0 : i32 -> vector<16xi32>
    %v0 = vc4kernel.splat %roundtrip_bits : i32 -> vector<16xi32>
    %off0 = vc4kernel.fragment_alu.add %base0_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off0, %v0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s1 = arith.addi %offset_elems, %c16 : i32
    %base1 = arith.shli %s1, %c2 : i32
    %base1_v = vc4kernel.splat %base1 : i32 -> vector<16xi32>
    %v1 = vc4kernel.splat %one_bits : i32 -> vector<16xi32>
    %off1 = vc4kernel.fragment_alu.add %base1_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off1, %v1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s2 = arith.addi %s1, %c16 : i32
    %base2 = arith.shli %s2, %c2 : i32
    %base2_v = vc4kernel.splat %base2 : i32 -> vector<16xi32>
    %v2 = vc4kernel.splat %and_i : i32 -> vector<16xi32>
    %off2 = vc4kernel.fragment_alu.add %base2_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off2, %v2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s3 = arith.addi %s2, %c16 : i32
    %base3 = arith.shli %s3, %c2 : i32
    %base3_v = vc4kernel.splat %base3 : i32 -> vector<16xi32>
    %v3 = vc4kernel.splat %or_i : i32 -> vector<16xi32>
    %off3 = vc4kernel.fragment_alu.add %base3_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off3, %v3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s4 = arith.addi %s3, %c16 : i32
    %base4 = arith.shli %s4, %c2 : i32
    %base4_v = vc4kernel.splat %base4 : i32 -> vector<16xi32>
    %v4 = vc4kernel.splat %xor_i : i32 -> vector<16xi32>
    %off4 = vc4kernel.fragment_alu.add %base4_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off4, %v4, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s5 = arith.addi %s4, %c16 : i32
    %base5 = arith.shli %s5, %c2 : i32
    %base5_v = vc4kernel.splat %base5 : i32 -> vector<16xi32>
    %v5 = vc4kernel.splat %low_i : i32 -> vector<16xi32>
    %off5 = vc4kernel.fragment_alu.add %base5_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off5, %v5, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s6 = arith.addi %s5, %c16 : i32
    %base6 = arith.shli %s6, %c2 : i32
    %base6_v = vc4kernel.splat %base6 : i32 -> vector<16xi32>
    %v6 = vc4kernel.splat %selected : i32 -> vector<16xi32>
    %off6 = vc4kernel.fragment_alu.add %base6_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off6, %v6, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    %s7 = arith.addi %s6, %c16 : i32
    %base7 = arith.shli %s7, %c2 : i32
    %base7_v = vc4kernel.splat %base7 : i32 -> vector<16xi32>
    %v7 = vc4kernel.splat %control : i32 -> vector<16xi32>
    %off7 = vc4kernel.fragment_alu.add %base7_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %off7, %v7, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>

    vc4kernel.return
  }
}
