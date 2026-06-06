module {
  vc4kernel.kernel @fragment_alu_add_i32_logic_shift_vc4kernel(%a : i32, %b : i32, %out : i32) attributes {
    public_name = "fragment_alu_add_i32_logic_shift_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c7 = arith.constant 7 : i32
    %c15 = arith.constant 15 : i32
    %c31 = arith.constant 31 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %c2v = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_alu.add %lanes, %c2v {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %av = vc4kernel.tmu_load_fragment %a, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %bv = vc4kernel.tmu_load_fragment %b, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %s0 = vc4kernel.splat %c0 : i32 -> vector<16xi32>
    %s1 = vc4kernel.splat %c1 : i32 -> vector<16xi32>
    %s7 = vc4kernel.splat %c7 : i32 -> vector<16xi32>
    %s15 = vc4kernel.splat %c15 : i32 -> vector<16xi32>
    %s31 = vc4kernel.splat %c31 : i32 -> vector<16xi32>
    %r0 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r1 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<sub>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r2 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r3 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r4 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<xor>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r5 = vc4kernel.fragment_alu.add %av {opcode = #vc4kernel.add_alu_opcode<not>} : (vector<16xi32>) -> vector<16xi32>
    %r6 = vc4kernel.fragment_alu.add %av {opcode = #vc4kernel.add_alu_opcode<clz>} : (vector<16xi32>) -> vector<16xi32>
    %r7 = vc4kernel.fragment_alu.add %av, %s0 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r8 = vc4kernel.fragment_alu.add %av, %s1 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r9 = vc4kernel.fragment_alu.add %av, %s7 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r10 = vc4kernel.fragment_alu.add %av, %s15 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r11 = vc4kernel.fragment_alu.add %av, %s31 {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r12 = vc4kernel.fragment_alu.add %av, %s0 {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r13 = vc4kernel.fragment_alu.add %av, %s1 {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r14 = vc4kernel.fragment_alu.add %av, %s7 {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r15 = vc4kernel.fragment_alu.add %av, %s15 {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r16 = vc4kernel.fragment_alu.add %av, %s31 {opcode = #vc4kernel.add_alu_opcode<shr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r17 = vc4kernel.fragment_alu.add %av, %s0 {opcode = #vc4kernel.add_alu_opcode<asr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r18 = vc4kernel.fragment_alu.add %av, %s1 {opcode = #vc4kernel.add_alu_opcode<asr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r19 = vc4kernel.fragment_alu.add %av, %s7 {opcode = #vc4kernel.add_alu_opcode<asr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r20 = vc4kernel.fragment_alu.add %av, %s15 {opcode = #vc4kernel.add_alu_opcode<asr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r21 = vc4kernel.fragment_alu.add %av, %s31 {opcode = #vc4kernel.add_alu_opcode<asr>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r22 = vc4kernel.fragment_alu.add %av, %s0 {opcode = #vc4kernel.add_alu_opcode<ror>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r23 = vc4kernel.fragment_alu.add %av, %s1 {opcode = #vc4kernel.add_alu_opcode<ror>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r24 = vc4kernel.fragment_alu.add %av, %s7 {opcode = #vc4kernel.add_alu_opcode<ror>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r25 = vc4kernel.fragment_alu.add %av, %s15 {opcode = #vc4kernel.add_alu_opcode<ror>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %r26 = vc4kernel.fragment_alu.add %av, %s31 {opcode = #vc4kernel.add_alu_opcode<ror>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %r0, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b1 = arith.constant 64 : i32
    %b1v = vc4kernel.splat %b1 : i32 -> vector<16xi32>
    %o1 = vc4kernel.fragment_alu.add %b1v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o1, %r1, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b2 = arith.constant 128 : i32
    %b2v = vc4kernel.splat %b2 : i32 -> vector<16xi32>
    %o2 = vc4kernel.fragment_alu.add %b2v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o2, %r2, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b3 = arith.constant 192 : i32
    %b3v = vc4kernel.splat %b3 : i32 -> vector<16xi32>
    %o3 = vc4kernel.fragment_alu.add %b3v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o3, %r3, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b4 = arith.constant 256 : i32
    %b4v = vc4kernel.splat %b4 : i32 -> vector<16xi32>
    %o4 = vc4kernel.fragment_alu.add %b4v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o4, %r4, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b5 = arith.constant 320 : i32
    %b5v = vc4kernel.splat %b5 : i32 -> vector<16xi32>
    %o5 = vc4kernel.fragment_alu.add %b5v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o5, %r5, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b6 = arith.constant 384 : i32
    %b6v = vc4kernel.splat %b6 : i32 -> vector<16xi32>
    %o6 = vc4kernel.fragment_alu.add %b6v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o6, %r6, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b7 = arith.constant 448 : i32
    %b7v = vc4kernel.splat %b7 : i32 -> vector<16xi32>
    %o7 = vc4kernel.fragment_alu.add %b7v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o7, %r7, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b8 = arith.constant 512 : i32
    %b8v = vc4kernel.splat %b8 : i32 -> vector<16xi32>
    %o8 = vc4kernel.fragment_alu.add %b8v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o8, %r8, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b9 = arith.constant 576 : i32
    %b9v = vc4kernel.splat %b9 : i32 -> vector<16xi32>
    %o9 = vc4kernel.fragment_alu.add %b9v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o9, %r9, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b10 = arith.constant 640 : i32
    %b10v = vc4kernel.splat %b10 : i32 -> vector<16xi32>
    %o10 = vc4kernel.fragment_alu.add %b10v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o10, %r10, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b11 = arith.constant 704 : i32
    %b11v = vc4kernel.splat %b11 : i32 -> vector<16xi32>
    %o11 = vc4kernel.fragment_alu.add %b11v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o11, %r11, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b12 = arith.constant 768 : i32
    %b12v = vc4kernel.splat %b12 : i32 -> vector<16xi32>
    %o12 = vc4kernel.fragment_alu.add %b12v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o12, %r12, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b13 = arith.constant 832 : i32
    %b13v = vc4kernel.splat %b13 : i32 -> vector<16xi32>
    %o13 = vc4kernel.fragment_alu.add %b13v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o13, %r13, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b14 = arith.constant 896 : i32
    %b14v = vc4kernel.splat %b14 : i32 -> vector<16xi32>
    %o14 = vc4kernel.fragment_alu.add %b14v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o14, %r14, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b15 = arith.constant 960 : i32
    %b15v = vc4kernel.splat %b15 : i32 -> vector<16xi32>
    %o15 = vc4kernel.fragment_alu.add %b15v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o15, %r15, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b16 = arith.constant 1024 : i32
    %b16v = vc4kernel.splat %b16 : i32 -> vector<16xi32>
    %o16 = vc4kernel.fragment_alu.add %b16v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o16, %r16, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b17 = arith.constant 1088 : i32
    %b17v = vc4kernel.splat %b17 : i32 -> vector<16xi32>
    %o17 = vc4kernel.fragment_alu.add %b17v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o17, %r17, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b18 = arith.constant 1152 : i32
    %b18v = vc4kernel.splat %b18 : i32 -> vector<16xi32>
    %o18 = vc4kernel.fragment_alu.add %b18v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o18, %r18, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b19 = arith.constant 1216 : i32
    %b19v = vc4kernel.splat %b19 : i32 -> vector<16xi32>
    %o19 = vc4kernel.fragment_alu.add %b19v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o19, %r19, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b20 = arith.constant 1280 : i32
    %b20v = vc4kernel.splat %b20 : i32 -> vector<16xi32>
    %o20 = vc4kernel.fragment_alu.add %b20v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o20, %r20, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b21 = arith.constant 1344 : i32
    %b21v = vc4kernel.splat %b21 : i32 -> vector<16xi32>
    %o21 = vc4kernel.fragment_alu.add %b21v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o21, %r21, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b22 = arith.constant 1408 : i32
    %b22v = vc4kernel.splat %b22 : i32 -> vector<16xi32>
    %o22 = vc4kernel.fragment_alu.add %b22v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o22, %r22, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b23 = arith.constant 1472 : i32
    %b23v = vc4kernel.splat %b23 : i32 -> vector<16xi32>
    %o23 = vc4kernel.fragment_alu.add %b23v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o23, %r23, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b24 = arith.constant 1536 : i32
    %b24v = vc4kernel.splat %b24 : i32 -> vector<16xi32>
    %o24 = vc4kernel.fragment_alu.add %b24v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o24, %r24, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b25 = arith.constant 1600 : i32
    %b25v = vc4kernel.splat %b25 : i32 -> vector<16xi32>
    %o25 = vc4kernel.fragment_alu.add %b25v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o25, %r25, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b26 = arith.constant 1664 : i32
    %b26v = vc4kernel.splat %b26 : i32 -> vector<16xi32>
    %o26 = vc4kernel.fragment_alu.add %b26v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o26, %r26, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
