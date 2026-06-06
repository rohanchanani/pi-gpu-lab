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
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %p7_safe0 = arith.constant 0 : i32
    %av = vc4kernel.tmu_load_fragment %a, %lane_bytes, %full, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %p7_safe1 = arith.constant 0 : i32
    %bv = vc4kernel.tmu_load_fragment %b, %lane_bytes, %full, %p7_safe1 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xi32>
    %s0 = vc4kernel.fragment_const {value = dense<0> : vector<16xi32>} : vector<16xi32>
    %s1 = vc4kernel.fragment_const {value = dense<1> : vector<16xi32>} : vector<16xi32>
    %s7 = vc4kernel.fragment_const {value = dense<7> : vector<16xi32>} : vector<16xi32>
    %s15 = vc4kernel.fragment_const {value = dense<15> : vector<16xi32>} : vector<16xi32>
    %s31 = vc4kernel.fragment_const {value = dense<31> : vector<16xi32>} : vector<16xi32>
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
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %r0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b1v = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %o1 = vc4kernel.fragment_alu.add %b1v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o1, %r1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b2v = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %o2 = vc4kernel.fragment_alu.add %b2v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o2, %r2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b3v = vc4kernel.fragment_const {value = dense<192> : vector<16xi32>} : vector<16xi32>
    %o3 = vc4kernel.fragment_alu.add %b3v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o3, %r3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b4v = vc4kernel.fragment_const {value = dense<256> : vector<16xi32>} : vector<16xi32>
    %o4 = vc4kernel.fragment_alu.add %b4v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o4, %r4, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b5v = vc4kernel.fragment_const {value = dense<320> : vector<16xi32>} : vector<16xi32>
    %o5 = vc4kernel.fragment_alu.add %b5v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o5, %r5, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b6v = vc4kernel.fragment_const {value = dense<384> : vector<16xi32>} : vector<16xi32>
    %o6 = vc4kernel.fragment_alu.add %b6v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o6, %r6, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b7v = vc4kernel.fragment_const {value = dense<448> : vector<16xi32>} : vector<16xi32>
    %o7 = vc4kernel.fragment_alu.add %b7v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o7, %r7, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b8v = vc4kernel.fragment_const {value = dense<512> : vector<16xi32>} : vector<16xi32>
    %o8 = vc4kernel.fragment_alu.add %b8v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o8, %r8, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b9v = vc4kernel.fragment_const {value = dense<576> : vector<16xi32>} : vector<16xi32>
    %o9 = vc4kernel.fragment_alu.add %b9v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o9, %r9, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b10v = vc4kernel.fragment_const {value = dense<640> : vector<16xi32>} : vector<16xi32>
    %o10 = vc4kernel.fragment_alu.add %b10v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o10, %r10, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b11v = vc4kernel.fragment_const {value = dense<704> : vector<16xi32>} : vector<16xi32>
    %o11 = vc4kernel.fragment_alu.add %b11v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o11, %r11, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b12v = vc4kernel.fragment_const {value = dense<768> : vector<16xi32>} : vector<16xi32>
    %o12 = vc4kernel.fragment_alu.add %b12v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o12, %r12, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b13v = vc4kernel.fragment_const {value = dense<832> : vector<16xi32>} : vector<16xi32>
    %o13 = vc4kernel.fragment_alu.add %b13v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o13, %r13, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b14v = vc4kernel.fragment_const {value = dense<896> : vector<16xi32>} : vector<16xi32>
    %o14 = vc4kernel.fragment_alu.add %b14v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o14, %r14, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b15v = vc4kernel.fragment_const {value = dense<960> : vector<16xi32>} : vector<16xi32>
    %o15 = vc4kernel.fragment_alu.add %b15v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o15, %r15, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b16v = vc4kernel.fragment_const {value = dense<1024> : vector<16xi32>} : vector<16xi32>
    %o16 = vc4kernel.fragment_alu.add %b16v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o16, %r16, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b17v = vc4kernel.fragment_const {value = dense<1088> : vector<16xi32>} : vector<16xi32>
    %o17 = vc4kernel.fragment_alu.add %b17v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o17, %r17, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b18v = vc4kernel.fragment_const {value = dense<1152> : vector<16xi32>} : vector<16xi32>
    %o18 = vc4kernel.fragment_alu.add %b18v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o18, %r18, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b19v = vc4kernel.fragment_const {value = dense<1216> : vector<16xi32>} : vector<16xi32>
    %o19 = vc4kernel.fragment_alu.add %b19v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o19, %r19, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b20v = vc4kernel.fragment_const {value = dense<1280> : vector<16xi32>} : vector<16xi32>
    %o20 = vc4kernel.fragment_alu.add %b20v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o20, %r20, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b21v = vc4kernel.fragment_const {value = dense<1344> : vector<16xi32>} : vector<16xi32>
    %o21 = vc4kernel.fragment_alu.add %b21v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o21, %r21, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b22v = vc4kernel.fragment_const {value = dense<1408> : vector<16xi32>} : vector<16xi32>
    %o22 = vc4kernel.fragment_alu.add %b22v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o22, %r22, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b23v = vc4kernel.fragment_const {value = dense<1472> : vector<16xi32>} : vector<16xi32>
    %o23 = vc4kernel.fragment_alu.add %b23v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o23, %r23, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b24v = vc4kernel.fragment_const {value = dense<1536> : vector<16xi32>} : vector<16xi32>
    %o24 = vc4kernel.fragment_alu.add %b24v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o24, %r24, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b25v = vc4kernel.fragment_const {value = dense<1600> : vector<16xi32>} : vector<16xi32>
    %o25 = vc4kernel.fragment_alu.add %b25v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o25, %r25, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b26v = vc4kernel.fragment_const {value = dense<1664> : vector<16xi32>} : vector<16xi32>
    %o26 = vc4kernel.fragment_alu.add %b26v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o26, %r26, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
