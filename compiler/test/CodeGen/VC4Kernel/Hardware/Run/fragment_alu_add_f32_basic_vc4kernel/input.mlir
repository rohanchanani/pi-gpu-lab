module {
  vc4kernel.kernel @fragment_alu_add_f32_basic_vc4kernel(%a : i32, %b : i32, %out : i32) attributes {
    public_name = "fragment_alu_add_f32_basic_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "a", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "b", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %c2v = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_alu.add %lanes, %c2v {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %av = vc4kernel.tmu_load_fragment %a, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %bv = vc4kernel.tmu_load_fragment %b, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %r0 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r1 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fsub>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r2 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fmin>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r3 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fmax>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r4 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fminabs>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %r5 = vc4kernel.fragment_alu.add %av, %bv {opcode = #vc4kernel.add_alu_opcode<fmaxabs>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %r0, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b1 = arith.constant 64 : i32
    %b1v = vc4kernel.splat %b1 : i32 -> vector<16xi32>
    %o1 = vc4kernel.fragment_alu.add %b1v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o1, %r1, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b2 = arith.constant 128 : i32
    %b2v = vc4kernel.splat %b2 : i32 -> vector<16xi32>
    %o2 = vc4kernel.fragment_alu.add %b2v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o2, %r2, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b3 = arith.constant 192 : i32
    %b3v = vc4kernel.splat %b3 : i32 -> vector<16xi32>
    %o3 = vc4kernel.fragment_alu.add %b3v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o3, %r3, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b4 = arith.constant 256 : i32
    %b4v = vc4kernel.splat %b4 : i32 -> vector<16xi32>
    %o4 = vc4kernel.fragment_alu.add %b4v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o4, %r4, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    %b5 = arith.constant 320 : i32
    %b5v = vc4kernel.splat %b5 : i32 -> vector<16xi32>
    %o5 = vc4kernel.fragment_alu.add %b5v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o5, %r5, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
