module {
  vc4kernel.kernel @saxpy_tail_vc4kernel(%x : i32, %y : i32, %alpha : f32, %n : i32) attributes {
    public_name = "saxpy_tail_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "y", kind = "buffer", direction = "inout", elem_type = "f32"},
      {name = "alpha", kind = "scalar", direction = "by_value", type = "f32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    %request = vc4kernel.program_id {axis = 0 : i32} : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %base_index = arith.shli %request, %c4 : i32
    %base_bytes = arith.shli %request, %c6 : i32
    %tail = vc4kernel.pred.tail %base_index, %n : i32, i32 -> !vc4kernel.pred<16>
    %lane_bytes_shift = vc4kernel.splat %c2 : i32 -> vector<16xi32>
    %lane_bytes = vc4kernel.fragment_alu.add %lanes, %lane_bytes_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %base_bytes_v = vc4kernel.splat %base_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %base_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %xv = vc4kernel.tmu_load_fragment %x, %byte_offsets, %tail : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %yv = vc4kernel.tmu_load_fragment %y, %byte_offsets, %tail : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %alpha_v = vc4kernel.splat %alpha : f32 -> vector<16xf32>
    %scaled = vc4kernel.fragment_alu.mul %alpha_v, %xv {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %sum = vc4kernel.fragment_alu.add %scaled, %yv {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    vc4kernel.vdw_store_fragment %y, %byte_offsets, %sum, %tail : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
