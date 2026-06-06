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
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    %request = vc4kernel.program_id {axis = 0 : i32} : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %base_index = arith.shli %request, %c4 : i32
    %base_bytes = arith.shli %request, %c6 : i32
    %tail = vc4kernel.pred.tail %base_index, %n : i32, i32 -> !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %base_bytes_v = vc4kernel.splat %base_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_alu.add %base_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %p7_safe0 = arith.constant 0 : i32
    %xv = vc4kernel.tmu_load_fragment %x, %byte_offsets, %tail, %p7_safe0 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %p7_safe1 = arith.constant 0 : i32
    %yv = vc4kernel.tmu_load_fragment %y, %byte_offsets, %tail, %p7_safe1 {inactive_load = #vc4kernel.inactive_load<zero>, memory_path = #vc4kernel.memory_path<tmu_global_read>, coherency = #vc4kernel.coherency<readonly_tmu>} : i32, vector<16xi32>, !vc4kernel.pred<16>, i32 -> vector<16xf32>
    %alpha_v = vc4kernel.splat %alpha : f32 -> vector<16xf32>
    %scaled = vc4kernel.fragment_alu.mul %alpha_v, %xv {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %sum = vc4kernel.fragment_alu.add %scaled, %yv {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    vc4kernel.vdw_store_fragment %y, %byte_offsets, %sum, %tail {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
