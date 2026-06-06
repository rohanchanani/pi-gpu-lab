module {
  vc4kernel.kernel @cooperative_id_writeback_vc4kernel(%out : i32) attributes {
    public_name = "cooperative_id_writeback_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "u32"}
    ],
    warps_per_block = 12 : i32
  } {
    %c6 = arith.constant 6 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %warp = vc4kernel.warp_id : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %warp_bytes = arith.shli %warp, %c6 : i32
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %warp_bytes_v = vc4kernel.splat %warp_bytes : i32 -> vector<16xi32>
    %offsets = vc4kernel.fragment_alu.add %warp_bytes_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tag_v = vc4kernel.fragment_const {value = dense<1660944384> : vector<16xi32>} : vector<16xi32>
    %warp_v = vc4kernel.splat %warp : i32 -> vector<16xi32>
    %warp_scaled_shift = vc4kernel.fragment_const {value = dense<4> : vector<16xi32>} : vector<16xi32>
    %warp_scaled = vc4kernel.fragment_alu.add %warp_v, %warp_scaled_shift {opcode = #vc4kernel.add_alu_opcode<shl>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %base = vc4kernel.fragment_alu.add %tag_v, %warp_scaled {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %values = vc4kernel.fragment_alu.add %base, %lanes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offsets, %values, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>, inactive_store = #vc4kernel.inactive_store<preserve>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
