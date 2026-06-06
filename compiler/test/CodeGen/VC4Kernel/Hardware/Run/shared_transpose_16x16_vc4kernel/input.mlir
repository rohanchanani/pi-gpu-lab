module {
  vc4kernel.kernel @shared_transpose_16x16_vc4kernel(%input : i32, %out : i32) attributes {
    public_name = "shared_transpose_16x16_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [
      {name = "input", kind = "buffer", direction = "in", elem_type = "u32"},
      {name = "out", kind = "buffer", direction = "out", elem_type = "u32"}
    ],
    warps_per_block = 4 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c6 = arith.constant 6 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %warp = vc4kernel.warp_id : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %row_base = arith.shli %warp, %c2 : i32
    %row0 = arith.addi %row_base, %c0 : i32
    %row1 = arith.addi %row_base, %c1 : i32
    %row2 = arith.addi %row_base, %c2 : i32
    %row3 = arith.addi %row_base, %c3 : i32
    %row0_bytes = arith.shli %row0, %c6 : i32
    %row1_bytes = arith.shli %row1, %c6 : i32
    %row2_bytes = arith.shli %row2, %c6 : i32
    %row3_bytes = arith.shli %row3, %c6 : i32
    %row0_v = vc4kernel.splat %row0_bytes : i32 -> vector<16xi32>
    %row1_v = vc4kernel.splat %row1_bytes : i32 -> vector<16xi32>
    %row2_v = vc4kernel.splat %row2_bytes : i32 -> vector<16xi32>
    %row3_v = vc4kernel.splat %row3_bytes : i32 -> vector<16xi32>
    %offs0 = vc4kernel.fragment_alu.add %row0_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs1 = vc4kernel.fragment_alu.add %row1_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs2 = vc4kernel.fragment_alu.add %row2_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %offs3 = vc4kernel.fragment_alu.add %row3_v, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %tile = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
    %load0 = vc4kernel.tmu_load_fragment %input, %offs0, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row0, %load0, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %load1 = vc4kernel.tmu_load_fragment %input, %offs1, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row1, %load1, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %load2 = vc4kernel.tmu_load_fragment %input, %offs2, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row2, %load2, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    %load3 = vc4kernel.tmu_load_fragment %input, %offs3, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vpm_write_fragment %tile, %row3, %load3, %full {orientation = #vc4kernel.vpm_orientation<horizontal>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.barrier
    %read0 = vc4kernel.vpm_read_fragment %tile, %row0, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs0, %read0, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %read1 = vc4kernel.vpm_read_fragment %tile, %row1, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs1, %read1, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %read2 = vc4kernel.vpm_read_fragment %tile, %row2, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs2, %read2, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %read3 = vc4kernel.vpm_read_fragment %tile, %row3, %full {orientation = #vc4kernel.vpm_orientation<vertical>, width = #vc4kernel.vpm_width<w32>, subword = #vc4kernel.vpm_subword<none>, x = 0 : i32, stride = 1 : i32} : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offs3, %read3, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
