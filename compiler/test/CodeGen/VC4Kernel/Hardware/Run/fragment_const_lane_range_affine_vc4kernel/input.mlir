module {
  vc4kernel.kernel @fragment_const_lane_range_affine_vc4kernel(%out : i32) attributes {
    public_name = "fragment_const_lane_range_affine_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %v0 = vc4kernel.fragment_const {value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>} : vector<16xi32>
    %v1 = vc4kernel.fragment_const {value = dense<[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]> : vector<16xi32>} : vector<16xi32>
    %v2 = vc4kernel.fragment_const {value = dense<[7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37]> : vector<16xi32>} : vector<16xi32>
    %v3 = vc4kernel.fragment_const {value = dense<[7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67]> : vector<16xi32>} : vector<16xi32>
    %v4 = vc4kernel.fragment_const {value = dense<[7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127]> : vector<16xi32>} : vector<16xi32>
    %v5 = vc4kernel.fragment_const {value = dense<[7, 23, 39, 55, 71, 87, 103, 119, 135, 151, 167, 183, 199, 215, 231, 247]> : vector<16xi32>} : vector<16xi32>
    %v6 = vc4kernel.fragment_const {value = dense<[1000, 999, 998, 997, 996, 995, 994, 993, 992, 991, 990, 989, 988, 987, 986, 985]> : vector<16xi32>} : vector<16xi32>
    %v7 = vc4kernel.fragment_const {value = dense<[1000, 998, 996, 994, 992, 990, 988, 986, 984, 982, 980, 978, 976, 974, 972, 970]> : vector<16xi32>} : vector<16xi32>
    %v8 = vc4kernel.fragment_const {value = dense<[1000, 996, 992, 988, 984, 980, 976, 972, 968, 964, 960, 956, 952, 948, 944, 940]> : vector<16xi32>} : vector<16xi32>
    %v9 = vc4kernel.fragment_const {value = dense<[1000, 992, 984, 976, 968, 960, 952, 944, 936, 928, 920, 912, 904, 896, 888, 880]> : vector<16xi32>} : vector<16xi32>
    %v10 = vc4kernel.fragment_const {value = dense<[1000, 984, 968, 952, 936, 920, 904, 888, 872, 856, 840, 824, 808, 792, 776, 760]> : vector<16xi32>} : vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %lane_bytes, %v0, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b1 = vc4kernel.fragment_const {value = dense<64> : vector<16xi32>} : vector<16xi32>
    %o1 = vc4kernel.fragment_alu.add %b1, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o1, %v1, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b2 = vc4kernel.fragment_const {value = dense<128> : vector<16xi32>} : vector<16xi32>
    %o2 = vc4kernel.fragment_alu.add %b2, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o2, %v2, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b3 = vc4kernel.fragment_const {value = dense<192> : vector<16xi32>} : vector<16xi32>
    %o3 = vc4kernel.fragment_alu.add %b3, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o3, %v3, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b4 = vc4kernel.fragment_const {value = dense<256> : vector<16xi32>} : vector<16xi32>
    %o4 = vc4kernel.fragment_alu.add %b4, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o4, %v4, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b5 = vc4kernel.fragment_const {value = dense<320> : vector<16xi32>} : vector<16xi32>
    %o5 = vc4kernel.fragment_alu.add %b5, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o5, %v5, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b6 = vc4kernel.fragment_const {value = dense<384> : vector<16xi32>} : vector<16xi32>
    %o6 = vc4kernel.fragment_alu.add %b6, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o6, %v6, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b7 = vc4kernel.fragment_const {value = dense<448> : vector<16xi32>} : vector<16xi32>
    %o7 = vc4kernel.fragment_alu.add %b7, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o7, %v7, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b8 = vc4kernel.fragment_const {value = dense<512> : vector<16xi32>} : vector<16xi32>
    %o8 = vc4kernel.fragment_alu.add %b8, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o8, %v8, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b9 = vc4kernel.fragment_const {value = dense<576> : vector<16xi32>} : vector<16xi32>
    %o9 = vc4kernel.fragment_alu.add %b9, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o9, %v9, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    %b10 = vc4kernel.fragment_const {value = dense<640> : vector<16xi32>} : vector<16xi32>
    %o10 = vc4kernel.fragment_alu.add %b10, %lane_bytes {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %o10, %v10, %full {memory_path = #vc4kernel.memory_path<vdw_global_store>, coherency = #vc4kernel.coherency<dma_ordered>} : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
