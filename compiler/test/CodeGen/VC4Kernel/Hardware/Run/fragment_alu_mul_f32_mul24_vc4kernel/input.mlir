module {
  vc4kernel.kernel @fragment_alu_mul_f32_mul24_vc4kernel(%fa : i32, %fb : i32, %ia : i32, %ib : i32, %out_f32 : i32, %out_i32 : i32) attributes {
    public_name = "fragment_alu_mul_f32_mul24_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "fa", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "fb", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "ia", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "ib", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out_f32", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "out_i32", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %fav = vc4kernel.tmu_load_fragment %fa, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %fbv = vc4kernel.tmu_load_fragment %fb, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %iav = vc4kernel.tmu_load_fragment %ia, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %ibv = vc4kernel.tmu_load_fragment %ib, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %fm = vc4kernel.fragment_alu.mul %fav, %fbv {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %im = vc4kernel.fragment_alu.mul %iav, %ibv {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out_f32, %lane_bytes, %fm, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out_i32, %lane_bytes, %im, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
