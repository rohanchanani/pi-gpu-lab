module {
  vc4kernel.kernel @fragment_alu_add_convert_vc4kernel(%fin : i32, %iin : i32, %out_i32 : i32, %out_f32 : i32) attributes {
    public_name = "fragment_alu_add_convert_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "fin", kind = "buffer", direction = "in", elem_type = "f32"},
      {name = "iin", kind = "buffer", direction = "in", elem_type = "i32"},
      {name = "out_i32", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "out_f32", kind = "buffer", direction = "out", elem_type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %lane_bytes = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %fv = vc4kernel.tmu_load_fragment %fin, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
    %iv = vc4kernel.tmu_load_fragment %iin, %lane_bytes, %full : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %to_i = vc4kernel.fragment_alu.add %fv {opcode = #vc4kernel.add_alu_opcode<ftoi>} : (vector<16xf32>) -> vector<16xi32>
    %to_f = vc4kernel.fragment_alu.add %iv {opcode = #vc4kernel.add_alu_opcode<itof>} : (vector<16xi32>) -> vector<16xf32>
    vc4kernel.vdw_store_fragment %out_i32, %lane_bytes, %to_i, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out_f32, %lane_bytes, %to_f, %full : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
