module {
  vc4kernel.kernel @program_id_3d_num_programs_vc4kernel(%out : i32) attributes {
    public_name = "program_id_3d_num_programs_vc4kernel",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c2 = arith.constant 2 : i32
    %c4 = arith.constant 4 : i32
    %c6 = arith.constant 6 : i32
    %c10 = arith.constant 10 : i32
    %c100 = arith.constant 100 : i32
    %c1000 = arith.constant 1000 : i32
    %c10000 = arith.constant 10000 : i32
    %c100000 = arith.constant 100000 : i32
    %tag = arith.constant 1895825408 : i32
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %pid_x = vc4kernel.program_id {axis = 0 : i32} : i32
    %pid_y = vc4kernel.program_id {axis = 1 : i32} : i32
    %pid_z = vc4kernel.program_id {axis = 2 : i32} : i32
    %num_x = vc4kernel.num_programs {axis = 0 : i32} : i32
    %num_y = vc4kernel.num_programs {axis = 1 : i32} : i32
    %num_z = vc4kernel.num_programs {axis = 2 : i32} : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>

    %z_rows = arith.muli %pid_z, %num_y : i32
    %zy_rows = arith.addi %z_rows, %pid_y : i32
    %zyx_rows = arith.muli %zy_rows, %num_x : i32
    %request = arith.addi %zyx_rows, %pid_x : i32
    %request_bytes = arith.shli %request, %c6 : i32
    %lane_bytes = vc4kernel.fragment_shl %lanes, %c2 : vector<16xi32>, i32 -> vector<16xi32>
    %request_bytes_v = vc4kernel.splat %request_bytes : i32 -> vector<16xi32>
    %byte_offsets = vc4kernel.fragment_add %request_bytes_v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>

    %pid_y_scaled = arith.muli %pid_y, %c10 : i32
    %pid_z_scaled = arith.muli %pid_z, %c100 : i32
    %num_x_scaled = arith.muli %num_x, %c1000 : i32
    %num_y_scaled = arith.muli %num_y, %c10000 : i32
    %num_z_scaled = arith.muli %num_z, %c100000 : i32
    %v0 = arith.addi %tag, %pid_x : i32
    %v1 = arith.addi %v0, %pid_y_scaled : i32
    %v2 = arith.addi %v1, %pid_z_scaled : i32
    %v3 = arith.addi %v2, %num_x_scaled : i32
    %v4 = arith.addi %v3, %num_y_scaled : i32
    %base_scalar = arith.addi %v4, %num_z_scaled : i32
    %base = vc4kernel.splat %base_scalar : i32 -> vector<16xi32>
    %values = vc4kernel.fragment_add %base, %lanes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %values, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
