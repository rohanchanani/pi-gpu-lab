func.func @value_multi_axis_pid3d_i32_vc4value(
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 3 : i32} {
  %pid0 = vc4value.program_id {axis = 0 : i32} : index
  %pid1 = vc4value.program_id {axis = 1 : i32} : index
  %pid2 = vc4value.program_id {axis = 2 : i32} : index
  %num0 = vc4value.num_programs {axis = 0 : i32} : index
  %num1 = vc4value.num_programs {axis = 1 : i32} : index
  %c16 = arith.constant 16 : index
  %plane_base = arith.muli %pid2, %num1 : index
  %row = arith.addi %plane_base, %pid1 : index
  %row_base = arith.muli %row, %num0 : index
  %block_id = arith.addi %row_base, %pid0 : index
  %base = arith.muli %block_id, %c16 : index
  %pid0_i32 = arith.index_cast %pid0 : index to i32
  %pid1_i32 = arith.index_cast %pid1 : index to i32
  %pid2_i32 = arith.index_cast %pid2 : index to i32
  %scale_pid2 = arith.constant 10000 : i32
  %scale_pid1 = arith.constant 1000 : i32
  %scale_pid0 = arith.constant 100 : i32
  %pid2_part = arith.muli %pid2_i32, %scale_pid2 : i32
  %pid1_part = arith.muli %pid1_i32, %scale_pid1 : i32
  %pid0_part = arith.muli %pid0_i32, %scale_pid0 : i32
  %plane_row = arith.addi %pid2_part, %pid1_part : i32
  %encoded_scalar = arith.addi %plane_row, %pid0_part : i32
  %encoded = vector.broadcast %encoded_scalar : i32 to vector<16xi32>
  %lanes = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
  %value = arith.addi %encoded, %lanes : vector<16xi32>
  vector.transfer_write %value, %out[%base] : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
