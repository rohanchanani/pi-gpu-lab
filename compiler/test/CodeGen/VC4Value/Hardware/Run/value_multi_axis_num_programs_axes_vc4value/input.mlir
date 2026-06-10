func.func @value_multi_axis_num_programs_axes_vc4value(
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
  %num2 = vc4value.num_programs {axis = 2 : i32} : index
  %c16 = arith.constant 16 : index
  %plane_base = arith.muli %pid2, %num1 : index
  %row = arith.addi %plane_base, %pid1 : index
  %row_base = arith.muli %row, %num0 : index
  %block_id = arith.addi %row_base, %pid0 : index
  %base = arith.muli %block_id, %c16 : index
  %num0_i32 = arith.index_cast %num0 : index to i32
  %num1_i32 = arith.index_cast %num1 : index to i32
  %num2_i32 = arith.index_cast %num2 : index to i32
  %scale_num0 = arith.constant 10000 : i32
  %scale_num1 = arith.constant 100 : i32
  %num0_part = arith.muli %num0_i32, %scale_num0 : i32
  %num1_part = arith.muli %num1_i32, %scale_num1 : i32
  %xy = arith.addi %num0_part, %num1_part : i32
  %encoded_scalar = arith.addi %xy, %num2_i32 : i32
  %encoded = vector.broadcast %encoded_scalar : i32 to vector<16xi32>
  %lanes = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
  %value = arith.addi %encoded, %lanes : vector<16xi32>
  vector.transfer_write %value, %out[%base] : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
