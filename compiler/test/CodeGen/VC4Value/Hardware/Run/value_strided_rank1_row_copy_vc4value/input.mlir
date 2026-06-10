func.func @value_strided_rank1_row_copy_vc4value(
    %in: memref<?xi32, #vc4value.global> {vc4value.arg_name = "in",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["total"]},
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "inout",
                                            vc4value.shape_args = ["total"]},
    %total: index {vc4value.arg_name = "total", vc4value.scalar_role = "extent"},
    %ncols: index {vc4value.arg_name = "ncols", vc4value.scalar_role = "extent"},
    %lda: index {vc4value.arg_name = "lda", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
  %pid_col = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %c16 = arith.constant 16 : index
  %col = arith.muli %pid_col, %c16 : index
  %row_base = arith.muli %row, %lda : index
  %idx = arith.addi %row_base, %col : index
  %remaining = arith.subi %ncols, %col : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %values = vector.transfer_read %in[%idx], %zero, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  vector.transfer_write %values, %out[%idx], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
