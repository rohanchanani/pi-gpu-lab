func.func @value_rank2_row_slice_copy_vc4value(
    %in: memref<?x?xi32, #vc4value.global> {vc4value.arg_name = "in",
                                             vc4value.direction = "in",
                                             vc4value.shape_args = ["rows", "cols"]},
    %out: memref<?x?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                              vc4value.direction = "inout",
                                              vc4value.shape_args = ["rows", "cols"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
  %pid_col = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %c16 = arith.constant 16 : index
  %col = arith.muli %pid_col, %c16 : index
  %remaining = arith.subi %cols, %col : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %values = vector.transfer_read %in[%row, %col], %zero, %mask : memref<?x?xi32, #vc4value.global>, vector<16xi32>
  vector.transfer_write %values, %out[%row, %col], %mask : vector<16xi32>, memref<?x?xi32, #vc4value.global>
  return
}
