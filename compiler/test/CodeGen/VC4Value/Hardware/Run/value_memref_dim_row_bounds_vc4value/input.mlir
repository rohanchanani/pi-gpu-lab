func.func @value_memref_dim_row_bounds_vc4value(
    %in: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "in",
                                                                        vc4value.direction = "in",
                                                                        vc4value.shape_args = ["rows", "cols"],
                                                                        vc4value.stride_args = ["row_stride"]},
    %out: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "out",
                                                                         vc4value.direction = "inout",
                                                                         vc4value.shape_args = ["rows", "cols"],
                                                                         vc4value.stride_args = ["row_stride"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pid_col = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %dim_rows = memref.dim %in, %c0 : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  %dim_cols = memref.dim %in, %c1 : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  %inside = arith.cmpi ult, %row, %dim_rows : index
  cf.cond_br %inside, ^copy, ^done
^copy:
  %c16 = arith.constant 16 : index
  %col = arith.muli %pid_col, %c16 : index
  %remaining = arith.subi %dim_cols, %col : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %values = vector.transfer_read %in[%row, %col], %zero, %mask {in_bounds = [true]} : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xi32>
  vector.transfer_write %values, %out[%row, %col], %mask {in_bounds = [true]} : vector<16xi32>, memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  cf.br ^done
^done:
  return
}
