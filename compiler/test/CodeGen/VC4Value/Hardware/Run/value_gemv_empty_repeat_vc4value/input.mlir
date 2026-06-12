func.func @value_gemv_empty_repeat_vc4value(
    %a: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "a",
                                                                        vc4value.direction = "in",
                                                                        vc4value.shape_args = ["rows", "k"],
                                                                        vc4value.stride_args = ["lda"]},
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["k"]},
    %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y",
                                          vc4value.direction = "inout",
                                          vc4value.shape_args = ["rows"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %k: index {vc4value.arg_name = "k", vc4value.scalar_role = "extent"},
    %lda: index {vc4value.arg_name = "lda", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %row = vc4value.program_id {axis = 0 : i32} : index
  %c0 = arith.constant 0 : index
  %row_active = arith.cmpi ult, %row, %rows : index
  %nonempty = arith.cmpi ugt, %k, %c0 : index
  cf.cond_br %row_active, ^check_k, ^exit
^check_k:
  cf.cond_br %nonempty, ^dot, ^exit
^dot:
  %mask = vector.create_mask %k : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %av = vector.transfer_read %a[%row, %c0], %zero, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
  %xv = vector.transfer_read %x[%c0], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %prod = arith.mulf %av, %xv : vector<16xf32>
  %dot_result = vector.reduction <add>, %prod : vector<16xf32> into f32
  memref.store %dot_result, %y[%row] : memref<?xf32, #vc4value.global>
  cf.br ^exit
^exit:
  return
}
