func.func @value_gemv_partial_kblock_f32_vc4value(
    %a: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "a",
                                                                        vc4value.direction = "in",
                                                                        vc4value.shape_args = ["rows", "k_total"],
                                                                        vc4value.stride_args = ["lda"]},
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["k_total"]},
    %partials: memref<?xf32, #vc4value.global> {vc4value.arg_name = "partials",
                                                 vc4value.direction = "inout",
                                                 vc4value.shape_args = ["partial_count"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %k_total: index {vc4value.arg_name = "k_total", vc4value.scalar_role = "extent"},
    %lda: index {vc4value.arg_name = "lda", vc4value.scalar_role = "stride"},
    %num_kblocks: index {vc4value.arg_name = "num_kblocks", vc4value.scalar_role = "extent"},
    %partial_count: index {vc4value.arg_name = "partial_count", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %kblock = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %row_active = arith.cmpi ult, %row, %rows : index
  %kblock_active = arith.cmpi ult, %kblock, %num_kblocks : index
  cf.cond_br %row_active, ^check_kblock, ^exit
^check_kblock:
  cf.cond_br %kblock_active, ^dot, ^exit
^dot:
  %c16 = arith.constant 16 : index
  %kbase = arith.muli %kblock, %c16 : index
  %remaining = arith.subi %k_total, %kbase : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %av = vector.transfer_read %a[%row, %kbase], %zero, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
  %xv = vector.transfer_read %x[%kbase], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %prod = arith.mulf %av, %xv : vector<16xf32>
  %dot_result = vector.reduction <add>, %prod : vector<16xf32> into f32
  %row_base = arith.muli %row, %num_kblocks : index
  %out = arith.addi %row_base, %kblock : index
  memref.store %dot_result, %partials[%out] : memref<?xf32, #vc4value.global>
  cf.br ^exit
^exit:
  return
}
