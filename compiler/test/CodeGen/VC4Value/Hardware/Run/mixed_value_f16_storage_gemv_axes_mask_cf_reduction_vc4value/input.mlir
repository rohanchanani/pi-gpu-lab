func.func @mixed_value_f16_storage_gemv_axes_mask_cf_reduction_vc4value(
    %rank_i: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "rank_i",
                                                                            vc4value.direction = "in",
                                                                            vc4value.shape_args = ["rows", "cols"],
                                                                            vc4value.stride_args = ["row_stride"]},
    %rank_f16: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "rank_f16",
                                                                              vc4value.direction = "in",
                                                                              vc4value.shape_args = ["rows", "cols"],
                                                                              vc4value.stride_args = ["row_stride"]},
    %x_f16: memref<?xf16, #vc4value.global> {vc4value.arg_name = "x_f16",
                                             vc4value.direction = "in",
                                             vc4value.shape_args = ["cols"]},
    %partial_out_f: memref<?xf32, #vc4value.global> {vc4value.arg_name = "partial_out_f",
                                                      vc4value.direction = "inout",
                                                      vc4value.shape_args = ["partial_count"]},
    %tail_out_i: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "tail_out_i",
                                                                                vc4value.direction = "inout",
                                                                                vc4value.shape_args = ["rows", "cols"],
                                                                                vc4value.stride_args = ["row_stride"]},
    %tail_out_f16: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "tail_out_f16",
                                                                                  vc4value.direction = "inout",
                                                                                  vc4value.shape_args = ["rows", "cols"],
                                                                                  vc4value.stride_args = ["row_stride"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"},
    %num_kblocks: index {vc4value.arg_name = "num_kblocks", vc4value.scalar_role = "extent"},
    %partial_count: index {vc4value.arg_name = "partial_count", vc4value.scalar_role = "extent"},
    %threshold_i: i32 {vc4value.arg_name = "threshold_i"},
    %threshold_f: f32 {vc4value.arg_name = "threshold_f"},
    %trip: index {vc4value.arg_name = "trip"},
    %use_i32_path: i32 {vc4value.arg_name = "use_i32_path"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree",
                vc4value.f16_storage_policy = "finite"} {
  %pid0 = vc4value.program_id {axis = 0 : i32} : index
  %pid1 = vc4value.program_id {axis = 1 : i32} : index
  %num0 = vc4value.num_programs {axis = 0 : i32} : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %zero_i32 = arith.constant 0 : i32
  %zero_f = arith.constant 0.000000e+00 : f32
  %zero_h = arith.constant 0.000000e+00 : f16
  %one_i32 = arith.constant 1 : i32
  %one_f = arith.constant 1.000000e+00 : f32
  %row_base = arith.muli %pid1, %num0 : index
  %block_id = arith.addi %row_base, %pid0 : index
  %dim_rows = memref.dim %rank_i, %c0 : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  %dim_cols = memref.dim %rank_i, %c1 : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  %inside_row = arith.cmpi ult, %pid1, %dim_rows : index
  cf.cond_br %inside_row, ^load, ^exit

^load:
  %col = arith.muli %pid0, %c16 : index
  %remaining = arith.subi %dim_cols, %col : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %rank_tail_i = vector.transfer_read %rank_i[%pid1, %col], %zero_i32, %mask {in_bounds = [true]} : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xi32>
  %rank_tail_h = vector.transfer_read %rank_f16[%pid1, %col], %zero_h, %mask {in_bounds = [true]} : memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf16>
  %x_tail_h = vector.transfer_read %x_f16[%col], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
  %rank_tail_f = arith.extf %rank_tail_h : vector<16xf16> to vector<16xf32>
  %x_tail = arith.extf %x_tail_h : vector<16xf16> to vector<16xf32>
  %threshold_iv = vector.broadcast %threshold_i : i32 to vector<16xi32>
  %threshold_fv = vector.broadcast %threshold_f : f32 to vector<16xf32>
  %one_iv = vector.broadcast %one_i32 : i32 to vector<16xi32>
  %one_fv = vector.broadcast %one_f : f32 to vector<16xf32>
  %two_iv = arith.addi %one_iv, %one_iv : vector<16xi32>
  %two_fv = arith.addf %one_fv, %one_fv : vector<16xf32>
  %prod = arith.mulf %rank_tail_f, %x_tail : vector<16xf32>
  %has_trip = arith.cmpi ugt, %trip, %c0 : index
  cf.cond_br %has_trip, ^trip_bias(%two_iv, %two_fv : vector<16xi32>, vector<16xf32>), ^after_trip_bias(%one_iv, %one_fv : vector<16xi32>, vector<16xf32>)

^trip_bias(%trip_i_bias: vector<16xi32>, %trip_f_bias: vector<16xf32>):
  cf.br ^after_trip_bias(%trip_i_bias, %trip_f_bias : vector<16xi32>, vector<16xf32>)

^after_trip_bias(%i_bias: vector<16xi32>, %f_bias: vector<16xf32>):
  %use_i32 = arith.cmpi ne, %use_i32_path, %zero_i32 : i32
  cf.cond_br %use_i32, ^i32_path(%i_bias, %f_bias : vector<16xi32>, vector<16xf32>), ^f32_path(%i_bias, %f_bias : vector<16xi32>, vector<16xf32>)

^i32_path(%i_bias_a: vector<16xi32>, %f_bias_a: vector<16xf32>):
  %sum_i = arith.addi %rank_tail_i, %i_bias_a : vector<16xi32>
  %cond_i = arith.cmpi sgt, %sum_i, %threshold_iv : vector<16xi32>
  %selected_i = arith.select %cond_i, %sum_i, %rank_tail_i : vector<16xi1>, vector<16xi32>
  %candidate_f = arith.addf %rank_tail_f, %prod : vector<16xf32>
  %biased_f = arith.addf %candidate_f, %f_bias_a : vector<16xf32>
  %selected_f = arith.select %cond_i, %biased_f, %rank_tail_f : vector<16xi1>, vector<16xf32>
  cf.br ^store(%selected_i, %selected_f : vector<16xi32>, vector<16xf32>)

^f32_path(%i_bias_b: vector<16xi32>, %f_bias_b: vector<16xf32>):
  %cond_f = arith.cmpf olt, %rank_tail_f, %threshold_fv : vector<16xf32>
  %candidate_i = arith.addi %rank_tail_i, %i_bias_b : vector<16xi32>
  %candidate_f_alt = arith.addf %prod, %f_bias_b : vector<16xf32>
  %selected_i_alt = arith.select %cond_f, %candidate_i, %rank_tail_i : vector<16xi1>, vector<16xi32>
  %selected_f_alt = arith.select %cond_f, %candidate_f_alt, %rank_tail_f : vector<16xi1>, vector<16xf32>
  cf.br ^store(%selected_i_alt, %selected_f_alt : vector<16xi32>, vector<16xf32>)

^store(%store_i: vector<16xi32>, %store_f: vector<16xf32>):
  vector.transfer_write %store_i, %tail_out_i[%pid1, %col], %mask {in_bounds = [true]} : vector<16xi32>, memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  %store_h = arith.truncf %store_f : vector<16xf32> to vector<16xf16>
  vector.transfer_write %store_h, %tail_out_f16[%pid1, %col], %mask {in_bounds = [true]} : vector<16xf16>, memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>
  %dot = vector.reduction <add>, %prod : vector<16xf32> into f32
  %partial_row_base = arith.muli %pid1, %num_kblocks : index
  %partial_index = arith.addi %partial_row_base, %pid0 : index
  memref.store %dot, %partial_out_f[%partial_index] : memref<?xf32, #vc4value.global>
  cf.br ^exit

^exit:
  return
}
