func.func @mixed_value_strided_ranked_memory_axis_mask_cf_vc4value(
    %flat_i: memref<?xi32, #vc4value.global> {vc4value.arg_name = "flat_i",
                                               vc4value.direction = "in",
                                               vc4value.shape_args = ["total"]},
    %rank_i: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "rank_i",
                                                                            vc4value.direction = "in",
                                                                            vc4value.shape_args = ["rows", "cols"],
                                                                            vc4value.stride_args = ["row_stride"]},
    %rank_f: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "rank_f",
                                                                            vc4value.direction = "in",
                                                                            vc4value.shape_args = ["rows", "cols"],
                                                                            vc4value.stride_args = ["row_stride"]},
    %rank_out_f: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "rank_out_f",
                                                                                vc4value.direction = "inout",
                                                                                vc4value.shape_args = ["rows", "cols"],
                                                                                vc4value.stride_args = ["row_stride"]},
    %rank_out_i: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "rank_out_i",
                                                                                vc4value.direction = "inout",
                                                                                vc4value.shape_args = ["rows", "cols"],
                                                                                vc4value.stride_args = ["row_stride"]},
    %flat_policy_out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "flat_policy_out",
                                                       vc4value.direction = "inout",
                                                       vc4value.shape_args = ["total"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %row_stride: index {vc4value.arg_name = "row_stride", vc4value.scalar_role = "stride"},
    %total: index {vc4value.arg_name = "total", vc4value.scalar_role = "extent"},
    %threshold_i: i32 {vc4value.arg_name = "threshold_i"},
    %threshold_f: f32 {vc4value.arg_name = "threshold_f"},
    %a: f32 {vc4value.arg_name = "a"},
    %trip: index {vc4value.arg_name = "trip"},
    %use_i32_path: i32 {vc4value.arg_name = "use_i32_path"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.fp_domain = "finite"} {
  %pid0 = vc4value.program_id {axis = 0 : i32} : index
  %pid1 = vc4value.program_id {axis = 1 : i32} : index
  %num0 = vc4value.num_programs {axis = 0 : i32} : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %zero_i32 = arith.constant 0 : i32
  %zero_f = arith.constant 0.000000e+00 : f32
  %one_f = arith.constant 1.000000e+00 : f32
  %row_base = arith.muli %pid1, %num0 : index
  %block_id = arith.addi %row_base, %pid0 : index
  %dim_rows = memref.dim %rank_i, %c0 : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  %dim_cols = memref.dim %rank_i, %c1 : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  %inside = arith.cmpi ult, %pid1, %dim_rows : index
  cf.cond_br %inside, ^copy, ^exit

^copy:
  %col = arith.muli %pid0, %c16 : index
  %remaining = arith.subi %dim_cols, %col : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %row_base_idx = arith.muli %pid1, %row_stride : index
  %flat_idx = arith.addi %row_base_idx, %col : index
  %flat_tail = vector.transfer_read %flat_i[%flat_idx], %zero_i32, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %rank_tail_i = vector.transfer_read %rank_i[%pid1, %col], %zero_i32, %mask {in_bounds = [true]} : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xi32>
  %rank_tail_f = vector.transfer_read %rank_f[%pid1, %col], %zero_f, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
  vector.transfer_write %flat_tail, %flat_policy_out[%flat_idx] : vector<16xi32>, memref<?xi32, #vc4value.global>
  %threshold_iv = vector.broadcast %threshold_i : i32 to vector<16xi32>
  %threshold_fv = vector.broadcast %threshold_f : f32 to vector<16xf32>
  %av = vector.broadcast %a : f32 to vector<16xf32>
  %onev = vector.broadcast %one_f : f32 to vector<16xf32>
  %num0_i32 = arith.index_cast %num0 : index to i32
  %num0v = vector.broadcast %num0_i32 : i32 to vector<16xi32>
  cf.br ^loop(%c0, %av : index, vector<16xf32>)

^loop(%i: index, %factor: vector<16xf32>):
  %more = arith.cmpi ult, %i, %trip : index
  cf.cond_br %more, ^body(%i, %factor : index, vector<16xf32>), ^after_loop(%factor : vector<16xf32>)

^body(%body_i: index, %body_factor: vector<16xf32>):
  %next_factor = arith.addf %body_factor, %onev : vector<16xf32>
  %next_i = arith.addi %body_i, %c1 : index
  cf.br ^loop(%next_i, %next_factor : index, vector<16xf32>)

^after_loop(%loop_factor: vector<16xf32>):
  %use_i32 = arith.cmpi ne, %use_i32_path, %zero_i32 : i32
  cf.cond_br %use_i32, ^i32_path(%loop_factor : vector<16xf32>), ^f32_path(%loop_factor : vector<16xf32>)

^i32_path(%factor_i32_path: vector<16xf32>):
  %sum_i = arith.addi %rank_tail_i, %flat_tail : vector<16xi32>
  %cond_i = arith.cmpi sgt, %sum_i, %threshold_iv : vector<16xi32>
  %scaled = arith.mulf %factor_i32_path, %rank_tail_f : vector<16xf32>
  %candidate_f = arith.addf %scaled, %onev : vector<16xf32>
  %selected_f = arith.select %cond_i, %candidate_f, %rank_tail_f : vector<16xi1>, vector<16xf32>
  %biased_i = arith.addi %sum_i, %num0v : vector<16xi32>
  %selected_i = arith.select %cond_i, %biased_i, %flat_tail : vector<16xi1>, vector<16xi32>
  cf.br ^store(%selected_f, %selected_i : vector<16xf32>, vector<16xi32>)

^f32_path(%factor_f32_path: vector<16xf32>):
  %cond_f = arith.cmpf olt, %rank_tail_f, %threshold_fv : vector<16xf32>
  %scaled_alt = arith.mulf %factor_f32_path, %rank_tail_f : vector<16xf32>
  %candidate_alt = arith.addf %scaled_alt, %onev : vector<16xf32>
  %selected_f_alt = arith.select %cond_f, %candidate_alt, %rank_tail_f : vector<16xi1>, vector<16xf32>
  %sum_i_alt = arith.addi %rank_tail_i, %flat_tail : vector<16xi32>
  %selected_i_alt = arith.addi %sum_i_alt, %num0v : vector<16xi32>
  cf.br ^store(%selected_f_alt, %selected_i_alt : vector<16xf32>, vector<16xi32>)

^store(%store_f: vector<16xf32>, %store_i: vector<16xi32>):
  vector.transfer_write %store_f, %rank_out_f[%pid1, %col], %mask {in_bounds = [true]} : vector<16xf32>, memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>
  vector.transfer_write %store_i, %rank_out_i[%pid1, %col], %mask {in_bounds = [true]} : vector<16xi32>, memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>
  cf.br ^exit

^exit:
  return
}
