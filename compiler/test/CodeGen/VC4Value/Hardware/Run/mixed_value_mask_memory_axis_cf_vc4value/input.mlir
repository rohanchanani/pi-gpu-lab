func.func @mixed_value_mask_memory_axis_cf_vc4value(
    %xi: memref<?xi32, #vc4value.global> {vc4value.arg_name = "xi",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
    %xf: memref<?xf32, #vc4value.global> {vc4value.arg_name = "xf",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
    %yf: memref<?xf32, #vc4value.global> {vc4value.arg_name = "yf",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
    %tail_out_f: memref<?xf32, #vc4value.global> {vc4value.arg_name = "tail_out_f",
                                                   vc4value.direction = "inout",
                                                   vc4value.shape_args = ["n"]},
    %tail_out_i: memref<?xi32, #vc4value.global> {vc4value.arg_name = "tail_out_i",
                                                   vc4value.direction = "inout",
                                                   vc4value.shape_args = ["n"]},
    %full_out_i: memref<?xi32, #vc4value.global> {vc4value.arg_name = "full_out_i",
                                                   vc4value.direction = "out",
                                                   vc4value.shape_args = ["n"]},
    %policy_out_i: memref<?xi32, #vc4value.global> {vc4value.arg_name = "policy_out_i",
                                                     vc4value.direction = "out",
                                                     vc4value.shape_args = ["n"]},
    %empty_out_i: memref<?xi32, #vc4value.global> {vc4value.arg_name = "empty_out_i",
                                                    vc4value.direction = "inout",
                                                    vc4value.shape_args = ["n"]},
    %threshold_i: i32 {vc4value.arg_name = "threshold_i"},
    %threshold_f: f32 {vc4value.arg_name = "threshold_f"},
    %a: f32 {vc4value.arg_name = "a"},
    %trip: index {vc4value.arg_name = "trip"},
    %use_i32_path: i32 {vc4value.arg_name = "use_i32_path"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.fp_domain = "finite"} {
  %pid0 = vc4value.program_id {axis = 0 : i32} : index
  %pid1 = vc4value.program_id {axis = 1 : i32} : index
  %num0 = vc4value.num_programs {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %row_base = arith.muli %pid1, %num0 : index
  %block_id = arith.addi %row_base, %pid0 : index
  %base = arith.muli %block_id, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %c0_index = arith.constant 0 : index
  %c1_index = arith.constant 1 : index
  %empty = vector.create_mask %c0_index : vector<16xi1>
  %zero_i32 = arith.constant 0 : i32
  %zero_f = arith.constant 0.000000e+00 : f32
  %one_f = arith.constant 1.000000e+00 : f32
  %empty_value = arith.constant dense<12345> : vector<16xi32>
  %tail_xi = vector.transfer_read %xi[%base], %zero_i32, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %full_xi = vector.transfer_read %xi[%base], %zero_i32 : memref<?xi32, #vc4value.global>, vector<16xi32>
  %xfv = vector.transfer_read %xf[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %yfv = vector.transfer_read %yf[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %threshold_iv = vector.broadcast %threshold_i : i32 to vector<16xi32>
  %threshold_fv = vector.broadcast %threshold_f : f32 to vector<16xf32>
  %av = vector.broadcast %a : f32 to vector<16xf32>
  %onev = vector.broadcast %one_f : f32 to vector<16xf32>
  %num0_i32 = arith.index_cast %num0 : index to i32
  %num0v = vector.broadcast %num0_i32 : i32 to vector<16xi32>
  %full_biased = arith.addi %full_xi, %num0v : vector<16xi32>
  vector.transfer_write %full_biased, %full_out_i[%base] : vector<16xi32>, memref<?xi32, #vc4value.global>
  vector.transfer_write %tail_xi, %policy_out_i[%base] : vector<16xi32>, memref<?xi32, #vc4value.global>
  vector.transfer_write %empty_value, %empty_out_i[%base], %empty : vector<16xi32>, memref<?xi32, #vc4value.global>
  cf.br ^loop(%c0_index, %av : index, vector<16xf32>)

^loop(%i: index, %factor: vector<16xf32>):
  %more = arith.cmpi ult, %i, %trip : index
  cf.cond_br %more, ^body(%i, %factor : index, vector<16xf32>), ^after_loop(%factor : vector<16xf32>)

^body(%body_i: index, %body_factor: vector<16xf32>):
  %next_factor = arith.addf %body_factor, %onev : vector<16xf32>
  %next_i = arith.addi %body_i, %c1_index : index
  cf.br ^loop(%next_i, %next_factor : index, vector<16xf32>)

^after_loop(%loop_factor: vector<16xf32>):
  %use_i32 = arith.cmpi ne, %use_i32_path, %zero_i32 : i32
  cf.cond_br %use_i32, ^i32_path(%loop_factor : vector<16xf32>), ^f32_path(%loop_factor : vector<16xf32>)

^i32_path(%factor_i32_path: vector<16xf32>):
  %cond_i = arith.cmpi sgt, %tail_xi, %threshold_iv : vector<16xi32>
  %scaled = arith.mulf %factor_i32_path, %xfv : vector<16xf32>
  %candidate_f = arith.addf %scaled, %yfv : vector<16xf32>
  %selected_f = arith.select %cond_i, %candidate_f, %yfv : vector<16xi1>, vector<16xf32>
  %biased_i = arith.addi %tail_xi, %num0v : vector<16xi32>
  %selected_i = arith.select %cond_i, %biased_i, %tail_xi : vector<16xi1>, vector<16xi32>
  cf.br ^store(%selected_f, %selected_i : vector<16xf32>, vector<16xi32>)

^f32_path(%factor_f32_path: vector<16xf32>):
  %cond_f = arith.cmpf olt, %xfv, %threshold_fv : vector<16xf32>
  %scaled_alt = arith.mulf %factor_f32_path, %yfv : vector<16xf32>
  %candidate_alt = arith.addf %scaled_alt, %xfv : vector<16xf32>
  %selected_f_alt = arith.select %cond_f, %candidate_alt, %xfv : vector<16xi1>, vector<16xf32>
  %selected_i_alt = arith.addi %tail_xi, %num0v : vector<16xi32>
  cf.br ^store(%selected_f_alt, %selected_i_alt : vector<16xf32>, vector<16xi32>)

^store(%store_f: vector<16xf32>, %store_i: vector<16xi32>):
  vector.transfer_write %store_f, %tail_out_f[%base], %mask : vector<16xf32>, memref<?xf32, #vc4value.global>
  vector.transfer_write %store_i, %tail_out_i[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  cf.br ^exit

^exit:
  return
}
