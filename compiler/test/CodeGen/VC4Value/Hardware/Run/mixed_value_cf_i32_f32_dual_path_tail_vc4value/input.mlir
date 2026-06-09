func.func @mixed_value_cf_i32_f32_dual_path_tail_vc4value(
    %xi: memref<?xi32, #vc4value.global> {vc4value.arg_name = "xi",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
    %xf: memref<?xf32, #vc4value.global> {vc4value.arg_name = "xf",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
    %yf: memref<?xf32, #vc4value.global> {vc4value.arg_name = "yf",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %threshold_i: i32 {vc4value.arg_name = "threshold_i"},
    %a: f32 {vc4value.arg_name = "a"},
    %trip: index {vc4value.arg_name = "trip"},
    %use_i32_path: i32 {vc4value.arg_name = "use_i32_path"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %lanes = vector.step : vector<16xindex>
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero_i32 = arith.constant 0 : i32
  %zero_index = arith.constant 0 : index
  %one_index = arith.constant 1 : index
  %zero_f = arith.constant 0.000000e+00 : f32
  %one_f = arith.constant 1.000000e+00 : f32
  %xiv = vector.transfer_read %xi[%base], %zero_i32, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %xfv = vector.transfer_read %xf[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %yfv = vector.transfer_read %yf[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %thresholdv = vector.broadcast %threshold_i : i32 to vector<16xi32>
  %av = vector.broadcast %a : f32 to vector<16xf32>
  %onev = vector.broadcast %one_f : f32 to vector<16xf32>
  cf.br ^loop(%zero_index, %av : index, vector<16xf32>)

^loop(%i: index, %factor: vector<16xf32>):
  %more = arith.cmpi ult, %i, %trip : index
  cf.cond_br %more, ^body(%i, %factor : index, vector<16xf32>), ^after_loop(%factor : vector<16xf32>)

^body(%body_i: index, %body_factor: vector<16xf32>):
  %next_factor = arith.addf %body_factor, %onev : vector<16xf32>
  %next_i = arith.addi %body_i, %one_index : index
  cf.br ^loop(%next_i, %next_factor : index, vector<16xf32>)

^after_loop(%loop_factor: vector<16xf32>):
  %use_i32 = arith.cmpi ne, %use_i32_path, %zero_i32 : i32
  cf.cond_br %use_i32, ^i32_path(%loop_factor : vector<16xf32>), ^float_path(%loop_factor : vector<16xf32>)

^i32_path(%factor_i32_path: vector<16xf32>):
  %cond = arith.cmpi sgt, %xiv, %thresholdv : vector<16xi32>
  %scaled = arith.mulf %factor_i32_path, %xfv : vector<16xf32>
  %candidate = arith.addf %scaled, %yfv : vector<16xf32>
  %selected = arith.select %cond, %candidate, %yfv : vector<16xi1>, vector<16xf32>
  cf.br ^merge(%selected : vector<16xf32>)

^float_path(%factor_float_path: vector<16xf32>):
  %scaled_alt = arith.mulf %factor_float_path, %yfv : vector<16xf32>
  %candidate_alt = arith.addf %scaled_alt, %xfv : vector<16xf32>
  cf.br ^merge(%candidate_alt : vector<16xf32>)

^merge(%result: vector<16xf32>):
  vector.transfer_write %result, %out[%base], %mask : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}
