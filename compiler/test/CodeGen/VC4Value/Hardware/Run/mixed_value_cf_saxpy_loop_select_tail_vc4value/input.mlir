func.func @mixed_value_cf_saxpy_loop_select_tail_vc4value(
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["n"]},
    %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["n"]},
    %fallback: memref<?xf32, #vc4value.global> {vc4value.arg_name = "fallback",
                                                 vc4value.direction = "in",
                                                 vc4value.shape_args = ["n"]},
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %a: f32 {vc4value.arg_name = "a"},
    %bias: f32 {vc4value.arg_name = "bias"},
    %threshold: f32 {vc4value.arg_name = "threshold"},
    %trip: index {vc4value.arg_name = "trip"},
    %use_select: i32 {vc4value.arg_name = "use_select"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %lanes = vector.step : vector<16xindex>
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero_f = arith.constant 0.000000e+00 : f32
  %zero_i = arith.constant 0 : index
  %one_i = arith.constant 1 : index
  %zero_flag = arith.constant 0 : i32
  %xv = vector.transfer_read %x[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %yv = vector.transfer_read %y[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %fallbackv = vector.transfer_read %fallback[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %av = vector.broadcast %a : f32 to vector<16xf32>
  %biasv = vector.broadcast %bias : f32 to vector<16xf32>
  %thresholdv = vector.broadcast %threshold : f32 to vector<16xf32>
  cf.br ^loop(%zero_i, %yv : index, vector<16xf32>)

^loop(%i: index, %acc: vector<16xf32>):
  %more = arith.cmpi ult, %i, %trip : index
  cf.cond_br %more, ^body(%i, %acc : index, vector<16xf32>), ^after_loop(%acc : vector<16xf32>)

^body(%body_i: index, %body_acc: vector<16xf32>):
  %scaled = arith.mulf %av, %xv : vector<16xf32>
  %next_acc = arith.addf %body_acc, %scaled : vector<16xf32>
  %next_i = arith.addi %body_i, %one_i : index
  cf.br ^loop(%next_i, %next_acc : index, vector<16xf32>)

^after_loop(%looped: vector<16xf32>):
  %take_select = arith.cmpi ne, %use_select, %zero_flag : i32
  cf.cond_br %take_select, ^select_path(%looped : vector<16xf32>), ^bias_path(%looped : vector<16xf32>)

^select_path(%select_in: vector<16xf32>):
  %cmp = arith.cmpf ogt, %select_in, %thresholdv : vector<16xf32>
  %selected = arith.select %cmp, %select_in, %fallbackv : vector<16xi1>, vector<16xf32>
  cf.br ^merge(%selected : vector<16xf32>)

^bias_path(%bias_in: vector<16xf32>):
  %biased = arith.addf %bias_in, %biasv : vector<16xf32>
  cf.br ^merge(%biased : vector<16xf32>)

^merge(%result: vector<16xf32>):
  vector.transfer_write %result, %out[%base], %mask : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}
