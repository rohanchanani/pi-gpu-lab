func.func @mixed_value_i32_f32_dual_path_tail_vc4value(
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
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %lanes = vector.step : vector<16xindex>
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero_i = arith.constant 0 : i32
  %zero_f = arith.constant 0.000000e+00 : f32
  %xiv = vector.transfer_read %xi[%base], %zero_i, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %xfv = vector.transfer_read %xf[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %yfv = vector.transfer_read %yf[%base], %zero_f, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %thresholdv = vector.broadcast %threshold_i : i32 to vector<16xi32>
  %av = vector.broadcast %a : f32 to vector<16xf32>
  %cond = arith.cmpi sgt, %xiv, %thresholdv : vector<16xi32>
  %scaled = arith.mulf %av, %xfv : vector<16xf32>
  %candidate = arith.addf %scaled, %yfv : vector<16xf32>
  %selected = arith.select %cond, %candidate, %yfv : vector<16xi1>, vector<16xf32>
  vector.transfer_write %selected, %out[%base], %mask : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}
