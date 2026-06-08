func.func @mixed_value_saxpy_select_tail_vc4value(
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
    %threshold: f32 {vc4value.arg_name = "threshold"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %lanes = vector.step : vector<16xindex>
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %xv = vector.transfer_read %x[%base], %zero, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %yv = vector.transfer_read %y[%base], %zero, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %fallbackv = vector.transfer_read %fallback[%base], %zero, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %av = vector.broadcast %a : f32 to vector<16xf32>
  %thresholdv = vector.broadcast %threshold : f32 to vector<16xf32>
  %scaled = arith.mulf %av, %xv : vector<16xf32>
  %tmp = arith.addf %scaled, %yv : vector<16xf32>
  %cond = arith.cmpf ogt, %tmp, %thresholdv : vector<16xf32>
  %selected = arith.select %cond, %tmp, %fallbackv : vector<16xi1>, vector<16xf32>
  vector.transfer_write %selected, %out[%base], %mask : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}
