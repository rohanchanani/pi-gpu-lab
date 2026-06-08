func.func @value_saxpy_f32_tail_vc4value(
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["n"]},
    %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y",
                                          vc4value.direction = "inout",
                                          vc4value.shape_args = ["n"]},
    %a: f32 {vc4value.arg_name = "a"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %lanes = vector.step : vector<16xindex>
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %xv = vector.transfer_read %x[%base], %zero, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %yv = vector.transfer_read %y[%base], %zero, %mask : memref<?xf32, #vc4value.global>, vector<16xf32>
  %av = vector.broadcast %a : f32 to vector<16xf32>
  %scaled = arith.mulf %av, %xv : vector<16xf32>
  %sum = arith.addf %scaled, %yv : vector<16xf32>
  vector.transfer_write %sum, %y[%base], %mask : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}
