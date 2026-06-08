func.func @value_i32_add_select_tail_vc4value(
    %x: memref<?xi32, #vc4value.global> {vc4value.arg_name = "x",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["n"]},
    %y: memref<?xi32, #vc4value.global> {vc4value.arg_name = "y",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["n"]},
    %alt: memref<?xi32, #vc4value.global> {vc4value.arg_name = "alt",
                                            vc4value.direction = "in",
                                            vc4value.shape_args = ["n"]},
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %bias: i32 {vc4value.arg_name = "bias"},
    %threshold: i32 {vc4value.arg_name = "threshold"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %lanes = vector.step : vector<16xindex>
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %xv = vector.transfer_read %x[%base], %zero, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %yv = vector.transfer_read %y[%base], %zero, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %altv = vector.transfer_read %alt[%base], %zero, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %biasv = vector.broadcast %bias : i32 to vector<16xi32>
  %thresholdv = vector.broadcast %threshold : i32 to vector<16xi32>
  %sum = arith.addi %xv, %yv : vector<16xi32>
  %t = arith.subi %sum, %biasv : vector<16xi32>
  %cond = arith.cmpi sgt, %t, %thresholdv : vector<16xi32>
  %selected = arith.select %cond, %t, %altv : vector<16xi1>, vector<16xi32>
  vector.transfer_write %selected, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
