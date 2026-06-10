func.func @value_mask_memory_policy_other_zero_vc4value(
    %x: memref<?xi32, #vc4value.global> {vc4value.arg_name = "x",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["n"]},
    %full_out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "full_out",
                                                 vc4value.direction = "out",
                                                 vc4value.shape_args = ["n"]},
    %tail_out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "tail_out",
                                                 vc4value.direction = "inout",
                                                 vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %values = vector.transfer_read %x[%base], %zero, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  vector.transfer_write %values, %full_out[%base] : vector<16xi32>, memref<?xi32, #vc4value.global>
  vector.transfer_write %values, %tail_out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
