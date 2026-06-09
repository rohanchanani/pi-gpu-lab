func.func @value_cf_branch_guard_tail_vc4value(
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %has_work = arith.cmpi ult, %base, %n : index
  cf.cond_br %has_work, ^store, ^exit

^store:
  %lanes = vector.step : vector<16xindex>
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %value = arith.constant dense<101> : vector<16xi32>
  vector.transfer_write %value, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  cf.br ^exit

^exit:
  return
}
