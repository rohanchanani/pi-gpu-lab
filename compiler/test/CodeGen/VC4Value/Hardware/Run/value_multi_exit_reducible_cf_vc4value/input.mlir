func.func @value_multi_exit_reducible_cf_vc4value(
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %trip: index {vc4value.arg_name = "trip"},
    %early: index {vc4value.arg_name = "early"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero_i32 = arith.constant 0 : i32
  %done_bias = arith.constant 100 : i32
  %early_bias = arith.constant 200 : i32
  cf.br ^loop(%c0, %zero_i32 : index, i32)

^loop(%i: index, %acc: i32):
  %done = arith.cmpi uge, %i, %trip : index
  cf.cond_br %done, ^exit_done(%acc : i32), ^check_early(%i, %acc : index, i32)

^check_early(%check_i: index, %check_acc: i32):
  %take_early = arith.cmpi uge, %check_i, %early : index
  cf.cond_br %take_early, ^exit_early(%check_acc : i32), ^body(%check_i, %check_acc : index, i32)

^body(%body_i: index, %body_acc: i32):
  %body_i_i32 = arith.index_cast %body_i : index to i32
  %one_i32 = arith.constant 1 : i32
  %term = arith.addi %body_i_i32, %one_i32 : i32
  %next_acc = arith.addi %body_acc, %term : i32
  %next_i = arith.addi %body_i, %c1 : index
  cf.br ^loop(%next_i, %next_acc : index, i32)

^exit_done(%done_acc: i32):
  %done_value = arith.addi %done_acc, %done_bias : i32
  cf.br ^merge(%done_value : i32)

^exit_early(%early_acc: i32):
  %early_value = arith.addi %early_acc, %early_bias : i32
  cf.br ^merge(%early_value : i32)

^merge(%final: i32):
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %finalv = vector.broadcast %final : i32 to vector<16xi32>
  vector.transfer_write %finalv, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
