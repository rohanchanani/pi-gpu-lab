func.func @value_cf_loop_vector_carry_tail_vc4value(
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %trip: index {vc4value.arg_name = "trip"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %lanes = vector.step : vector<16xindex>
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %init = arith.constant dense<0> : vector<16xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  cf.br ^loop(%c0, %init : index, vector<16xi32>)

^loop(%i: index, %acc: vector<16xi32>):
  %keep_going = arith.cmpi ult, %i, %trip : index
  cf.cond_br %keep_going, ^body(%i, %acc : index, vector<16xi32>), ^exit(%acc : vector<16xi32>)

^body(%body_i: index, %body_acc: vector<16xi32>):
  %body_i_i32 = arith.index_cast %body_i : index to i32
  %one_i32 = arith.constant 1 : i32
  %inc = arith.addi %body_i_i32, %one_i32 : i32
  %incv = vector.broadcast %inc : i32 to vector<16xi32>
  %next_acc = arith.addi %body_acc, %incv : vector<16xi32>
  %next_i = arith.addi %body_i, %c1 : index
  cf.br ^loop(%next_i, %next_acc : index, vector<16xi32>)

^exit(%final_acc: vector<16xi32>):
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  vector.transfer_write %final_acc, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
