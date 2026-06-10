func.func @value_scf_while_scalar_loop_vc4value(
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %trip: index {vc4value.arg_name = "trip"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero_i32 = arith.constant 0 : i32
  %result:2 = scf.while (%i = %c0, %acc = %zero_i32) : (index, i32) -> (index, i32) {
    %keep_going = arith.cmpi ult, %i, %trip : index
    scf.condition(%keep_going) %i, %acc : index, i32
  } do {
  ^bb0(%body_i: index, %body_acc: i32):
    %body_i_i32 = arith.index_cast %body_i : index to i32
    %one_i32 = arith.constant 1 : i32
    %term = arith.addi %body_i_i32, %one_i32 : i32
    %next_acc = arith.addi %body_acc, %term : i32
    %next_i = arith.addi %body_i, %c1 : index
    scf.yield %next_i, %next_acc : index, i32
  }
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %accv = vector.broadcast %result#1 : i32 to vector<16xi32>
  vector.transfer_write %accv, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
