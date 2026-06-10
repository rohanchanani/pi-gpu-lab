func.func @value_scf_while_vector_carry_vc4value(
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %trip: index {vc4value.arg_name = "trip"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %init = arith.constant dense<0> : vector<16xi32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result:2 = scf.while (%i = %c0, %acc = %init) : (index, vector<16xi32>) -> (index, vector<16xi32>) {
    %keep_going = arith.cmpi ult, %i, %trip : index
    scf.condition(%keep_going) %i, %acc : index, vector<16xi32>
  } do {
  ^bb0(%body_i: index, %body_acc: vector<16xi32>):
    %body_i_i32 = arith.index_cast %body_i : index to i32
    %one_i32 = arith.constant 1 : i32
    %inc = arith.addi %body_i_i32, %one_i32 : i32
    %incv = vector.broadcast %inc : i32 to vector<16xi32>
    %next_acc = arith.addi %body_acc, %incv : vector<16xi32>
    %next_i = arith.addi %body_i, %c1 : index
    scf.yield %next_i, %next_acc : index, vector<16xi32>
  }
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  vector.transfer_write %result#1, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
