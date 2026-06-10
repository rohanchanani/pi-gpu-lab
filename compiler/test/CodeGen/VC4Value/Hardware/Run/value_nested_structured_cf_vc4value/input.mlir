func.func @value_nested_structured_cf_vc4value(
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %trip: index {vc4value.arg_name = "trip"},
    %selector: i32 {vc4value.arg_name = "selector"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero_i32 = arith.constant 0 : i32
  %three_i32 = arith.constant 3 : i32
  %use_nested = arith.cmpi ne, %selector, %zero_i32 : i32
  %final = scf.for %i = %c0 to %trip step %c1 iter_args(%acc = %zero_i32) -> (i32) {
    %next = scf.if %use_nested -> (i32) {
      %limit = arith.addi %i, %c1 : index
      %inner:2 = scf.while (%j = %c0, %inner_acc = %acc) : (index, i32) -> (index, i32) {
        %keep_inner = arith.cmpi ult, %j, %limit : index
        scf.condition(%keep_inner) %j, %inner_acc : index, i32
      } do {
      ^bb0(%body_j: index, %body_acc: i32):
        %i_i32 = arith.index_cast %i : index to i32
        %j_i32 = arith.index_cast %body_j : index to i32
        %term0 = arith.addi %i_i32, %j_i32 : i32
        %one_i32 = arith.constant 1 : i32
        %term = arith.addi %term0, %one_i32 : i32
        %next_acc = arith.addi %body_acc, %term : i32
        %next_j = arith.addi %body_j, %c1 : index
        scf.yield %next_j, %next_acc : index, i32
      }
      scf.yield %inner#1 : i32
    } else {
      %fallback = arith.addi %acc, %three_i32 : i32
      scf.yield %fallback : i32
    }
    scf.yield %next : i32
  }
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %finalv = vector.broadcast %final : i32 to vector<16xi32>
  vector.transfer_write %finalv, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
