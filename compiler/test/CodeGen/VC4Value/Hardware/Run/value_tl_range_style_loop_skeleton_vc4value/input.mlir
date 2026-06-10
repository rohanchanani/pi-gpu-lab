func.func @value_tl_range_style_loop_skeleton_vc4value(
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "out",
                                            vc4value.shape_args = ["n"]},
    %start: index {vc4value.arg_name = "start"},
    %end: index {vc4value.arg_name = "end"},
    %step: index {vc4value.arg_name = "step"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %zero_i32 = arith.constant 0 : i32
  %final = scf.for %tile = %start to %end step %step iter_args(%acc = %zero_i32) -> (i32) {
    %tile_i32 = arith.index_cast %tile : index to i32
    %one_i32 = arith.constant 1 : i32
    %term = arith.addi %tile_i32, %one_i32 : i32
    %next_acc = arith.addi %acc, %term : i32
    scf.yield %next_acc : i32
  }
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %finalv = vector.broadcast %final : i32 to vector<16xi32>
  vector.transfer_write %finalv, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
