func.func @mixed_value_cf_completeness_loop_branch_tail_vc4value(
    %x: memref<?xi32, #vc4value.global> {vc4value.arg_name = "x",
                                           vc4value.direction = "in",
                                           vc4value.shape_args = ["n"]},
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                             vc4value.direction = "out",
                                             vc4value.shape_args = ["n"]},
    %start: index {vc4value.arg_name = "start"},
    %end: index {vc4value.arg_name = "end"},
    %step: index {vc4value.arg_name = "step"},
    %trip: index {vc4value.arg_name = "trip"},
    %early: index {vc4value.arg_name = "early"},
    %use_alt: i32 {vc4value.arg_name = "use_alt"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %zero_i32 = arith.constant 0 : i32
  %one_i32 = arith.constant 1 : i32
  %done_bias = arith.constant 100 : i32
  %early_bias = arith.constant 200 : i32
  %base = arith.muli %pid, %c16 : index
  %lanes = vector.step : vector<16xindex>
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %xv = vector.transfer_read %x[%base], %zero_i32, %mask : memref<?xi32, #vc4value.global>, vector<16xi32>
  %use_alt_cond = arith.cmpi ne, %use_alt, %zero_i32 : i32

  %range_sum = scf.for %tile = %start to %end step %step iter_args(%range_acc = %zero_i32) -> (i32) {
    %tile_i32 = arith.index_cast %tile : index to i32
    %tile_term = arith.addi %tile_i32, %one_i32 : i32
    %term = scf.if %use_alt_cond -> (i32) {
      scf.yield %tile_term : i32
    } else {
      scf.yield %one_i32 : i32
    }
    %next_range_acc = arith.addi %range_acc, %term : i32
    scf.yield %next_range_acc : i32
  }
  %rangev = vector.broadcast %range_sum : i32 to vector<16xi32>
  %seed_with_range = arith.addi %xv, %rangev : vector<16xi32>

  %while_result:2 = scf.while (%i = %c0, %acc = %seed_with_range) : (index, vector<16xi32>) -> (index, vector<16xi32>) {
    %keep_going = arith.cmpi ult, %i, %trip : index
    scf.condition(%keep_going) %i, %acc : index, vector<16xi32>
  } do {
  ^bb0(%body_i: index, %body_acc: vector<16xi32>):
    %body_i_i32 = arith.index_cast %body_i : index to i32
    %term = arith.addi %body_i_i32, %one_i32 : i32
    %termv = vector.broadcast %term : i32 to vector<16xi32>
    %next_acc = arith.addi %body_acc, %termv : vector<16xi32>
    %next_i = arith.addi %body_i, %c1 : index
    scf.yield %next_i, %next_acc : index, vector<16xi32>
  }

  cf.br ^loop(%c0, %zero_i32 : index, i32)

^loop(%loop_i: index, %loop_acc: i32):
  %done = arith.cmpi uge, %loop_i, %trip : index
  cf.cond_br %done, ^exit_done(%loop_acc : i32), ^check_early(%loop_i, %loop_acc : index, i32)

^check_early(%check_i: index, %check_acc: i32):
  %take_early = arith.cmpi uge, %check_i, %early : index
  cf.cond_br %take_early, ^exit_early(%check_acc : i32), ^body(%check_i, %check_acc : index, i32)

^body(%body_i2: index, %body_acc2: i32):
  %body_i2_i32 = arith.index_cast %body_i2 : index to i32
  %body_term = arith.addi %body_i2_i32, %one_i32 : i32
  %next_loop_acc = arith.addi %body_acc2, %body_term : i32
  %next_loop_i = arith.addi %body_i2, %c1 : index
  cf.br ^loop(%next_loop_i, %next_loop_acc : index, i32)

^exit_done(%done_acc: i32):
  %done_value = arith.addi %done_acc, %done_bias : i32
  cf.br ^merge(%done_value : i32)

^exit_early(%early_acc: i32):
  %early_value = arith.addi %early_acc, %early_bias : i32
  cf.br ^merge(%early_value : i32)

^merge(%multi_exit_value: i32):
  %multi_exit_v = vector.broadcast %multi_exit_value : i32 to vector<16xi32>
  %final = arith.addi %while_result#1, %multi_exit_v : vector<16xi32>
  vector.transfer_write %final, %out[%base], %mask : vector<16xi32>, memref<?xi32, #vc4value.global>
  return
}
