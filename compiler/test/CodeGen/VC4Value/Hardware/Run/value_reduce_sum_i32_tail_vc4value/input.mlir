func.func @value_reduce_sum_i32_tail_vc4value(
    %in: memref<?xi32, #vc4value.global> {vc4value.arg_name = "in",
                                          vc4value.direction = "in",
                                          vc4value.shape_args = ["n"]},
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "inout",
                                            vc4value.shape_args = ["blocks"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
    %blocks: index {vc4value.arg_name = "blocks", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %active = arith.cmpi ult, %pid, %blocks : index
  cf.cond_br %active, ^reduce, ^exit
^reduce:
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<?xi32, #vc4value.global>, vector<16xi32>
  %sum = vector.reduction <add>, %v : vector<16xi32> into i32
  memref.store %sum, %out[%pid] : memref<?xi32, #vc4value.global>
  cf.br ^exit
^exit:
  return
}
