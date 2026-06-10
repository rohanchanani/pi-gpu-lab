func.func @value_reduce_scalar_store_guard_vc4value(
    %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out",
                                            vc4value.direction = "inout",
                                            vc4value.shape_args = ["blocks"]},
    %value: i32 {vc4value.arg_name = "value"},
    %blocks: index {vc4value.arg_name = "blocks", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %active = arith.cmpi ult, %pid, %blocks : index
  cf.cond_br %active, ^store, ^exit
^store:
  memref.store %value, %out[%pid] : memref<?xi32, #vc4value.global>
  cf.br ^exit
^exit:
  return
}
