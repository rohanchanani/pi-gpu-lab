func.func @value_reduce_max_f32_b16_vc4value(
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["blocks"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
    %blocks: index {vc4value.arg_name = "blocks", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree",
                vc4value.max_policy = "finite"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %active = arith.cmpi ult, %pid, %blocks : index
  cf.cond_br %active, ^reduce, ^exit
^reduce:
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %x[%base], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
  %active_v = arith.select %mask, %v, %low : vector<16xi1>, vector<16xf32>
  %max = vector.reduction <maxnumf>, %active_v : vector<16xf32> into f32
  memref.store %max, %out[%pid] : memref<?xf32, #vc4value.global>
  cf.br ^exit
^exit:
  return
}
