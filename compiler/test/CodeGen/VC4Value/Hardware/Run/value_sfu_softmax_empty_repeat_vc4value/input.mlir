func.func @value_sfu_softmax_empty_repeat_vc4value(
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %den: memref<?xf32, #vc4value.global> {vc4value.arg_name = "den", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite_nonzero"} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %pid, %c16 : index
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %xv = vector.transfer_read %x[%base], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %dv = vector.transfer_read %den[%base], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %e = math.exp %xv : vector<16xf32>
  %q = arith.divf %e, %dv : vector<16xf32>
  vector.transfer_write %q, %out[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}
