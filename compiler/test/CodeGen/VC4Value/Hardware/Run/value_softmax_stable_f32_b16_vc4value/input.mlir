func.func @value_softmax_stable_f32_b16_vc4value(
    %x: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["rows", "ncols"], vc4value.stride_args = ["stride"]},
    %y: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["rows", "ncols"], vc4value.stride_args = ["stride"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %ncols: index {vc4value.arg_name = "ncols", vc4value.scalar_role = "extent"},
    %stride: index {vc4value.arg_name = "stride", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree",
                vc4value.max_policy = "finite",
                vc4value.softmax_v0 = "one_block_active_1_to_16"} {
  %block = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %is_first = arith.cmpi eq, %block, %c0 : index
  cf.cond_br %is_first, ^body, ^done
^body:
  %col = arith.muli %block, %c16 : index
  %mask = vector.create_mask %ncols : vector<16xi1>
  %zero_s = arith.constant 0.000000e+00 : f32
  %xv = vector.transfer_read %x[%row, %col], %zero_s, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
  %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
  %zero = arith.constant dense<0.000000e+00> : vector<16xf32>
  %active_x = arith.select %mask, %xv, %low : vector<16xi1>, vector<16xf32>
  %max = vector.reduction <maxnumf>, %active_x : vector<16xf32> into f32
  %maxv = vector.broadcast %max : f32 to vector<16xf32>
  %shifted = arith.subf %xv, %maxv : vector<16xf32>
  %expv = math.exp %shifted : vector<16xf32>
  %active_e = arith.select %mask, %expv, %zero : vector<16xi1>, vector<16xf32>
  %denom = vector.reduction <add>, %active_e : vector<16xf32> into f32
  %one = arith.constant 1.000000e+00 : f32
  %inv = arith.divf %one, %denom : f32
  %invv = vector.broadcast %inv : f32 to vector<16xf32>
  %outv = arith.mulf %active_e, %invv : vector<16xf32>
  vector.transfer_write %outv, %y[%row, %col], %mask {in_bounds = [true]} : vector<16xf32>, memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>
  cf.br ^done
^done:
  return
}
