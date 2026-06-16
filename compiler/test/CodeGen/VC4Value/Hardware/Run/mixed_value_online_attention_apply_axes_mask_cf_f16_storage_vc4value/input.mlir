func.func @mixed_value_online_attention_apply_axes_mask_cf_f16_storage_vc4value(
    %scores: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "scores", vc4value.direction = "in", vc4value.shape_args = ["queries", "kdim"], vc4value.stride_args = ["lds"]},
    %vt: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "vt", vc4value.direction = "in", vc4value.shape_args = ["dims", "kdim"], vc4value.stride_args = ["ldv"]},
    %residual: memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "residual", vc4value.direction = "in", vc4value.shape_args = ["dims", "kdim"], vc4value.stride_args = ["ldr"]},
    %gate: memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "gate", vc4value.direction = "in", vc4value.shape_args = ["dims", "kdim"], vc4value.stride_args = ["ldg"]},
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["out_n"]},
    %audit: memref<?xf32, #vc4value.global> {vc4value.arg_name = "audit", vc4value.direction = "out", vc4value.shape_args = ["out_n"]},
    %k: index {vc4value.arg_name = "k", vc4value.scalar_role = "extent"},
    %queries: index {vc4value.arg_name = "queries", vc4value.scalar_role = "extent"},
    %kdim: index {vc4value.arg_name = "kdim", vc4value.scalar_role = "extent"},
    %dims: index {vc4value.arg_name = "dims", vc4value.scalar_role = "extent"},
    %out_n: index {vc4value.arg_name = "out_n", vc4value.scalar_role = "extent"},
    %lds: index {vc4value.arg_name = "lds", vc4value.scalar_role = "stride"},
    %ldv: index {vc4value.arg_name = "ldv", vc4value.scalar_role = "stride"},
    %ldr: index {vc4value.arg_name = "ldr", vc4value.scalar_role = "stride"},
    %ldg: index {vc4value.arg_name = "ldg", vc4value.scalar_role = "stride"},
    %ldo: index {vc4value.arg_name = "ldo", vc4value.scalar_role = "stride"},
    %scale: f32 {vc4value.arg_name = "scale", vc4value.scalar_role = "value"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree",
                vc4value.max_policy = "finite",
                vc4value.f16_storage_policy = "finite"} {
  %q = vc4value.program_id {axis = 0 : i32} : index
  %d = vc4value.program_id {axis = 1 : i32} : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %obase = arith.muli %q, %ldo : index
  %oidx = arith.addi %obase, %d : index
  %zero_h = arith.constant 0.000000e+00 : f16
  %zero_f = arith.constant 0.000000e+00 : f32
  %zero_i = arith.constant 0 : i32
  %zero_iv = arith.constant dense<0> : vector<16xi32>
  %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
  %zeros = arith.constant dense<0.000000e+00> : vector<16xf32>
  %m0 = arith.constant -8.000000e+01 : f32
  %l0 = arith.constant 0.000000e+00 : f32
  %acc0 = arith.constant 0.000000e+00 : f32
  %scale_v = vector.broadcast %scale : f32 to vector<16xf32>
  %state:3 = scf.for %start = %c0 to %k step %c16
      iter_args(%m_iter = %m0, %l_iter = %l0, %acc_iter = %acc0)
      -> (f32, f32, f32) {
    %remaining = arith.subi %k, %start : index
    %mask = vector.create_mask %remaining : vector<16xi1>
    %scores_h = vector.transfer_read %scores[%q, %start], %zero_h, %mask {in_bounds = [true]} : memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf16>
    %scores_f = arith.extf %scores_h : vector<16xf16> to vector<16xf32>
    %scores_v = arith.mulf %scores_f, %scale_v : vector<16xf32>
    %active_scores = arith.select %mask, %scores_v, %low : vector<16xi1>, vector<16xf32>
    %block_m = vector.reduction <maxnumf>, %active_scores : vector<16xf32> into f32
    %m_new = arith.maxnumf %m_iter, %block_m : f32
    %alpha_arg = arith.subf %m_iter, %m_new : f32
    %beta_arg = arith.subf %block_m, %m_new : f32
    %alpha = math.exp %alpha_arg : f32
    %beta = math.exp %beta_arg : f32
    %block_mv = vector.broadcast %block_m : f32 to vector<16xf32>
    %shifted = arith.subf %active_scores, %block_mv : vector<16xf32>
    %e = math.exp %shifted : vector<16xf32>
    %active_e = arith.select %mask, %e, %zeros : vector<16xi1>, vector<16xf32>
    %block_l = vector.reduction <add>, %active_e : vector<16xf32> into f32
    %vh = vector.transfer_read %vt[%d, %start], %zero_h, %mask {in_bounds = [true]} : memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf16>
    %v = arith.extf %vh : vector<16xf16> to vector<16xf32>
    %res = vector.transfer_read %residual[%d, %start], %zero_f, %mask {in_bounds = [true]} : memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf32>
    %gate_v = vector.transfer_read %gate[%d, %start], %zero_i, %mask {in_bounds = [true]} : memref<?x?xi32, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xi32>
    %gate_active = arith.cmpi sgt, %gate_v, %zero_iv : vector<16xi32>
    %res_scaled = arith.mulf %res, %scale_v : vector<16xf32>
    %v_plus = arith.addf %v, %res_scaled : vector<16xf32>
    %v_minus = arith.subf %v, %res : vector<16xf32>
    %effective_v = arith.select %gate_active, %v_plus, %v_minus : vector<16xi1>, vector<16xf32>
    %weighted = arith.mulf %active_e, %effective_v : vector<16xf32>
    %block_acc = vector.reduction <add>, %weighted : vector<16xf32> into f32
    %l_scaled = arith.mulf %l_iter, %alpha : f32
    %block_l_scaled = arith.mulf %block_l, %beta : f32
    %l_next = arith.addf %l_scaled, %block_l_scaled : f32
    %acc_scaled = arith.mulf %acc_iter, %alpha : f32
    %block_acc_scaled = arith.mulf %block_acc, %beta : f32
    %acc_next = arith.addf %acc_scaled, %block_acc_scaled : f32
    scf.yield %m_new, %l_next, %acc_next : f32, f32, f32
  }
  %outv = arith.divf %state#2, %state#1 : f32
  memref.store %outv, %out[%oidx] : memref<?xf32, #vc4value.global>
  memref.store %state#1, %audit[%oidx] : memref<?xf32, #vc4value.global>
  return
}
