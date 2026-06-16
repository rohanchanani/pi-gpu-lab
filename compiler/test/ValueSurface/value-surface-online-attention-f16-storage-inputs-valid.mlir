// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @online_attention_f16_storage_inputs
  func.func @online_attention_f16_storage_inputs(
      %scores: memref<?xf16, #vc4value.global> {vc4value.arg_name = "scores", vc4value.direction = "in", vc4value.shape_args = ["scores_n"]},
      %vt: memref<?xf16, #vc4value.global> {vc4value.arg_name = "vt", vc4value.direction = "in", vc4value.shape_args = ["vt_n"]},
      %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["out_n"]},
      %k: index {vc4value.arg_name = "k", vc4value.scalar_role = "extent"},
      %scores_n: index {vc4value.arg_name = "scores_n", vc4value.scalar_role = "extent"},
      %vt_n: index {vc4value.arg_name = "vt_n", vc4value.scalar_role = "extent"},
      %out_n: index {vc4value.arg_name = "out_n", vc4value.scalar_role = "extent"},
      %lds: index {vc4value.arg_name = "lds", vc4value.scalar_role = "stride"},
      %ldv: index {vc4value.arg_name = "ldv", vc4value.scalar_role = "stride"},
      %ldo: index {vc4value.arg_name = "ldo", vc4value.scalar_role = "stride"})
      attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree",
                  vc4value.max_policy = "finite",
                  vc4value.f16_storage_policy = "finite"} {
    %q = vc4value.program_id {axis = 0 : i32} : index
    %d = vc4value.program_id {axis = 1 : i32} : index
    %srow = arith.muli %q, %lds : index
    %vrow = arith.muli %d, %ldv : index
    %orow = arith.muli %q, %ldo : index
    %oidx = arith.addi %orow, %d : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %zero_h = arith.constant 0.000000e+00 : f16
    %m0 = arith.constant -8.000000e+01 : f32
    %l0 = arith.constant 0.000000e+00 : f32
    %acc0 = arith.constant 0.000000e+00 : f32
    %state:3 = scf.for %start = %c0 to %k step %c16
        iter_args(%m_iter = %m0, %l_iter = %l0, %acc_iter = %acc0)
        -> (f32, f32, f32) {
      %remaining = arith.subi %k, %start : index
      %mask = vector.create_mask %remaining : vector<16xi1>
      %sidx = arith.addi %srow, %start : index
      %scores_h = vector.transfer_read %scores[%sidx], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
      // CHECK: arith.extf
      %scores_v = arith.extf %scores_h : vector<16xf16> to vector<16xf32>
      %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
      %zeros = arith.constant dense<0.000000e+00> : vector<16xf32>
      %zero = arith.constant 0.000000e+00 : f32
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
      %vidx = arith.addi %vrow, %start : index
      %v_h = vector.transfer_read %vt[%vidx], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
      %v = arith.extf %v_h : vector<16xf16> to vector<16xf32>
      %weighted = arith.mulf %active_e, %v : vector<16xf32>
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
    return
  }
}
