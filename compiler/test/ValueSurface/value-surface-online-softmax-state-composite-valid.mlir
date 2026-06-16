// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @online_softmax_state_composite
  func.func @online_softmax_state_composite(
      %scores: memref<?xf32, #vc4value.global> {vc4value.arg_name = "scores", vc4value.direction = "in", vc4value.shape_args = ["scores_n"]},
      %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["out_n"]},
      %k: index {vc4value.arg_name = "k", vc4value.scalar_role = "extent"},
      %scores_n: index {vc4value.arg_name = "scores_n", vc4value.scalar_role = "extent"},
      %out_n: index {vc4value.arg_name = "out_n", vc4value.scalar_role = "extent"},
      %lds: index {vc4value.arg_name = "lds", vc4value.scalar_role = "stride"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree",
                  vc4value.max_policy = "finite"} {
    %q = vc4value.program_id {axis = 0 : i32} : index
    %row = arith.muli %q, %lds : index
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.000000e+00 : f32
    %one = arith.constant 1.000000e+00 : f32
    %m0 = arith.constant -8.000000e+01 : f32
    %l0 = arith.constant 0.000000e+00 : f32
    %acc0 = arith.constant 0.000000e+00 : f32
    // CHECK: scf.for
    %state:3 = scf.for %start = %c0 to %k step %c16
        iter_args(%m_iter = %m0, %l_iter = %l0, %acc_iter = %acc0)
        -> (f32, f32, f32) {
      %remaining = arith.subi %k, %start : index
      %mask = vector.create_mask %remaining : vector<16xi1>
      %sidx = arith.addi %row, %start : index
      %scores_v = vector.transfer_read %scores[%sidx], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
      %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
      %zeros = arith.constant dense<0.000000e+00> : vector<16xf32>
      %active_scores = arith.select %mask, %scores_v, %low : vector<16xi1>, vector<16xf32>
      // CHECK: vector.reduction <maxnumf>
      %block_m = vector.reduction <maxnumf>, %active_scores : vector<16xf32> into f32
      %m_new = arith.maxnumf %m_iter, %block_m : f32
      %alpha_arg = arith.subf %m_iter, %m_new : f32
      %beta_arg = arith.subf %block_m, %m_new : f32
      // CHECK: math.exp
      %alpha = math.exp %alpha_arg : f32
      %beta = math.exp %beta_arg : f32
      %block_mv = vector.broadcast %block_m : f32 to vector<16xf32>
      %shifted = arith.subf %active_scores, %block_mv : vector<16xf32>
      %e = math.exp %shifted : vector<16xf32>
      %active_e = arith.select %mask, %e, %zeros : vector<16xi1>, vector<16xf32>
      // CHECK: vector.reduction <add>
      %block_l = vector.reduction <add>, %active_e : vector<16xf32> into f32
      %l_scaled = arith.mulf %l_iter, %alpha : f32
      %block_l_scaled = arith.mulf %block_l, %beta : f32
      %l_next = arith.addf %l_scaled, %block_l_scaled : f32
      %acc_scaled = arith.mulf %acc_iter, %alpha : f32
      %block_acc_scaled = arith.mulf %block_l, %beta : f32
      %acc_next = arith.addf %acc_scaled, %block_acc_scaled : f32
      scf.yield %m_new, %l_next, %acc_next : f32, f32, f32
    }
    // CHECK: arith.divf
    %outv = arith.divf %state#2, %state#1 : f32
    memref.store %outv, %out[%q] : memref<?xf32, #vc4value.global>
    return
  }
}
