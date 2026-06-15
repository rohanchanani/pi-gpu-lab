// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @attention_apply_v0_f16_storage_inputs
  func.func @attention_apply_v0_f16_storage_inputs(
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
                  vc4value.attention_apply_v0 = "precomputed_transposed_v_active_1_to_16",
                  vc4value.softmax_v0 = "one_block_active_1_to_16",
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree",
                  vc4value.max_policy = "finite",
                  vc4value.f16_storage_policy = "finite"} {
    %q = vc4value.program_id {axis = 0 : i32} : index
    %d = vc4value.program_id {axis = 1 : i32} : index
    %sbase = arith.muli %q, %lds : index
    %vbase = arith.muli %d, %ldv : index
    %obase = arith.muli %q, %ldo : index
    %oidx = arith.addi %obase, %d : index
    %mask = vector.create_mask %k : vector<16xi1>
    %zero_h = arith.constant 0.000000e+00 : f16
    %zero = arith.constant 0.000000e+00 : f32
    %scores_h = vector.transfer_read %scores[%sbase], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
    // CHECK: arith.extf
    %scores_v = arith.extf %scores_h : vector<16xf16> to vector<16xf32>
    %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
    %zeros = arith.constant dense<0.000000e+00> : vector<16xf32>
    %active_scores = arith.select %mask, %scores_v, %low : vector<16xi1>, vector<16xf32>
    %max = vector.reduction <maxnumf>, %active_scores : vector<16xf32> into f32
    %maxv = vector.broadcast %max : f32 to vector<16xf32>
    %shifted = arith.subf %active_scores, %maxv : vector<16xf32>
    %e = math.exp %shifted : vector<16xf32>
    %active_e = arith.select %mask, %e, %zeros : vector<16xi1>, vector<16xf32>
    %denom = vector.reduction <add>, %active_e : vector<16xf32> into f32
    %one = arith.constant 1.000000e+00 : f32
    %inv = arith.divf %one, %denom : f32
    %invv = vector.broadcast %inv : f32 to vector<16xf32>
    %probs = arith.mulf %active_e, %invv : vector<16xf32>
    %v_h = vector.transfer_read %vt[%vbase], %zero_h, %mask {in_bounds = [true]} : memref<?xf16, #vc4value.global>, vector<16xf16>
    %v = arith.extf %v_h : vector<16xf16> to vector<16xf32>
    %weighted = arith.mulf %probs, %v : vector<16xf32>
    %acc = vector.reduction <add>, %weighted : vector<16xf32> into f32
    memref.store %acc, %out[%oidx] : memref<?xf32, #vc4value.global>
    return
  }
}
