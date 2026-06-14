// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @finite_f32_max_reduction
  func.func @finite_f32_max_reduction()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree",
                  vc4value.max_policy = "finite"} {
    %v = arith.constant dense<0.000000e+00> : vector<16xf32>
    // CHECK: vector.reduction <maxnumf>
    %m = vector.reduction <maxnumf>, %v : vector<16xf32> into f32
    return
  }
}
