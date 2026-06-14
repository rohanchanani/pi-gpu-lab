// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @approx_sfu_recip_div
  func.func @approx_sfu_recip_div()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite_nonzero"} {
    %num = arith.constant dense<1.000000e+00> : vector<16xf32>
    %den = arith.constant dense<2.000000e+00> : vector<16xf32>
    // CHECK: arith.divf
    %y = arith.divf %num, %den : vector<16xf32>
    return
  }
}
