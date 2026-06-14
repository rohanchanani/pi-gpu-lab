// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @approx_sfu_sqrt
  func.func @approx_sfu_sqrt()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite_positive"} {
    %x = arith.constant dense<4.000000e+00> : vector<16xf32>
    // CHECK: math.sqrt
    %sqrt = math.sqrt %x : vector<16xf32>
    return
  }
}
