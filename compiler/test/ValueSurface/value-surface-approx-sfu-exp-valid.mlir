// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @approx_sfu_exp
  func.func @approx_sfu_exp()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite"} {
    %x = arith.constant dense<0.000000e+00> : vector<16xf32>
    // CHECK: math.exp
    %y = math.exp %x : vector<16xf32>
    %s = arith.constant 0.000000e+00 : f32
    // CHECK: math.exp
    %e = math.exp %s : f32
    return
  }
}
