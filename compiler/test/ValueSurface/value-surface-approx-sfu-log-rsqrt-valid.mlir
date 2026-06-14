// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @approx_sfu_log_rsqrt
  func.func @approx_sfu_log_rsqrt()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite_positive"} {
    %x = arith.constant dense<1.000000e+00> : vector<16xf32>
    // CHECK: math.log
    %log = math.log %x : vector<16xf32>
    // CHECK: math.rsqrt
    %rsqrt = math.rsqrt %x : vector<16xf32>
    return
  }
}
