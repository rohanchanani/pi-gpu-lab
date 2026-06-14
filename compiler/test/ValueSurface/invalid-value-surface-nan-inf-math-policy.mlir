// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @nan_inf_math_policy()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "nan_inf"} {
    %x = arith.constant dense<0.000000e+00> : vector<16xf32>
    // expected-error @+1 {{Phase 15 approximate SFU math requires explicit vc4value.math_policy = "approx_sfu" and finite vc4value.fp_domain; exact/default math is staged}}
    %y = math.exp %x : vector<16xf32>
    return
  }
}
