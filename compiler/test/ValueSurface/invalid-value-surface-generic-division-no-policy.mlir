// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @generic_division_no_policy()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %num = arith.constant dense<1.000000e+00> : vector<16xf32>
    %den = arith.constant dense<2.000000e+00> : vector<16xf32>
    // expected-error @+1 {{Phase 15 approximate SFU math requires explicit vc4value.math_policy = "approx_sfu" and finite vc4value.fp_domain; exact/default math is staged}}
    %y = arith.divf %num, %den : vector<16xf32>
    return
  }
}
