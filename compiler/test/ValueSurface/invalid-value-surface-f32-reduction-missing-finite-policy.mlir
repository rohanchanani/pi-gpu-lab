// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @missing_finite_policy()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %v = arith.constant dense<0.000000e+00> : vector<16xf32>
    // expected-error @+1 {{f32 vector.reduction <add> requires explicit vc4value.fp_domain = "finite" and vc4value.reduction_policy = "finite_tree" in Phase 12}}
    %sum = vector.reduction <add>, %v : vector<16xf32> into f32
    return
  }
}
