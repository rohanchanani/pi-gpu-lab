// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @gemv_dot_missing_f32_policy()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %a = arith.constant dense<1.000000e+00> : vector<16xf32>
    %x = arith.constant dense<2.000000e+00> : vector<16xf32>
    %prod = arith.mulf %a, %x : vector<16xf32>
    // expected-error @+1 {{f32 vector.reduction <add> requires explicit vc4value.fp_domain = "finite" and vc4value.reduction_policy = "finite_tree" in Phase 12}}
    %dot = vector.reduction <add>, %prod : vector<16xf32> into f32
    return
  }
}
