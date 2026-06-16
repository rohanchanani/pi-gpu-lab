// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  // expected-error @+1 {{active-count-zero softmax is staged without an explicit finite no-op guard in Phase 15}}
  func.func @online_softmax_k_zero_staged()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.softmax_v0 = "zero_active",
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite"} {
    return
  }
}
