// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  // expected-error @+1 {{non-transposed V gather/lane-varying stride is staged in Phase 16 attention-apply v0}}
  func.func @attention_apply_nontransposed_gather_staged()
      attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                  vc4value.attention_apply_v0 = "nontransposed_v_gather",
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite"} {
    return
  }
}
