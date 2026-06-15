// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  // expected-error @+1 {{online/multiblock attention-apply softmax is staged in Phase 16}}
  func.func @attention_apply_multiblock_staged()
      attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                  vc4value.attention_apply_v0 = "multiblock",
                  vc4value.math_policy = "approx_sfu",
                  vc4value.fp_domain = "finite"} {
    return
  }
}
