// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

// expected-error @+1 {{multiblock softmax is staged in Phase 15}}
func.func @invalid_multiblock_softmax_staged()
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.softmax_v0 = "multiblock"} {
  return
}
