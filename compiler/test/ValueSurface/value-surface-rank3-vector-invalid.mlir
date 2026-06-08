// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  // expected-error @+1 {{vector rank greater than 2 is not legal in the Phase 3.5 VC4 value surface}}
  func.func @rank3_vector(%v: vector<2x4x16xf32>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}
