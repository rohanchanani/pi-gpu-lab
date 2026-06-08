// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  // expected-error @+1 {{scalable vector types are not legal in the VC4 value surface}}
  func.func @scalable_vector(%v: vector<[16]xf32>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}
