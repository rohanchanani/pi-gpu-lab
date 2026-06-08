// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  // expected-error @+1 {{scalable vector types are not legal in the VC4 value surface}}
  func.func @kernel(%v: vector<[16]xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}
