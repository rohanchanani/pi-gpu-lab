// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  // expected-error @+1 {{vector element type is not legal in the VC4 value surface}}
  func.func @f64_vector(%v: vector<16xf64>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vector element type is not legal in the VC4 value surface}}
  func.func @i64_vector(%v: vector<16xi64>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}
