// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @missing_grid_rank() attributes {vc4value.kernel} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @bad_grid_rank_zero() attributes {vc4value.kernel, vc4value.grid_rank = 0 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @bad_grid_rank_type() attributes {vc4value.kernel, vc4value.grid_rank = "one"} {
    return
  }
}

// -----

// expected-error @+1 {{vc4value kernel metadata is only legal on func.func}}
builtin.module attributes {vc4value.grid_rank = 1 : i32} {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}
