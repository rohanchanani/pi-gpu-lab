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
  func.func @non_integer_grid_rank() attributes {vc4value.kernel, vc4value.grid_rank = "one"} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @zero_grid_rank() attributes {vc4value.kernel, vc4value.grid_rank = 0 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @four_grid_rank() attributes {vc4value.kernel, vc4value.grid_rank = 4 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @negative_grid_rank() attributes {vc4value.kernel, vc4value.grid_rank = -1 : i32} {
    return
  }
}
