// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  func.func @axis_equal_grid_rank_rejects()
      attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
    // expected-error @+1 {{vc4value launch axis must be within the enclosing vc4value.grid_rank}}
    %pid = vc4value.program_id {axis = 2 : i32} : index
    return
  }
}

// -----

builtin.module {
  func.func @num_programs_axis_equal_grid_rank_rejects()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{vc4value launch axis must be within the enclosing vc4value.grid_rank}}
    %np = vc4value.num_programs {axis = 1 : i32} : index
    return
  }
}

// -----

builtin.module {
  func.func @axis_greater_than_two_rejects()
      attributes {vc4value.kernel, vc4value.grid_rank = 3 : i32} {
    // expected-error @+1 {{axis must be 0, 1, or 2}}
    %pid = vc4value.program_id {axis = 3 : i32} : index
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @grid_rank_zero_rejects()
      attributes {vc4value.kernel, vc4value.grid_rank = 0 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @grid_rank_four_rejects()
      attributes {vc4value.kernel, vc4value.grid_rank = 4 : i32} {
    return
  }
}
