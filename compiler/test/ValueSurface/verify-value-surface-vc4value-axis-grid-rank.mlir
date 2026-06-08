// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }

  func.func @helper() {
    // expected-error @+1 {{vc4value launch identity ops require an enclosing vc4value.kernel function}}
    %pid = vc4value.program_id {axis = 0 : i32} : index
    return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{vc4value launch axis must be within the enclosing vc4value.grid_rank}}
    %pid = vc4value.program_id {axis = 1 : i32} : index
    return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
    // expected-error @+1 {{vc4value launch axis must be within the enclosing vc4value.grid_rank}}
    %np = vc4value.num_programs {axis = 2 : i32} : index
    return
  }
}
