// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }

  func.func @helper_program_id() {
    // expected-error @+1 {{vc4value launch identity ops require an enclosing vc4value.kernel function}}
    %pid = vc4value.program_id {axis = 0 : i32} : index
    return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }

  func.func @helper_num_programs() {
    // expected-error @+1 {{vc4value launch identity ops require an enclosing vc4value.kernel function}}
    %np = vc4value.num_programs {axis = 0 : i32} : index
    return
  }
}
