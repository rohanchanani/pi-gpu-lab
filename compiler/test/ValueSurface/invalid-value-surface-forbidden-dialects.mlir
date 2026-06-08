// RUN: vc4-opt %s --allow-unregistered-dialect --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{dialect 'foo' is not legal in the VC4 value surface}}
  "foo.some_op"() : () -> ()
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{producer dialect is not legal in the VC4 value surface}}
  "tt.get_program_id"() : () -> index
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{target/lower-half dialect is not legal in the VC4 value surface}}
  %v = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
}
