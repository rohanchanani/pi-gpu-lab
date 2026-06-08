// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{target/lower-half dialect is not legal in the VC4 value surface}}
  vc4kernel.kernel @bad attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // expected-error @+1 {{target/lower-half dialect is not legal in the VC4 value surface}}
    vc4kernel.return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{target/lower-half dialect is not legal in the VC4 value surface}}
  %v = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{target/lower-half dialect is not legal in the VC4 value surface}}
  vc4.module @bad {
  }
}
