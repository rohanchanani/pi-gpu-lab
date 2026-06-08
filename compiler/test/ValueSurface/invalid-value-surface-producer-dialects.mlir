// RUN: vc4-opt %s --allow-unregistered-dialect --vc4-verify-value-surface -verify-diagnostics -split-input-file

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
  // expected-error @+1 {{producer dialect is not legal in the VC4 value surface}}
  "ttg.some_op"() : () -> ()
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{producer dialect is not legal in the VC4 value surface}}
  "gpu.block_id"() : () -> index
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{producer dialect is not legal in the VC4 value surface}}
  "linalg.matmul"() : () -> ()
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{producer dialect is not legal in the VC4 value surface}}
  "nvgpu.some_op"() : () -> ()
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  // expected-error @+1 {{producer dialect is not legal in the VC4 value surface}}
  "nvvm.some_op"() : () -> ()
}
