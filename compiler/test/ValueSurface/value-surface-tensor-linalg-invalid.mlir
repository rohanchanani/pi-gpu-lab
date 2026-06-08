// RUN: vc4-opt %s --vc4-verify-value-surface --allow-unregistered-dialect -verify-diagnostics -split-input-file

builtin.module {
  // expected-error @+1 {{tensor types are not legal in the initial VC4 value surface}}
  func.func @tensor_type(%t: tensor<16xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  func.func @linalg_op() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{producer dialect is not legal in the VC4 value surface}}
    "linalg.matmul"() : () -> ()
    return
  }
}
