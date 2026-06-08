// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  // expected-error @+1 {{scalable vectors are not legal in the VC4 value surface}}
  func.func @scalable(%v: vector<[16]xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vector rank greater than 2 is not legal in the VC4 value surface}}
  func.func @vector_rank(%v: vector<2x2x2xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{unsupported vector element type in VC4 value surface}}
  func.func @vector_element(%v: vector<16xi64>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{unranked memrefs are not legal in the VC4 value surface}}
  func.func @unranked_memref(%m: memref<*xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{memref rank greater than 2 is not legal in the VC4 value surface}}
  func.func @memref_rank(%m: memref<1x1x1xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{unsupported memref element type in VC4 value surface}}
  func.func @memref_element(%m: memref<16xi64>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{tensor types are not legal in the VC4 value surface}}
  func.func @tensor_type(%t: tensor<16xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{complex types are not legal in the VC4 value surface}}
  func.func @complex_type(%c: complex<f32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}
