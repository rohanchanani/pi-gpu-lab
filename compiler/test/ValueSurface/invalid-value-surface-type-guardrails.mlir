// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  // expected-error @+1 {{scalable vector types are not legal in the VC4 value surface}}
  func.func @scalable(%v: vector<[16]xf32>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vector rank greater than 2 is not legal in the Phase 3.5 VC4 value surface}}
  func.func @vector_rank(%v: vector<2x2x2xf32>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vector element type is not legal in the VC4 value surface}}
  func.func @vector_element(%v: vector<16xi64>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vector element type is not legal in the VC4 value surface}}
  func.func @vector_f64_element(%v: vector<7xf64>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{scalar type is not legal in the VC4 value surface}}
  func.func @scalar_i64(%x: i64) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{scalar type is not legal in the VC4 value surface}}
  func.func @scalar_f64(%x: f64) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{unranked memrefs are not legal in the VC4 value surface}}
  func.func @unranked_memref(%m: memref<*xf32>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{memref rank greater than 2 is not legal in the VC4 value surface}}
  func.func @memref_rank(%m: memref<1x1x1xf32>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{unsupported memref element type in VC4 value surface}}
  func.func @memref_element(%m: memref<16xi64>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{unsupported memref element type in VC4 value surface}}
  func.func @memref_i1_element(%m: memref<16xi1>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{tensor types are not legal in the initial VC4 value surface}}
  func.func @tensor_type(%t: tensor<16xf32>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{complex types are not legal in the VC4 value surface}}
  func.func @complex_type(%c: complex<f32>) {
    return
  }
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  func.func @native_f16_arith() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %a = arith.constant dense<0.000000e+00> : vector<16xf16>
    %b = arith.constant dense<1.000000e+00> : vector<16xf16>
    // expected-error @+1 {{native f16 arithmetic is not legal in the Phase 3.5 VC4 value surface}}
    %sum = arith.addf %a, %b : vector<16xf16>
    return
  }
}
