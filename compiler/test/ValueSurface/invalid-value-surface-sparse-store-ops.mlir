// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  func.func @kernel(%out: memref<16xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.000000e+00 : f32
    %idx = vector.step : vector<16xindex>
    %mask = vector.create_mask %c16 : vector<16xi1>
    %value = vector.broadcast %zero : f32 to vector<16xf32>
    // expected-error @+1 {{sparse-store-shaped vector operation is not legal in the VC4 value surface}}
    vector.scatter %out[%c0][%idx], %mask, %value {alignment = 4 : i64} : memref<16xf32>, vector<16xindex>, vector<16xi1>, vector<16xf32>
    return
  }
}

// -----

builtin.module {
  func.func @kernel(%out: memref<16xf32>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %zero = arith.constant 0.000000e+00 : f32
    %mask = vector.create_mask %c16 : vector<16xi1>
    %value = vector.broadcast %zero : f32 to vector<16xf32>
    // expected-error @+1 {{sparse-store-shaped vector operation is not legal in the VC4 value surface}}
    vector.compressstore %out[%c0], %mask, %value {alignment = 4 : i64} : memref<16xf32>, vector<16xi1>, vector<16xf32>
    return
  }
}
