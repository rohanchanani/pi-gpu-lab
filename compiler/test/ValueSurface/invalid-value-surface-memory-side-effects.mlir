// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  func.func @kernel(%in: memref<16xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    %v = memref.load %in[%c0] : memref<16xf32, #vc4value.global>
    return
  }
}

// -----

builtin.module {
  func.func @kernel(%out: memref<4x16xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f32
    // expected-error @+1 {{direct memref.store is only legal for Phase 12 scalar i32/f32 reduction-output stores to rank-1 #vc4value.global memrefs}}
    memref.store %zero, %out[%c0, %c0] : memref<4x16xf32, #vc4value.global>
    return
  }
}

// -----

builtin.module {
  func.func @kernel(
      %src: memref<16xf32, #vc4value.global> {vc4value.arg_name = "src", vc4value.direction = "in"},
      %dst: memref<16xf32, #vc4value.global> {vc4value.arg_name = "dst", vc4value.direction = "out"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    memref.copy %src, %dst : memref<16xf32, #vc4value.global> to memref<16xf32, #vc4value.global>
    return
  }
}

// -----

builtin.module {
  func.func @kernel(%m: memref<16xi32, #vc4value.global> {vc4value.arg_name = "m", vc4value.direction = "inout"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %v = arith.constant 1 : i32
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    %old = memref.atomic_rmw addi %v, %m[%c0] : (i32, memref<16xi32, #vc4value.global>) -> i32
    return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  func.func @dma_ops(%src: memref<16xf32, 0>, %dst: memref<16xf32, 1>, %tag: memref<1xi32, 2>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    memref.dma_start %src[%c0], %dst[%c0], %c16, %tag[%c0] : memref<16xf32, 0>, memref<16xf32, 1>, memref<1xi32, 2>
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    memref.dma_wait %tag[%c0], %c16 : memref<1xi32, 2>
    return
  }
}
