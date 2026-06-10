// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

// This is value-surface admissible only. The sparse transfer mask is staged for
// Phase 10 value-to-VC4Kernel lowering and must not be treated as accepted
// memory lowering by this test.
builtin.module {
  // CHECK-LABEL: func.func @sparse_transfer_mask_surface
  func.func @sparse_transfer_mask_surface(
      %in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
      %out: memref<64xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
      %limit: f32 {vc4value.arg_name = "limit"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f32
    %v = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
    %limit_v = vector.broadcast %limit : f32 to vector<16xf32>
    // CHECK: arith.cmpf
    %sparse_mask = arith.cmpf ogt, %v, %limit_v : vector<16xf32>
    // CHECK: vector.transfer_write
    vector.transfer_write %v, %out[%c0], %sparse_mask {in_bounds = [true]} : vector<16xf32>, memref<64xf32, #vc4value.global>
    return
  }
}
