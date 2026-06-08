// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

func.func @write_i8(%out: memref<64xi8, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %value = arith.constant dense<1> : vector<16xi8>
  vector.transfer_write %value, %out[%c0] {in_bounds = [true]} : vector<16xi8>, memref<64xi8, #vc4value.global>
  return
}

// CHECK: expected rank-1 contiguous i32/f32 #vc4value.global memref
// CHECK: not Phase 5 lowerable

// -----

func.func @write_f16(%out: memref<64xf16, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %value = arith.constant dense<1.000000e+00> : vector<16xf16>
  vector.transfer_write %value, %out[%c0] {in_bounds = [true]} : vector<16xf16>, memref<64xf16, #vc4value.global>
  return
}

// CHECK: expected rank-1 contiguous i32/f32 #vc4value.global memref
// CHECK: staged value-surface feature
