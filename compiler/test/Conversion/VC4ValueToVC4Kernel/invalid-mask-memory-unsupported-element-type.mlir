// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

func.func @invalid_i8_transfer(
    %in: memref<64xi8, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0 : i8
  %v = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xi8, #vc4value.global>, vector<16xi8>
  return
}

// CHECK: unsupported transfer element type
// CHECK: expected rank-1 contiguous i32/f32 #vc4value.global memref

// -----

func.func @invalid_f16_transfer(
    %out: memref<64xf16, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %value = arith.constant dense<0.000000e+00> : vector<16xf16>
  vector.transfer_write %value, %out[%c0] {in_bounds = [true]} : vector<16xf16>, memref<64xf16, #vc4value.global>
  return
}

// CHECK: unsupported transfer element type
// CHECK: READY_FOR_TRITON remains NO
