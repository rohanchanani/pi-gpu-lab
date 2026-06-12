// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

func.func @invalid_i8_transfer(
    %in: memref<64xi8, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0 : i8
  %v = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<64xi8, #vc4value.global>, vector<16xi8>
  return
}

// CHECK: int8/int16 quantized storage is staged
// CHECK: READY_FOR_TRITON remains NO
