// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

func.func @write_i8(%out: memref<64xi8, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %value = arith.constant dense<1> : vector<16xi8>
  vector.transfer_write %value, %out[%c0] {in_bounds = [true]} : vector<16xi8>, memref<64xi8, #vc4value.global>
  return
}

// CHECK: int8/int16 quantized storage is staged
// CHECK: READY_FOR_TRITON remains NO
