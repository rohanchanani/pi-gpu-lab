// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_sparse_transfer_write(
    %out: memref<64xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite"} {
  %zero_v = arith.constant dense<0.000000e+00> : vector<16xf32>
  %value = arith.constant dense<1.000000e+00> : vector<16xf32>
  %mask = arith.cmpf ogt, %value, %zero_v : vector<16xf32>
  vector.transfer_write %value, %out[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<64xf32, #vc4value.global>
  return
}

// CHECK: sparse or unknown transfer_write mask is not Phase 5 lowerable
// CHECK: staged value-surface feature
// CHECK: READY_FOR_TRITON remains NO
