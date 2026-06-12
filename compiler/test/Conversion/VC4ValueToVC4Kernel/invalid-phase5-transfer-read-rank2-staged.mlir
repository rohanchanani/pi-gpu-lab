// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @read_rank2(%in: memref<4x16xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %in[%c0, %c0], %zero {in_bounds = [true]} : memref<4x16xf32, #vc4value.global>, vector<16xf32>
  return
}

// CHECK: expected rank-1 contiguous i32/f32/f16 #vc4value.global memref
// CHECK: not Phase 5 lowerable
// CHECK: staged value-surface feature
// CHECK: READY_FOR_TRITON remains NO
