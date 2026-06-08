// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @read_nonzero_padding(%in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 1.000000e+00 : f32
  %v = vector.transfer_read %in[%c0], %pad {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
  return
}

// CHECK: transfer_read padding must be zero for inactive_load<zero>
// CHECK: not Phase 5 lowerable
// CHECK: READY_FOR_TRITON remains NO
