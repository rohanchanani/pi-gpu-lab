// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_nonzero_load_other(
    %in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %base: index {vc4value.arg_name = "base"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %other = arith.constant 1.000000e+00 : f32
  %mask = vector.create_mask %n : vector<16xi1>
  %v = vector.transfer_read %in[%base], %other, %mask {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
  return
}

// CHECK: transfer_read nonzero padding value
// CHECK: transfer_read padding must be zero for inactive_load<zero>
// CHECK: READY_FOR_TRITON remains NO
