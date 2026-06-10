// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @mask_memory_load_other_zero(
    %in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %base: index {vc4value.arg_name = "base"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0.000000e+00 : f32
  %mask = vector.create_mask %n : vector<16xi1>
  %v = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @mask_memory_load_other_zero
// CHECK: vc4kernel.pred.tail
// CHECK: vc4kernel.tmu_load_fragment
// CHECK-SAME: inactive_load = #vc4kernel.inactive_load<zero>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
