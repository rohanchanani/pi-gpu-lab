// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @empty(
    %x: memref<64xi32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"},
    %y: memref<64xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out"},
    %n: index {vc4value.arg_name = "n"},
    %alpha: f32 {vc4value.arg_name = "alpha"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  return
}

// CHECK: vc4kernel.kernel @empty
// CHECK-SAME: arg_attrs =
// CHECK-SAME: schedule_mode = #vc4kernel.schedule_mode<independent_vector>
// CHECK-SAME: warps_per_block = 1
// CHECK: vc4kernel.return
// CHECK-NOT: func.func
// CHECK-NOT: vc4value
// CHECK-NOT: vector.transfer
// CHECK-NOT: memref.
