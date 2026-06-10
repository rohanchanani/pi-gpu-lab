// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @reduction_f32_finite_add_lowers(
    %in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<16xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %in[%base], %zero {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
  %sum = vector.reduction <add>, %v : vector<16xf32> into f32
  memref.store %sum, %out[%base] : memref<16xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @reduction_f32_finite_add_lowers
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>
// CHECK-SAME: kind = #vc4kernel.reduce<add>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
