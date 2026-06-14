// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @finite_f32_max_reduction_lowers(
    %out: memref<16xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %idx: index {vc4value.arg_name = "idx"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree",
                vc4value.max_policy = "finite"} {
  %v = arith.constant dense<1.000000e+00> : vector<16xf32>
  %max0 = vector.reduction <maxnumf>, %v : vector<16xf32> into f32
  %max1 = vector.reduction <maximumf>, %v : vector<16xf32> into f32
  memref.store %max0, %out[%idx] : memref<16xf32, #vc4value.global>
  memref.store %max1, %out[%idx] : memref<16xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @finite_f32_max_reduction_lowers
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: fp_policy = #vc4kernel.fp_reduce_policy<finite_tree>
// CHECK-SAME: kind = #vc4kernel.reduce<fmax>
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: kind = #vc4kernel.reduce<fmax>
// CHECK-NOT: vector.
// CHECK-NOT: memref.
