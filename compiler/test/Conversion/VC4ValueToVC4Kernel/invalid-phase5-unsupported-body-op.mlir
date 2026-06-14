// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @reduction()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %x = arith.constant 1.000000e+00 : f32
  %y = math.sqrt %x : f32
  return
}

// CHECK: math dialect operation
// CHECK: staged value-surface feature
// CHECK: READY_FOR_TRITON remains NO
