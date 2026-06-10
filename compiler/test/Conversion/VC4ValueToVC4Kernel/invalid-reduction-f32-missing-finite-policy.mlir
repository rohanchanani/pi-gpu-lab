// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_reduction_f32_missing_finite_policy()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %v = arith.constant dense<1.000000e+00> : vector<16xf32>
  %sum = vector.reduction <add>, %v : vector<16xf32> into f32
  return
}

// CHECK: f32 vector.reduction requires explicit finite-tree policy
