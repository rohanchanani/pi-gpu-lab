// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_gemv_exact_f32_policy()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %a = arith.constant dense<1.000000e+00> : vector<16xf32>
  %x = arith.constant dense<2.000000e+00> : vector<16xf32>
  %prod = arith.mulf %a, %x : vector<16xf32>
  %dot = vector.reduction <add>, %prod : vector<16xf32> into f32
  return
}

// CHECK: exact f32 dot requires unsupported exact reduction policy
