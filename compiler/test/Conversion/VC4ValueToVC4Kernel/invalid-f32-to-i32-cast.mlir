// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @f32_to_i32_cast()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %v = arith.constant dense<0.000000e+00> : vector<16xf32>
  %cast = arith.fptosi %v : vector<16xf32> to vector<16xi32>
  return
}

// CHECK: fp-to-int numeric cast is staged
// CHECK: READY_FOR_TRITON remains NO
