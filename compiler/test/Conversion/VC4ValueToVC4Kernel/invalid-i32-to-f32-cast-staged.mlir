// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @i32_to_f32_cast_staged()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %v = arith.constant dense<1> : vector<16xi32>
  %cast = arith.sitofp %v : vector<16xi32> to vector<16xf32>
  return
}

// CHECK: i32 to f32 numeric cast staged by lower-half gap
// CHECK: READY_FOR_TRITON remains NO
