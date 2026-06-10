// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_vector_multi_reduction()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %v = arith.constant dense<1> : vector<16xi32>
  %acc = arith.constant 0 : i32
  %sum = vector.multi_reduction <add>, %v, %acc [0] : vector<16xi32> to i32
  return
}

// CHECK: vector.multi_reduction is staged
