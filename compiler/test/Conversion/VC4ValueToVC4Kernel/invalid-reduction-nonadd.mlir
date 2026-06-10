// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_reduction_nonadd()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %v = arith.constant dense<1> : vector<16xi32>
  %sum = vector.reduction <mul>, %v : vector<16xi32> into i32
  return
}

// CHECK: non-add vector.reduction is staged
