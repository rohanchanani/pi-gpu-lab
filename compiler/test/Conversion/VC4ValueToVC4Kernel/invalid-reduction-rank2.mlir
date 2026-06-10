// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_reduction_rank2()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %v = arith.constant dense<1> : vector<32xi32>
  %sum = vector.reduction <add>, %v : vector<32xi32> into i32
  return
}

// CHECK: constant result type outside Phase 5 subset
